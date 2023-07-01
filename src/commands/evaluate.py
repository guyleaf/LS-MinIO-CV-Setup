import json
import os
import random
import tempfile
from typing import Annotated, Optional, Union

import minio
import numpy as np
import torch
import typer
from label_studio_sdk import Client, Project
from label_studio_sdk.data_manager import Column, Filters, Operator, Type
from rich.progress import track
from rich.prompt import Prompt
from rich.table import Table
from torch.utils.data import DataLoader

from .. import settings
from ..datasets.weather import WeatherImagesDataset
from ..models.weather import WeatherModel
from ..utils import (
    check_ls_connection,
    connect_minio,
    console,
    count_tasks_in_bucket,
    err_console,
    get_storages,
)
from .create import create_project, import_data
from .utils import validate_path

#################################################################################


# Argument type hints
_PROJECT_NAME_ARGUMENT = Annotated[
    str,
    typer.Argument(help="The name of label studio project"),
]

_LABEL_CONFIG_ARGUMENT = Annotated[
    str,
    typer.Argument(help="The path of label config", callback=validate_path),
]

_BUCKETS_ARGUMENT = Annotated[
    list[str],
    typer.Option(
        help="The list of buckets you'd like to import. If it is empty, import all buckets",
    ),
]


#################################################################################


def make_clients(
    apps: list[int],
) -> tuple[minio.Minio, list[Union[Client, None]], Client]:
    minio_client = connect_minio(
        settings.MINIO_HOST, settings.MINIO_ROOT_USER, settings.MINIO_ROOT_PASSWORD
    )

    ls_clients = [
        Client(
            f"{settings.LABEL_STUDIO_HOST}/{app_id}", settings.LABEL_STUDIO_USER_TOKEN
        )
        if app_id in apps
        else None
        for app_id in range(1, settings.NUM_LABEL_STUDIO_APPS + 1)
    ]
    ls_review_client = Client(
        f"{settings.LABEL_STUDIO_HOST}/0", settings.LABEL_STUDIO_USER_TOKEN
    )
    return minio_client, ls_clients, ls_review_client


def make_projects_table(projects_map: list[dict], title: str = "Projects") -> Table:
    table = Table(title=title)
    table.add_column("#")
    table.add_column("Name", style="cyan")
    for app_id in range(1, settings.NUM_LABEL_STUDIO_APPS + 1):
        table.add_column(f"ID in app-{app_id}", style="green")

    for row_id, project_map in enumerate(projects_map):
        table.add_row(str(row_id), project_map["name"], *map(str, project_map["ids"]))
    return table


def select_project(ls_clients: list[Union[Client, None]]) -> list[Union[Project, None]]:
    if not os.path.exists(settings.PROJECTS_MAP):
        raise RuntimeError(f"Cannot find projects map {settings.PROJECTS_MAP}.")
    with open(settings.PROJECTS_MAP, "r") as f:
        projects_map: list[dict] = json.load(f)

    if len(projects_map) == 0:
        raise RuntimeError("You haven't created any project.")

    # select project from table
    table = make_projects_table(projects_map)
    console.print(table, justify="center")

    while True:
        row_id = Prompt.ask(
            "Which project you'd like to evaluate?",
            console=console,
            default=str(len(projects_map) - 1),
        )
        row_id = int(row_id)

        if row_id >= 0 and row_id < table.row_count:
            break
        console.print("[red]Please select one of the available options.")

    project_name = projects_map[row_id]["name"]
    console.log(f"You selected {project_name} project.")

    # get projects by id
    projects = []
    project_ids = projects_map[row_id]["ids"]
    for project_id, ls_client in zip(project_ids, ls_clients):
        if ls_client is None:
            projects.append(None)
        else:
            check_ls_connection(ls_client)
            projects.append(ls_client.get_project(project_id))
    return projects


def are_inner_id_and_data_id_same(tasks: list[dict]) -> bool:
    for task in tasks:
        if task["inner_id"] != task["data"]["data_id"]:
            return False
    return True


def are_annotations_correct(tasks: list[dict]) -> list[str]:
    errors = []
    for task in tasks:
        task_id = task["id"]
        annotations = task["annotations"]
        if len(annotations) > 1:
            errors.append(f"\[task-{task_id}]The annotation should be only one.")

        intensities = []
        for result in annotations[0]["result"]:
            name: str = result["from_name"]
            if name.startswith("intensity"):
                intensities.append(result["value"]["choices"][0])

        if len(intensities) != len(tuple(intensities)):
            errors.append(f"\[task-{task_id}]The intensity choices should be unique.")

    return errors


def validate_annotations(
    projects: list[Union[Project, None]], minio_client: minio.Minio
):
    status = console.status("[yellow]Validating annotations...")

    total_tasks = 0
    has_error = False
    for app_id, project in enumerate(projects, start=1):
        if project is None:
            continue

        status.start()
        console.log(f"Validating app-{app_id} annotations...")

        check_ls_connection(project, show_status=False)
        if total_tasks == 0:
            # count total tasks in connected storages
            storages = get_storages(project)
            for storage in storages:
                total_tasks += count_tasks_in_bucket(minio_client, storage["title"])

        errors = []

        # validate task count
        tasks = project.tasks
        if len(tasks) != total_tasks:
            errors.append(
                "Inconsistent task count between storages and ls. Maybe the annotator deletes or creates tasks."
            )

        # validate if inner id and data id are same
        if not are_inner_id_and_data_id_same(tasks):
            errors.append(
                "Inconsistent inner id and data id. Maybe the annotator modify the task order."
            )

        # validate if tasks are labeled
        total_unlabeled_tasks = len(project.get_unlabeled_tasks())
        if total_unlabeled_tasks != 0:
            errors.append(
                f"There are {total_unlabeled_tasks} unlabeled tasks in project."
            )

        # task-specific: check annotation rules
        if len(errors) == 0:
            error_msgs = are_annotations_correct(tasks)
            if len(error_msgs) != 0:
                errors.append("Invalid annotations.")
                errors.extend(error_msgs)

        status.stop()

        if len(errors) != 0:
            has_error = True
            err_console.log("[red]Validation errors:")
            for i, error in enumerate(errors, start=1):
                err_console.log(f"  {i}. {error}")

    if has_error:
        err_console.log("[bold red]Validation Error, please check the errors above.")
        typer.confirm("Do you want to continue?", default=False, abort=True)


def collect_storage_files(samples: list[dict]) -> dict[str, list[str]]:
    storage_files = {}
    for sample in samples:
        pattern: list[str] = storage_files.setdefault(
            sample["data"]["metadata"]["bucket_name"], []
        )
        pattern.append(sample["storage_filename"])
    return storage_files


def collect_storage_images(samples: list[dict]) -> dict[str, list[str]]:
    storage_files = {}
    for sample in samples:
        pattern: list[str] = storage_files.setdefault(
            sample["data"]["metadata"]["bucket_name"], []
        )
        pattern.append(sample["data"]["metadata"]["object_name"])
    return storage_files


def review_annotations(
    projects: list[Union[Project, None]],
    num_review_samples: int,
    ls_review_client: Client,
    review_project_id: Optional[int],
    minio_client: minio.Minio,
):
    for project_ in projects:
        if project_ is not None:
            project = project_

    check_ls_connection(project)

    console.log("[yellow]Getting task informations...")

    if review_project_id is None:
        # random sample num_review_samples tasks
        tasks: list[dict] = project.tasks
        samples = random.sample(tasks, k=num_review_samples)

        console.log("[yellow]Creating projects...")

        # create project for review
        storage_files = collect_storage_files(samples)
        review_project = create_project(
            project.params["title"], project.params["label_config"], ls_review_client
        )

        # import data to review project
        for bucket, files in storage_files.items():
            regex_filter = "|".join(files)
            import_data(
                review_project,
                [bucket],
                minio_client,
                regex_filter=regex_filter,
                total_tasks=len(files),
            )
    else:
        review_project = ls_review_client.get_project(review_project_id)

    return review_project


def download_object_files(
    out_dir: str, minio_client: minio.Minio, storage_files: dict[str, list[str]]
):
    object_files = []
    for bucket, files in storage_files.items():
        for file in files:
            out_path = os.path.join(out_dir, file.split("/")[-1])
            minio_client.fget_object(bucket, file, out_path)
            object_files.append(out_path)
    return object_files


@torch.no_grad()
def predict_weathers(root_dir: str, ckpt_path: str) -> np.ndarray:
    device = torch.device("cuda")
    dataset = WeatherImagesDataset(root_dir)
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=4
    )

    model = WeatherModel(ckpt_path)
    model.eval()
    model = model.to(device)

    batch_predictions = []
    for images in track(dataloader, description="Predicting...", total=len(dataloader)):
        images: torch.Tensor
        images = images.to(device)
        batch_predictions.append(model(images).cpu())

    return torch.concatenate(batch_predictions).numpy()


def measure_intensity(root_dir: str) -> np.ndarray:
    pass

def evaluate_annotations(
    ckpt_path: str,
    review_project: Project,
    projects: list[Union[Project, None]],
    minio_client: minio.Minio,
):
    review_tasks: list[dict] = review_project.tasks
    storage_images = collect_storage_images(review_tasks)
    Filters.create(
        Filters.AND,
        [
            Filters.item(
                Column.data("data_id"),
                Operator.IN_LIST,
                Type.Number,
                Filters.value([task["data"]["data_id"] for task in review_tasks]),
            ),
        ],
    )

    tmp_dir = tempfile.TemporaryDirectory()

    # download images to temporary folder
    download_object_files(tmp_dir.name, minio_client, storage_images)

    weather_predictions = predict_weathers(tmp_dir.name, ckpt_path)
    measure_intensity(tmp_dir.name)

    console.print(weather_predictions)
    # for task in review_tasks:
    #     task["annotations"]

    tmp_dir.cleanup()


def evaluate(
    ckpt_path: str,
    num_review_samples: int = 50,
    apps: list[int] = list(range(1, settings.NUM_LABEL_STUDIO_APPS + 1)),
    review_project_id: Optional[int] = None,
    seed: int = 1234,
    validate: bool = True,
):
    if not os.path.exists(ckpt_path):
        raise typer.BadParameter("The checkpoint file does not exist.")

    random.seed(seed)
    minio_client, ls_clients, ls_review_client = make_clients(apps)
    projects = select_project(ls_clients)

    if validate:
        validate_annotations(projects, minio_client)

    review_project = review_annotations(
        projects, num_review_samples, ls_review_client, review_project_id, minio_client
    )

    Prompt.ask(
        "Press any key to continue if reviewer finished labeling...", console=console
    )

    evaluate_annotations(ckpt_path, review_project, projects, minio_client)
