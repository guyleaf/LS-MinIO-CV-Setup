import os
import random
import tempfile
from typing import Annotated, Optional, Union

import minio
import numpy as np
import pandas as pd
import typer
from cleanlab.multiannotator import (
    get_label_quality_multiannotator,
    get_label_quality_multiannotator_ensemble,
)
from label_studio_sdk import Client, Project
from label_studio_sdk.data_manager import Column, Filters, Operator, Type
from rich.align import Align
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .. import settings
from ..utils import (
    check_ls_connection,
    connect_minio,
    console,
    convert_df_to_rich_table,
    count_tasks_in_bucket,
    err_console,
    get_storages,
)
from ..works.intensity import (
    collect_intensity_level_from_tasks,
    predict_intensity_ensemble,
)
from ..works.weather import (
    collect_weather_from_tasks,
    predict_weathers_ensemble,
)
from .create import create_project, import_data
from .utils import make_ls_clients, select_project, validate_int_range, validate_path

#################################################################################


# Argument type hints
_CHECKPOINT_ARGUMENT = Annotated[
    str,
    typer.Argument(help="The path of model checkpoint", callback=validate_path),
]

_NUM_REVIEW_SAMPLES_ARGUMENT = Annotated[
    int,
    typer.Option(
        help="The number of review samples", callback=validate_int_range(1, 1000)
    ),
]

_APPS_ARGUMENT = Annotated[
    list[int],
    typer.Option(
        help="The list of apps you'd like to evaluate",
    ),
]

_REVIEW_PROJECT_ID_ARGUMENT = Annotated[
    Optional[int],
    typer.Option(
        help="Use the id of project in review app to evaluate",
    ),
]

_SEED_ARGUMENT = Annotated[
    int,
    typer.Option(
        help="The seed of sampling tasks",
    ),
]

_VALIDATE_ARGUMENT = Annotated[
    bool,
    typer.Option(
        help="Whether execute the validation step or not",
    ),
]


#################################################################################


def are_inner_id_and_data_id_same(tasks: list[dict]) -> bool:
    for task in tasks:
        if task["inner_id"] != task["data"]["data_id"]:
            return False
    return True


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
    console.print(len(object_files))
    return object_files


def collect_tasks(
    review_project: Project, projects: list[Union[Project, None]]
) -> tuple[list[dict], list[list[dict]]]:
    # sorted by data_id in ascending order
    ordering = [Column.data("data_id")]
    review_tasks: list[dict] = review_project.get_tasks(ordering=ordering)

    filters = Filters.create(
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

    annotators_tasks = []
    for project in projects:
        if project is None:
            annotators_tasks.append([])
            continue
        annotators_tasks.append(project.get_tasks(filters=filters, ordering=ordering))

    return review_tasks, annotators_tasks


def make_metric_table(layout: Layout, metric: dict[str, pd.DataFrame]) -> Layout:
    # reorder by worst quality
    label_quality = metric["label_quality"].sort_values("consensus_quality_score")
    detailed_label_quality = metric["detailed_label_quality"].iloc[label_quality.index]

    grid = Table(show_lines=True)
    detailed_label_quality_table = convert_df_to_rich_table(
        detailed_label_quality, grid, index_name="Samples"
    )
    grid = Table(show_lines=True)
    label_quality_table = convert_df_to_rich_table(
        label_quality, grid, index_name="Samples"
    )
    grid = Table(show_lines=True)
    annotator_performance_table = convert_df_to_rich_table(
        metric["annotator_stats"].sort_index(), grid, index_name="Annotators"
    )

    top_layout = Layout(
        Panel(
            Align.center(detailed_label_quality_table, pad=False, vertical="middle"),
            title="Detailed Label Quality",
            border_style="green",
        )
    )
    down_layout = Layout()
    down_layout.split_row(
        Layout(
            Panel(
                Align.center(label_quality_table, pad=False, vertical="middle"),
                title="Label Quality",
                border_style="green",
            )
        ),
        Layout(
            Panel(
                Align.center(annotator_performance_table, pad=False, vertical="middle"),
                title="Annotator performance",
                border_style="green",
            )
        ),
    )
    layout.split_column(top_layout, down_layout)
    return layout


def make_metrics_table(
    name: str, annotations: np.ndarray, predict_hats: np.ndarray
) -> list[Layout]:
    # trasnpose to (N, K)
    # K is number of annotators
    annotations = annotations.transpose()

    # weather, clip = predict_hats
    simple_weather_quality = get_label_quality_multiannotator(
        annotations,
        predict_hats[0],
        consensus_method="majority_vote",
        quality_method="agreement",
    )
    majority_weather_quality = get_label_quality_multiannotator(
        annotations,
        predict_hats[0],
        consensus_method="majority_vote",
    )
    best_weather_quality = get_label_quality_multiannotator(
        annotations,
        predict_hats[0],
        consensus_method="best_quality",
    )
    ensemble_weather_quality = get_label_quality_multiannotator_ensemble(
        annotations, predict_hats
    )

    layouts = [
        Panel(Text(name, justify="center"), border_style="bright_blue"),
        Panel(Text("Simple-agreement", justify="center"), border_style="bright_yellow"),
        make_metric_table(Layout(), simple_weather_quality),
        Panel(Text("Majority-vote", justify="center"), border_style="bright_yellow"),
        make_metric_table(Layout(), majority_weather_quality),
        Panel(Text("Best-quality", justify="center"), border_style="bright_yellow"),
        make_metric_table(Layout(), best_weather_quality),
        Panel(
            Text("Ensemble (Best-quality)", justify="center"),
            border_style="bright_yellow",
        ),
        make_metric_table(Layout(), ensemble_weather_quality),
    ]
    return layouts


def evaluate_annotations(
    ckpt_path: str,
    review_project: Project,
    projects: list[Union[Project, None]],
    minio_client: minio.Minio,
):
    status = console.status("[yellow]Evaluating annotations...")
    status.start()

    console.log("[blue]Collecting tasks...")
    review_tasks, annotators_tasks = collect_tasks(review_project, projects)

    # collect annotations from reviewer
    console.log("[blue]Collecting annotations...")
    review_weathers = collect_weather_from_tasks(review_tasks)
    review_intensities = collect_intensity_level_from_tasks(review_tasks)[0]

    # convert class ids to probability
    # convert_annotations_to_probabilities(review_weathers, len(get_weather_classes()))
    # convert_annotations_to_probabilities(
    #     review_intensities, len(get_intensity_classes())
    # )

    # collect annotations from annotators
    annotators_weathers = [review_weathers]
    annotators_intensities = [review_intensities]
    for tasks in annotators_tasks:
        if len(tasks) == 0:
            continue

        annotators_weathers.append(collect_weather_from_tasks(tasks))
        annotators_intensities.append(collect_intensity_level_from_tasks(tasks)[0])
    annotators_weathers = np.stack(annotators_weathers, axis=0)
    annotators_intensities = np.stack(annotators_intensities, axis=0)

    # create temporary folder
    tmp_dir = tempfile.TemporaryDirectory()

    # download images to temporary folder
    console.log("[blue]Downloading sampled images...")
    storage_images = collect_storage_images(review_tasks)
    download_object_files(tmp_dir.name, minio_client, storage_images)

    status.stop()

    # predict labels
    console.log("[blue]Predicting labels...")
    _, weather_hats = predict_weathers_ensemble(tmp_dir.name, ckpt_path)
    _, intensity_hats = predict_intensity_ensemble(tmp_dir.name, thresholds=[0.3, 0.67])

    status.start()
    console.log("[blue]Generating metics tables...")
    weather_layouts = make_metrics_table("Weather", annotators_weathers, weather_hats)
    intensity_layouts = make_metrics_table(
        "Intensity", annotators_intensities, intensity_hats
    )

    console.print(*weather_layouts, sep="\n")
    console.print("\n")
    console.print(*intensity_layouts, sep="\n")

    tmp_dir.cleanup()
    status.stop()


def evaluate(
    ckpt_path: _CHECKPOINT_ARGUMENT,
    num_review_samples: _NUM_REVIEW_SAMPLES_ARGUMENT = 50,
    apps: _APPS_ARGUMENT = list(range(1, settings.NUM_LABEL_STUDIO_APPS + 1)),
    review_project_id: _REVIEW_PROJECT_ID_ARGUMENT = None,
    seed: _SEED_ARGUMENT = 1234,
    validate: _VALIDATE_ARGUMENT = True,
):
    random.seed(seed)
    minio_client = connect_minio(
        settings.MINIO_HOST, settings.MINIO_ROOT_USER, settings.MINIO_ROOT_PASSWORD
    )
    ls_clients, ls_review_client = make_ls_clients(apps)
    projects = select_project(ls_clients)

    if validate:
        validate_annotations(projects, minio_client)

    review_project = review_annotations(
        projects, num_review_samples, ls_review_client, review_project_id, minio_client
    )

    console.print(
        "[bold]Press Enter to continue if reviewer finished labeling...", end=""
    )
    console.input()

    evaluate_annotations(ckpt_path, review_project, projects, minio_client)
