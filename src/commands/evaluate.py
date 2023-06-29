import json
import os
from typing import Annotated, Union

import minio
import typer
from label_studio_sdk import Client, Project
from rich.prompt import Prompt
from rich.table import Table

from .. import settings
from ..utils import (
    check_ls_connection,
    connect_minio,
    console,
    count_tasks_in_bucket,
    err_console,
    get_storages,
)
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


def make_clients(apps: list[int]) -> tuple[minio.Minio, list[Union[Client, None]]]:
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
    return minio_client, ls_clients


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
        projects_map = json.load(f)

    # select project from table
    table = make_projects_table(projects_map)
    console.print(table, justify="center")

    while True:
        row_id = Prompt.ask(
            "Which project you'd like to evaluate?", console=console, default="0"
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


def are_inner_ids_continuous(tasks: list[dict]) -> bool:
    inner_ids = [task["inner_id"] for task in tasks]
    inner_ids = sorted(inner_ids)
    for task_id in range(1, len(inner_ids) + 1):
        if inner_ids[task_id - 1] != task_id:
            return False
    return True


def validate_annotations(
    projects: list[Union[Project, None]], minio_client: minio.Minio
):
    status = console.status("[yellow]Validating annotations...")

    has_error = False
    for app_id, project in enumerate(projects, start=1):
        if project is None:
            continue

        status.start()
        console.log(f"Validating app-{app_id} annotations...")
        errors = []
        check_ls_connection(project, show_status=False)
        storages = get_storages(project)

        # count total tasks in connected storages
        total_tasks = 0
        for storage in storages:
            total_tasks += count_tasks_in_bucket(minio_client, storage["title"])

        # validate task count
        tasks = project.tasks
        if len(tasks) != total_tasks:
            errors.append(
                "Inconsistent task count between storages and ls. Maybe the annotator deletes the task."
            )

        # validate if inner ids are continuous
        # task id may not continuous
        if not are_inner_ids_continuous(tasks):
            errors.append(
                "Non-continuous task inner id. Maybe the annotator creates new task."
            )

        # validate if tasks are labeled
        total_unlabeled_tasks = len(project.get_unlabeled_tasks())
        if total_unlabeled_tasks != 0:
            errors.append(
                f"There are {total_unlabeled_tasks} unlabeled tasks in project."
            )

        has_error |= len(errors) != 0

        status.stop()

        if len(errors) != 0:
            has_error = True
            err_console.log("[red]Validation errors:")
            for i, error in enumerate(errors, start=1):
                err_console.log(f"  {i}. {error}")

    if has_error:
        err_console.log("[bold red]Validation Error, please check the errors above.")
        raise typer.Abort()


def evaluate(apps: list[int] = [], validate: bool = True):
    minio_client, ls_clients = make_clients(apps)
    projects = select_project(ls_clients)

    if validate:
        validate_annotations(projects, minio_client)
