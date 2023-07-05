import json
import os
import sys
from typing import Callable, Union
from urllib.parse import urlparse

import typer
from label_studio_sdk import Client, Project
from rich.prompt import Prompt
from rich.table import Table

from .. import settings
from ..utils import (
    check_ls_connection,
    console,
)


def validate_path(value: str) -> str:
    if not os.path.exists(value):
        raise typer.BadParameter(f"The path `{value}` is not found.")
    return value


def validate_int_range(low: int, high: int = sys.maxsize) -> Callable[[int], int]:
    """Validate int value if it is in the range
    low <= value <= high

    Args:
        low (int): lowest value accepted
        high (int): highest values accepted
    """

    def validate_number(value: int) -> int:
        if value < low or value > high:
            raise typer.BadParameter(
                f"The value `{value}` should be between {low} and {high}."
            )
        return value

    return validate_number


def validate_url(url: str) -> bool:
    parsed_url = urlparse(url)
    if (
        parsed_url.scheme != "http" and parsed_url.scheme != "https"
    ) or parsed_url.netloc == "":
        raise typer.BadParameter(f"The url `{url}` is not valid.")
    return url


def make_ls_clients(
    apps: list[int],
) -> tuple[list[Union[Client, None]], Client]:
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
    return ls_clients, ls_review_client


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
