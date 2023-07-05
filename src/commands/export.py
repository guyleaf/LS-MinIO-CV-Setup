import os
import shutil
import tempfile
from time import sleep
from typing import Annotated

import typer
from label_studio_converter.converter import Format
from label_studio_sdk import Project

from .. import settings
from ..utils import check_ls_connection, console, convert_export
from .utils import make_ls_clients, select_project

#################################################################################


def validate_export_format(value: str):
    Format.from_string(value)
    return value


# Argument type hints

_OUT_DIR_ARGUMENT = Annotated[str, typer.Option(help="The path of output folder")]

_APPS_ARGUMENT = Annotated[
    list[int],
    typer.Option(
        help="The list of apps you'd like to evaluate",
    ),
]

_EXPORT_FORMAT_ARGUMENT = Annotated[
    str,
    typer.Option(
        help="The export formats of annotations", callback=validate_export_format
    ),
]


#################################################################################


def export_annotations(path: str, project: Project, export_type: str = "JSON"):
    check_ls_connection(project)

    status = console.status("[yellow]Exporting annotations...")
    status.start()

    snapshot = project.export_snapshot_create(project.params["title"])

    snapshot_id = snapshot["id"]
    while True:
        snapshot_status = project.export_snapshot_status(snapshot_id)
        if snapshot_status.is_completed():
            break
        elif snapshot_status.is_failed():
            raise RuntimeError(snapshot_status.response)
        sleep(3)

    convert_export(project, snapshot_id, export_type)

    with tempfile.TemporaryDirectory() as tmp_dir:
        _, file_path = project.export_snapshot_download(
            export_id=snapshot_id, export_type=export_type, path=tmp_dir
        )
        file_path = os.path.join(tmp_dir, file_path)
        shutil.copyfile(file_path, path)

    status.stop()


def export(
    out_dir: _OUT_DIR_ARGUMENT = "exports",
    apps: _APPS_ARGUMENT = list(range(1, settings.NUM_LABEL_STUDIO_APPS + 1)),
    export_format: _EXPORT_FORMAT_ARGUMENT = "JSON",
):
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ls_clients, _ = make_ls_clients(apps)
    projects = select_project(ls_clients)

    for app_id, project in enumerate(projects, start=1):
        if project is None:
            continue

        title = project.params["title"].lower()
        title = "".join(c if c.isalnum() else "_" for c in title)

        out_path = os.path.join(out_dir, title)
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, f"app_{app_id}.json")

        console.log(f"Exporting app-{app_id} annotations...")
        export_annotations(out_path, project, export_type=str(export_format))
        console.log(
            f"Finished exporting app-{app_id} annotations [green]successfully[/]."
        )
