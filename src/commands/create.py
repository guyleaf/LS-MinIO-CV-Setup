import json
import os
import timeit
from time import sleep
from typing import Annotated, Optional

import minio
import typer
from label_studio_sdk import Client, Project
from label_studio_sdk.project import ProjectSampling
from requests import HTTPError

from .. import settings
from ..utils import (
    check_buckets_exists,
    check_ls_connection,
    connect_minio,
    console,
    count_tasks_in_bucket,
    create_view,
    is_xml,
    sync_storage,
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


def create_project(project_name: str, label_config: str, ls_client: Client) -> Project:
    if not is_xml(label_config):
        with open(label_config, "r") as f:
            label_config = f.read()

    check_ls_connection(ls_client)
    project = ls_client.start_project(
        title=project_name,
        label_config=label_config,
        maximum_annotations=1,
        sampling=ProjectSampling.RANDOM.value,
        show_skip_button=False,
        enable_empty_annotation=False,
    )

    # add predefined views
    for view in settings.LS_VIEWS:
        create_view(project, view)

    return project


def import_data(
    project: Project,
    buckets: list[str],
    minio_client: minio.Minio,
    regex_filter: Optional[str] = None,
    total_tasks: Optional[int] = None,
):
    check_ls_connection(project)
    console.log(
        "[bold yellow]Note: please ignore 502, 504, and 404 error in this step",
        "[bold yellow]due to very large data amount and SDK design.",
    )

    status = console.status("[yellow]Importing data...")
    status.start()
    for bucket in buckets:
        if total_tasks is None:
            total_tasks = count_tasks_in_bucket(minio_client, bucket)
        storage: dict = project.connect_s3_import_storage(
            bucket,
            prefix="tasks/",
            regex_filter=regex_filter,
            use_blob_urls=False,
            title=bucket,
            aws_access_key_id=settings.MINIO_ROOT_USER,
            aws_secret_access_key=settings.MINIO_ROOT_PASSWORD,
            s3_endpoint=settings.MINIO_HOST,
        )

        start_time = timeit.default_timer()
        while True:
            try:
                sync_storage(
                    project, storage["type"], storage["id"], timeout=(10.0, 330.0)
                )
            except HTTPError as e:
                if e.response.status_code not in (504, 502):
                    raise
                sleep(3)

            current_counts = len(project.tasks_ids)
            console.log("Number of tasks imported:", current_counts, "/", total_tasks)
            if current_counts == total_tasks:
                break

        console.log(
            "Finished importing data from the bucket",
            bucket,
            "in",
            f"{timeit.default_timer() - start_time}s.",
        )

    status.stop()


def create(
    project_name: _PROJECT_NAME_ARGUMENT,
    label_config: _LABEL_CONFIG_ARGUMENT,
    buckets: _BUCKETS_ARGUMENT = [],
):
    minio_client = connect_minio(
        settings.MINIO_HOST, settings.MINIO_ROOT_USER, settings.MINIO_ROOT_PASSWORD
    )

    # check & select bucket
    if len(buckets) != 0:
        check_buckets_exists(buckets, minio_client)
    else:
        buckets = [bucket.name for bucket in minio_client.list_buckets()]
        if len(buckets) == 0:
            raise RuntimeError("No buckets in the storage.")

    # create project & import data
    project_ids = []
    for app_id in range(1, settings.NUM_LABEL_STUDIO_APPS + 1):
        ls_host = f"{settings.LABEL_STUDIO_HOST}/{app_id}"
        ls_client = Client(ls_host, settings.LABEL_STUDIO_USER_TOKEN)
        project = create_project(project_name, label_config, ls_client)
        import_data(project, buckets, minio_client, regex_filter=".*json")
        console.log(f"\[app-{app_id}] Imported data [green]successfully[/].")

        project_ids.append(project.params["id"])

    # store project infos for evaluating and exporting
    if os.path.exists(settings.PROJECTS_MAP):
        with open(settings.PROJECTS_MAP, "r") as f:
            projects_map = json.load(f)
    else:
        projects_map = []

    projects_map.append(dict(name=project_name, ids=project_ids))

    with open(settings.PROJECTS_MAP, "w") as f:
        json.dump(projects_map, f)
