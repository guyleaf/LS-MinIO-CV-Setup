import timeit
from typing import Annotated, Union

import minio
import typer
from label_studio_sdk import Client, Project
from label_studio_sdk.client import TIMEOUT
from label_studio_sdk.data_manager import Column, Filters, Operator, Type
from label_studio_sdk.project import ProjectSampling
from requests import HTTPError

from .. import settings
from ..utils import connect_minio, console
from .utils import validate_path

LS_VIEWS = [
    {
        "filters": Filters.create(
            Filters.AND,
            [
                Filters.item(
                    Column.completed_at,
                    Operator.EMPTY,
                    Type.Datetime,
                    Filters.value(True),
                ),
            ],
        ),
        "ordering": [f"-{Column.created_at}"],
        "title": "Unlabeled",
    },
    {
        "filters": Filters.create(
            Filters.AND,
            [
                Filters.item(
                    Column.completed_at,
                    Operator.EMPTY,
                    Type.Datetime,
                    Filters.value(False),
                ),
            ],
        ),
        "ordering": [Column.completed_at],
        "title": "Annotated",
    },
]

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


def check_buckets_exists(buckets: list[str], minio_client: minio.Minio) -> None:
    for bucket in buckets:
        if not minio_client.bucket_exists(bucket):
            raise RuntimeError(f"The bucket {bucket} does not exist.")


def check_ls_connection(ls_client: Union[Client, Project]) -> None:
    with console.status("[yellow]Waiting LS up..."):
        try:
            ls_client.check_connection()
        except Exception:
            pass


def count_tasks_in_storage(minio_client: minio.Minio, bucket_name: str) -> int:
    count = 0
    for _ in minio_client.list_objects(bucket_name, prefix="tasks/", recursive=True):
        count += 1
    return count


def sync_storage(
    project: Project, storage_type: str, storage_id: int, timeout: int = TIMEOUT
) -> dict:
    response = project.make_request(
        "POST", f"/api/storages/{storage_type}/{str(storage_id)}/sync", timeout=timeout
    )
    return response.json()


def delete_view(project: Project, id: int) -> None:
    project.make_request("DELETE", f"/api/dm/views/{id}")


def create_project(project_name: str, label_config: str, ls_client: Client) -> Project:
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
    for view in LS_VIEWS:
        project.create_view(**view)

    # delete default view
    delete_view(project, 0)
    return project


def import_data(project: Project, buckets: list[str], minio_client: minio.Minio):
    check_ls_connection(project)
    console.log(
        "[bold yellow]Note: please ignore 502, 504, and 404 error in this step",
        "[bold yellow]due to very large data amount and SDK design.",
    )

    status = console.status("[yellow]Importing data...")
    status.start()
    for bucket in buckets:
        total_tasks = count_tasks_in_storage(minio_client, bucket)
        storage: dict = project.connect_s3_import_storage(
            bucket,
            prefix="tasks/",
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
    if len(buckets) != 0:
        check_buckets_exists(buckets, minio_client)
    else:
        buckets = [bucket.name for bucket in minio_client.list_buckets()]
        if len(buckets) == 0:
            raise RuntimeError("No buckets in the storage.")

    for app_id in range(1, settings.NUM_LABEL_STUDIO_APPS + 1):
        ls_host = f"{settings.LABEL_STUDIO_HOST}/{app_id}"
        ls_client = Client(ls_host, settings.LABEL_STUDIO_USER_TOKEN)
        project = create_project(project_name, label_config, ls_client)
        import_data(project, buckets, minio_client)
        console.log(f"[app-{app_id}] Imported data [green]successfully[/].")
