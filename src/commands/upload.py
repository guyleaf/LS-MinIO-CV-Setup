import json
import os
from io import BytesIO
from typing import Annotated

import minio
import typer
from minio.deleteobjects import DeleteError, DeleteObject
from minio.helpers import ObjectWriteResult
from rich.progress import track

from .. import settings
from ..datasets.common import ImageDataset
from ..utils import connect_minio, console, err_console, generate_token
from .utils import validate_path

#################################################################################


# Argument type hints
_DATA_DIR_ARGUMENT = Annotated[
    str,
    typer.Argument(help="The path of data folder", callback=validate_path),
]

_BUCKET_NAME_ARGUMENT = Annotated[
    str,
    typer.Option(help="The name of the bucket"),
]


#################################################################################


def clear_bucket(bucket_name: str, minio_client: minio.Minio):
    console.log("Clearing bucket...")
    objects = minio_client.list_objects(bucket_name, recursive=True)
    objects = [DeleteObject(obj.object_name) for obj in objects]
    errors: list[DeleteError] = list(minio_client.remove_objects(bucket_name, objects))
    if len(errors) != 0:
        for error in errors:
            err_console.log("[red]Error code:", error.code)
            err_console.log("[red]Error message:", error.message)
            err_console.log()

        raise RuntimeError(f"Found {len(errors)} errors.")


def create_bucket(bucket_name: str, minio_client: minio.Minio):
    if minio_client.bucket_exists(bucket_name):
        console.log(f"The bucket {bucket_name} already exists.")
        typer.confirm("Do you want to continue(clear) ?", abort=True)
        clear_bucket(bucket_name, minio_client)
    else:
        console.log("Creating bucket...")
        minio_client.make_bucket(bucket_name)


def create_task_format(id: int, bucket_name: str, object_name: str, image_path: str):
    return dict(
        data=dict(
            image=f"s3://{bucket_name}/{object_name}",
            data_id=id,
            metadata=dict(
                path=image_path,
                bucket_name=bucket_name,
                object_name=object_name,
            ),
        )
    )


def upload_data(data_dir: str, bucket_name: str, minio_client: minio.Minio):
    dataset = ImageDataset(data_dir)
    total = len(dataset)

    console.log("Total images:", total)

    for id, (_, image_path, content_type) in track(
        enumerate(dataset, start=1), total=total, description="Uploading..."
    ):
        # upload data
        relative_image_path = os.path.relpath(image_path, start=data_dir)
        result: ObjectWriteResult = minio_client.fput_object(
            bucket_name, relative_image_path, image_path, content_type=content_type
        )

        # upload a task for data
        task = create_task_format(
            id, result.bucket_name, result.object_name, relative_image_path
        )
        f = BytesIO(json.dumps(task).encode("utf-8"))

        minio_client.put_object(
            bucket_name,
            "tasks/task_" + str(id).zfill(8) + ".json",
            f,
            f.getbuffer().nbytes,
            content_type="application/json",
        )


def upload(
    data_dir: _DATA_DIR_ARGUMENT, bucket_name: _BUCKET_NAME_ARGUMENT = generate_token(4)
):
    data_dir = os.path.expanduser(data_dir)

    minio_client = connect_minio(
        settings.MINIO_HOST, settings.MINIO_ROOT_USER, settings.MINIO_ROOT_PASSWORD
    )

    create_bucket(bucket_name, minio_client)

    upload_data(data_dir, bucket_name, minio_client)

    console.log(f"[bold green]Uploaded data to the bucket {bucket_name} successfully.")
