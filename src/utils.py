import secrets
from typing import Union
from urllib.parse import urlparse
from xml.etree import ElementTree

import minio
from label_studio_sdk import Client, Project
from label_studio_sdk.client import TIMEOUT
from requests import RequestException
from rich.console import Console

console = Console()
err_console = Console(stderr=True)


def generate_token(length: int) -> str:
    return secrets.token_hex(length)


def connect_minio(host: str, access_key: str, secret_key: str) -> minio.Minio:
    parsed_minio_host = urlparse(host)
    blob_client = minio.Minio(
        parsed_minio_host.netloc,
        access_key=access_key,
        secret_key=secret_key,
        secure=parsed_minio_host.scheme == "https",
    )
    return blob_client


def check_buckets_exists(buckets: list[str], minio_client: minio.Minio) -> None:
    for bucket in buckets:
        if not minio_client.bucket_exists(bucket):
            raise RuntimeError(f"The bucket {bucket} does not exist.")


def count_tasks_in_bucket(minio_client: minio.Minio, bucket_name: str) -> int:
    count = 0
    for _ in minio_client.list_objects(bucket_name, prefix="tasks/", recursive=True):
        count += 1
    return count


def check_ls_connection(
    ls_client: Union[Client, Project], show_status: bool = True
) -> None:
    if show_status:
        status = console.status("[yellow]Waiting LS up...")
        status.start()

    while True:
        try:
            ls_client.check_connection()
            break
        except RequestException:
            pass

    if show_status:
        status.stop()


def sync_storage(
    project: Project, storage_type: str, storage_id: int, timeout: int = TIMEOUT
) -> dict:
    response = project.make_request(
        "POST", f"/api/storages/{storage_type}/{str(storage_id)}/sync", timeout=timeout
    )
    return response.json()


def create_view(project: Project, data: dict) -> dict:
    data = {
        "project": project.params["id"],
        "data": data,
    }
    response = project.make_request("POST", "/api/dm/views", json=data)
    return response.json()


def get_storages(project: Project) -> list[dict]:
    params = dict(project=project.params["id"])
    response = project.make_request("GET", "/api/storages", params=params)
    return response.json()


def is_xml(content: str):
    try:
        ElementTree.fromstring(content)
    except ElementTree.ParseError:
        return False
    return True
