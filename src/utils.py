import secrets
from typing import Optional, Union
from urllib.parse import urlparse
from xml.etree import ElementTree

import minio
import numpy as np
import pandas as pd
from label_studio_sdk import Client, Project
from label_studio_sdk.client import TIMEOUT
from requests import RequestException
from rich.console import Console
from rich.table import Table

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
    data = dict(project=project.params["id"], data=data)
    response = project.make_request("POST", "/api/dm/views", json=data)
    return response.json()


def get_storages(project: Project) -> list[dict]:
    params = dict(project=project.params["id"])
    response = project.make_request("GET", "/api/storages", params=params)
    return response.json()


def convert_export(project: Project, export_id: int, export_type: str):
    id = project.params["id"]
    data = dict(export_type=export_type)
    response = project.make_request(
        "POST", f"/api/projects/{id}/exports/{export_id}/convert", json=data
    )
    return response.json()


def is_xml(content: str):
    try:
        ElementTree.fromstring(content)
    except ElementTree.ParseError:
        return False
    return True


def convert_annotations_to_probabilities(
    annotations: np.ndarray, num_classes: int
) -> np.ndarray:
    probabilities = np.zeros((annotations.size, num_classes), dtype=np.float64)
    np.put_along_axis(probabilities, np.expand_dims(annotations, axis=1), 1, axis=1)
    return probabilities


def convert_df_to_rich_table(
    pandas_dataframe: pd.DataFrame,
    rich_table: Table,
    show_index: bool = True,
    index_name: Optional[str] = None,
) -> Table:
    """Convert a pandas.DataFrame obj into a rich.Table obj.
    Args:
        pandas_dataframe (DataFrame): A Pandas DataFrame to be converted to a rich Table.
        rich_table (Table): A rich Table that should be populated by the DataFrame values.
        show_index (bool): Add a column with a row count to the table. Defaults to True.
        index_name (str, optional): The column name to give to the index column. Defaults to None, showing no value.
    Returns:
        Table: The rich Table instance passed, populated with the DataFrame values."""

    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    for column in pandas_dataframe.columns:
        rich_table.add_column(str(column))

    for index, *value_list in pandas_dataframe.itertuples():
        row = [str(index)] if show_index else []
        row += [str(x) for x in value_list]
        rich_table.add_row(*row)

    return rich_table
