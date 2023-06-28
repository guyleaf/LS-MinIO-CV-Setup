import secrets
from urllib.parse import urlparse

import minio
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
