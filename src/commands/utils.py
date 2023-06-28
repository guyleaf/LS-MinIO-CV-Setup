import os
from urllib.parse import urlparse

import typer


def validate_path(value: str) -> str:
    if not os.path.exists(value):
        raise typer.BadParameter(f"The path `{value}` is not found.")
    return value


def validate_url(url: str) -> bool:
    parsed_url = urlparse(url)
    if (
        (parsed_url.scheme != "http"
        and parsed_url.scheme != "https")
        or parsed_url.netloc == ""
    ):
        raise typer.BadParameter(f"The url `{url}` is not valid.")
    return url
