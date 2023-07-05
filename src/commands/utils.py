import os
import sys
from typing import Callable
from urllib.parse import urlparse

import typer


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
