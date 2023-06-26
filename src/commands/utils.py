import os

import typer


def validate_path(value: str):
    if not os.path.exists(value):
        raise typer.BadParameter(f"The path `{value}` is not found.")
    return value