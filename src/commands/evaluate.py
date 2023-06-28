from typing import Annotated

import typer

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


def evaluate(project_name: _PROJECT_NAME_ARGUMENT):
    pass
