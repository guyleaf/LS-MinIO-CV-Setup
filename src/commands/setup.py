import os
from io import StringIO
from typing import Annotated
from urllib.parse import urlparse

import typer
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

from .. import settings
from ..utils import console, generate_token
from .utils import validate_path, validate_url

#################################################################################


def validate_num_annotators(value: int):
    if not value > 0:
        raise typer.BadParameter("num_annotators should be larger than 0.")
    return value


# Argument type hints
_NUM_ANNOTATORS_ARGUMENT = Annotated[
    int,
    typer.Argument(
        help="How many annotators do you need?", callback=validate_num_annotators
    ),
]

_TEMPLATES_DIR_ARGUMENT = Annotated[
    str,
    typer.Option(help="The path of templates folder", callback=validate_path),
]

_LS_HOST_ARGUMENT = Annotated[
    str,
    typer.Option(help="The url of label studio host", callback=validate_url),
]

_MINIO_HOST_ARGUMENT = Annotated[
    str,
    typer.Option(help="The url of minio host", callback=validate_url),
]


#################################################################################


def save_template(path: str, content: str):
    with open(path, "w") as f:
        f.write(content)


def render_env_template(
    num_annotators: int,
    ls_host: str,
    minio_host: str,
    env: Environment,
) -> str:
    template = env.get_template(settings.ENV_TEMPLATE)

    # generate tokens
    ls_token = generate_token(settings.LS_TOKEN_LENGTH)
    ls_reviewer_password = generate_token(settings.LS_PASSWORD_LENGTH)
    ls_app_passwords = [
        generate_token(settings.LS_PASSWORD_LENGTH) for _ in range(num_annotators)
    ]
    minio_password = generate_token(settings.MINIO_PASSWORD_LENGTH)

    result = template.render(
        ls_token=ls_token,
        ls_reviewer_password=ls_reviewer_password,
        ls_app_passwords=ls_app_passwords,
        minio_password=minio_password,
        ls_host=ls_host,
        minio_host=minio_host,
    )
    return result


def render_app_compose_template(num_annotators: int, env: Environment) -> str:
    template = env.get_template(settings.APP_COMPOSE_TEMPLATE)
    result = template.render(num_apps=num_annotators)
    return result


def render_init_db_template(num_annotators: int, env: Environment) -> str:
    template = env.get_template(settings.INIT_DB_TEMPLATE)
    result = template.render(num_apps=num_annotators)
    return result


def render_nginx_template(host: str, env: Environment) -> str:
    parsed_host = urlparse(host)
    prefix_path = parsed_host.path.removesuffix("/")

    console.log("Use prefix host path:", prefix_path)

    template = env.get_template(settings.NGINX_TEMPLATE)
    result = template.render(prefix_path=prefix_path)
    return result


def print_start_hints():
    console.log("[bold green]Finished.")

    compose_path = "docker-compose.yml"
    app_compose_path = settings.APP_COMPOSE_TEMPLATE.removesuffix(settings.TEMPLATE_SUFFIX)
    minio_compose_path = "docker-compose.minio.yml"

    console.log("\nPlease run this command to start the service:")
    console.log(f"[blue]docker compose -f {compose_path} -f {app_compose_path} up -d")

    console.log(
        "\n[yellow]Note: if you'd like to use MinIO container, run this command"
    )
    console.log(
        f"[blue]docker compose -f {compose_path} -f {app_compose_path} -f {minio_compose_path} up -d"
    )


def print_ls_url():
    # TODO: Print ls url
    pass


def setup(
    num_annotators: _NUM_ANNOTATORS_ARGUMENT,
    ls_host: _LS_HOST_ARGUMENT = "http://localhost:8085/app",
    minio_host: _MINIO_HOST_ARGUMENT = "http://localhost:9000",
    templates_dir: _TEMPLATES_DIR_ARGUMENT = "templates",
):
    if os.path.exists(settings.PROJECTS_MAP):
        os.remove(settings.PROJECTS_MAP)

    console.log("Number of annotators:", num_annotators)

    # render templates
    env = Environment(loader=FileSystemLoader(templates_dir))
    with console.status("[bold green]Rendering templates..."):
        env_content = render_env_template(num_annotators, ls_host, minio_host, env)
        env_path = settings.ENV_TEMPLATE.removesuffix(settings.TEMPLATE_SUFFIX)
        console.log("Rendered env template [green]successfully[/].")

        if not load_dotenv(stream=StringIO(env_content)):
            raise RuntimeError("Error: invalid env content.")

        nginx_content = render_nginx_template(
            os.environ.get("LABEL_STUDIO_HOST", ""), env
        )
        nginx_path = settings.NGINX_TEMPLATE.removesuffix(settings.TEMPLATE_SUFFIX)
        console.log("Rendered nginx template [green]successfully[/].")

        app_compose_content = render_app_compose_template(num_annotators, env)
        app_compose_path = settings.APP_COMPOSE_TEMPLATE.removesuffix(settings.TEMPLATE_SUFFIX)
        console.log("Rendered app compose template [green]successfully[/].")

        init_db_content = render_init_db_template(num_annotators, env)
        init_db_path = settings.INIT_DB_TEMPLATE.removesuffix(settings.TEMPLATE_SUFFIX)
        console.log("Rendered init db template [green]successfully[/].")

    save_template(env_path, env_content)
    save_template(nginx_path, nginx_content)
    save_template(app_compose_path, app_compose_content)
    save_template(init_db_path, init_db_content)
    print_start_hints()
