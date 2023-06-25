import os
from io import StringIO
from typing import Annotated
from urllib.parse import urlparse

import typer
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

from ..utils import console, generate_token

_TEMPLATE_SUFFIX = ".template"
_ENV_TEMPLATE = f".env{_TEMPLATE_SUFFIX}"
_APP_COMPOSE_TEMPLATE = f"docker-compose.app.yml{_TEMPLATE_SUFFIX}"

# deploy
_DEPLOY_DIR = "deploy"
_INIT_DB_TEMPLATE = f"{_DEPLOY_DIR}/init-apps-db.sql{_TEMPLATE_SUFFIX}"
_NGINX_TEMPLATE = f"{_DEPLOY_DIR}/nginx.conf{_TEMPLATE_SUFFIX}"

_LS_TOKEN_LENGTH = 20
_LS_PASSWORD_LENGTH = 16
_MINIO_PASSWORD_LENGTH = 16


#################################################################################


def validate_num_annotators(value: int):
    if not value > 0:
        raise typer.BadParameter("num_annotators should be larger than 0.")
    return value


def validate_templates_dir(value: str):
    if not os.path.exists(value):
        raise typer.BadParameter("templates_dir is not found.")
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
    typer.Option(help="The path of templates folder", callback=validate_templates_dir),
]


#################################################################################


def save_template(path: str, content: str):
    with open(path, "w") as f:
        f.write(content)


def is_host_valid(host: str) -> bool:
    return host.startswith("http://") or host.startswith("https://")


def render_env_template(num_annotators: int, env: Environment) -> str:
    template = env.get_template(_ENV_TEMPLATE)

    # generate tokens
    ls_token = generate_token(_LS_TOKEN_LENGTH)
    ls_passwords = [generate_token(_LS_PASSWORD_LENGTH) for _ in range(num_annotators)]
    minio_password = generate_token(_MINIO_PASSWORD_LENGTH)

    result = template.render(
        ls_token=ls_token, ls_passwords=ls_passwords, minio_password=minio_password
    )
    return result


def render_app_compose_template(num_annotators: int, env: Environment) -> str:
    template = env.get_template(_APP_COMPOSE_TEMPLATE)
    result = template.render(num_apps=num_annotators)
    return result


def render_init_db_template(num_annotators: int, env: Environment) -> str:
    template = env.get_template(_INIT_DB_TEMPLATE)
    result = template.render(num_apps=num_annotators)
    return result


def render_nginx_template(host: str, env: Environment) -> str:
    prefix_path = ""
    if is_host_valid(host):
        parsed_host = urlparse(host)
        prefix_path = parsed_host.path.removesuffix("/")

    console.log("Use prefix host path:", prefix_path)

    template = env.get_template(_NGINX_TEMPLATE)
    result = template.render(prefix_path=prefix_path)
    return result


def print_start_hints(out_dir: str):
    console.log("[bold green]Finished.")

    compose_path = os.path.join(out_dir, "docker-compose.yml")
    app_compose_path = os.path.join(
        out_dir, _APP_COMPOSE_TEMPLATE.removesuffix(_TEMPLATE_SUFFIX)
    )
    minio_compose_path = os.path.join(out_dir, "docker-compose.minio.yml")

    console.log("\nPlease run this command to start the service:")
    console.log(f"[blue]docker compose -f {compose_path} -f {app_compose_path} up -d")

    console.log(
        "\n[yellow]Note: if you'd like to use MinIO container, append this file"
    )
    console.log(f"[blue]-f {minio_compose_path}")


def print_ls_url():
    pass


def setup(
    num_annotators: _NUM_ANNOTATORS_ARGUMENT,
    templates_dir: _TEMPLATES_DIR_ARGUMENT = "templates",
    out_dir: str = ".",
):
    # create out_dir if not exist
    os.makedirs(out_dir, exist_ok=True)

    console.log("Number of annotators:", num_annotators)

    # render templates
    env = Environment(loader=FileSystemLoader(templates_dir))
    with console.status("[bold green]Rendering templates..."):
        env_content = render_env_template(num_annotators, env)
        env_path = os.path.join(out_dir, _ENV_TEMPLATE.removesuffix(_TEMPLATE_SUFFIX))
        console.log("Rendered env template [green]successfully[/].")

        if not load_dotenv(stream=StringIO(env_content)):
            raise RuntimeError("Error: invalid env file.")

        nginx_content = render_nginx_template(
            os.environ.get("LABEL_STUDIO_HOST", ""), env
        )
        nginx_path = os.path.join(
            out_dir, _NGINX_TEMPLATE.removesuffix(_TEMPLATE_SUFFIX)
        )
        console.log("Rendered nginx template [green]successfully[/].")

        app_compose_content = render_app_compose_template(num_annotators, env)
        app_compose_path = os.path.join(
            out_dir, _APP_COMPOSE_TEMPLATE.removesuffix(_TEMPLATE_SUFFIX)
        )
        console.log("Rendered app compose template [green]successfully[/].")

        init_db_content = render_init_db_template(num_annotators, env)
        init_db_path = os.path.join(
            out_dir, _INIT_DB_TEMPLATE.removesuffix(_TEMPLATE_SUFFIX)
        )
        console.log("Rendered init db template [green]successfully[/].")

    save_template(env_path, env_content)
    save_template(nginx_path, nginx_content)
    save_template(app_compose_path, app_compose_content)
    save_template(init_db_path, init_db_content)
    print_start_hints(out_dir)
