import os

from dotenv import load_dotenv
from label_studio_sdk.data_manager import Column, Filters, Operator, Type

load_dotenv(override=True)

# shared
LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST", "")
LABEL_STUDIO_USER_TOKEN = os.getenv("LABEL_STUDIO_USER_TOKEN")

NUM_LABEL_STUDIO_APPS = 0
while os.getenv(f"LABEL_STUDIO_USERNAME_{NUM_LABEL_STUDIO_APPS + 1}", None) is not None:
    NUM_LABEL_STUDIO_APPS += 1


MINIO_HOST = os.getenv("MINIO_HOST")
MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD")

# setup command
TEMPLATE_SUFFIX = ".template"
ENV_TEMPLATE = f".env{TEMPLATE_SUFFIX}"
APP_COMPOSE_TEMPLATE = f"docker-compose.app.yml{TEMPLATE_SUFFIX}"

DEPLOY_DIR = "deploy"
INIT_DB_TEMPLATE = f"{DEPLOY_DIR}/init-apps-db.sql{TEMPLATE_SUFFIX}"
NGINX_TEMPLATE = f"{DEPLOY_DIR}/nginx.conf{TEMPLATE_SUFFIX}"

LS_TOKEN_LENGTH = 20
LS_PASSWORD_LENGTH = 16
MINIO_PASSWORD_LENGTH = 16

# create command
PROJECTS_MAP = "projects.json"
LS_VIEWS = [
    {
        "filters": Filters.create(
            Filters.AND,
            [
                Filters.item(
                    Column.completed_at,
                    Operator.EMPTY,
                    Type.Datetime,
                    Filters.value(True),
                ),
            ],
        ),
        "ordering": [f"-{Column.created_at}"],
        "hiddenColumns": {
            "explore": [
                Column.cancelled_annotations,
                Column.total_predictions,
                Column.annotations_results,
                "tasks:annotations_ids",
                Column.predictions_score,
                Column.predictions_model_versions,
                Column.predictions_results,
                Column.file_upload,
                "tasks:storage_filename",
                "tasks:inner_id",
                Column.total_annotations,
                Column.completed_at,
                Column.annotators,
                "tasks:updated_by",
                "tasks:avg_lead_time",
                Column.data("data_id"),
                Column.data("metadata"),
            ],
            "labeling": [],
        },
        "title": "Unlabeled",
    },
    {
        "filters": Filters.create(
            Filters.AND,
            [
                Filters.item(
                    Column.completed_at,
                    Operator.EMPTY,
                    Type.Datetime,
                    Filters.value(False),
                ),
            ],
        ),
        "ordering": [Column.completed_at],
        "hiddenColumns": {
            "explore": [
                Column.cancelled_annotations,
                Column.total_predictions,
                Column.annotations_results,
                "tasks:annotations_ids",
                Column.predictions_score,
                Column.predictions_model_versions,
                Column.predictions_results,
                Column.file_upload,
                "tasks:storage_filename",
                "tasks:inner_id",
                Column.total_annotations,
                Column.annotators,
                "tasks:updated_by",
                "tasks:avg_lead_time",
                Column.data("data_id"),
                Column.data("metadata"),
            ],
            "labeling": [],
        },
        "title": "Annotated",
    },
]
