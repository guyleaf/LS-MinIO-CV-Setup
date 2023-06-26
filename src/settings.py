import os

from dotenv import load_dotenv

load_dotenv(override=True)

LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST", "")
LABEL_STUDIO_USER_TOKEN = os.getenv("LABEL_STUDIO_USER_TOKEN")

NUM_LABEL_STUDIO_APPS = 0
while os.getenv(f"LABEL_STUDIO_USERNAME_{NUM_LABEL_STUDIO_APPS + 1}", None) is not None:
    NUM_LABEL_STUDIO_APPS += 1


MINIO_HOST = os.getenv("MINIO_HOST")
MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD")
