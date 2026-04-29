import os
from pathlib import Path

import boto3


MODEL_SOURCE_BUCKET = "project-moi-ai"
MODEL_PREFIX = "OceanLens/demo/v1/"

FILES_TO_DOWNLOAD = [
    "checkpoints/cno_v2_loggrad.ckpt",
    "checkpoints/fm_v4_s1_logit_t.ckpt",
    "configs/base.yaml",
    "configs/variants/v4_s1_logit_t.yaml",
    "norm_stats.json",
]


def _endpoint_url() -> str:
    endpoint = os.environ.get("AWS_S3_ENDPOINT")
    if not endpoint:
        raise ValueError("AWS_S3_ENDPOINT environment variable is not set.")
    endpoint = endpoint.replace("https://", "").replace("http://", "").rstrip("/")
    return f"https://{endpoint}"


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=_endpoint_url(),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ["AWS_SESSION_TOKEN"],
    )


def synchronize_model_locally(local_dir: str):
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    print(f"Synchronizing OceanLens assets from: s3://{MODEL_SOURCE_BUCKET}/{MODEL_PREFIX}")
    client = _s3_client()

    for file_name in FILES_TO_DOWNLOAD:
        target = local_path / file_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        print(f"Downloading: {file_name}")
        object_key = MODEL_PREFIX + file_name
        with open(target, "wb") as handle:
            client.download_fileobj(MODEL_SOURCE_BUCKET, object_key, handle)

    print(f"OceanLens assets synchronized into {local_dir}")
