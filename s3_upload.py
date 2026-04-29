import boto3
import os
from boto3.s3.transfer import TransferConfig


# Build and return the S3 endpoint URL with HTTPS protocol.
def get_s3_endpoint_url_with_protocol() -> str:
    s3_endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
    if not s3_endpoint_url:
        raise ValueError("AWS_S3_ENDPOINT environment variable is not set.")
    s3_endpoint_url = s3_endpoint_url.replace("https://", "").replace("http://", "").rstrip("/")
    return f"https://{s3_endpoint_url}"


# Create and return a boto3 S3 client using environment credentials.
def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=get_s3_endpoint_url_with_protocol(),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ["AWS_SESSION_TOKEN"],
    )


# List all objects in a bucket under the given prefix.
def list_objects(bucket_name: str, prefix_key: str) -> list[dict]:
    try:
        response = get_s3_client().list_objects(Bucket=bucket_name, Prefix=prefix_key)
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            return response["Contents"]
        raise Exception(response)
    except Exception as e:
        print(f"Failed to list objects {bucket_name}/{prefix_key}: {e}")
        raise


# Delete a specific object from S3.
def delete_object(bucket_name: str, object_key: str):
    try:
        response = get_s3_client().delete_object(Bucket=bucket_name, Key=object_key)
        if response["ResponseMetadata"]["HTTPStatusCode"] in [200, 204]:
            print(f"Deleted: {bucket_name}/{object_key}")
        else:
            raise Exception(response)
    except Exception as e:
        print(f"Failed to delete {bucket_name}/{object_key}: {e}")
        raise


# Upload raw bytes to S3.
def save_bytes_to_s3(bucket_name: str, object_bytes, object_key: str):
    try:
        response = get_s3_client().put_object(Bucket=bucket_name, Body=object_bytes, Key=object_key)
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise Exception(response)
    except Exception as e:
        print(f"Failed to upload bytes to {bucket_name}/{object_key}: {e}")
        raise


# Upload a local file to S3 using multipart with large chunks to avoid rate limiting.
def save_file_to_s3(bucket_name: str, local_file_path: str, object_key: str):
    try:
        config = TransferConfig(
            multipart_threshold=100 * 1024 * 1024,
            multipart_chunksize=100 * 1024 * 1024,
            max_concurrency=1,
        )
        print(f"Uploading {local_file_path} to s3://{bucket_name}/{object_key}...")
        get_s3_client().upload_file(local_file_path, bucket_name, object_key, Config=config)
        print(f"Upload succeeded to s3://{bucket_name}/{object_key}")
    except Exception as e:
        print(f"Failed to upload {local_file_path} to s3://{bucket_name}/{object_key}: {e}")
        raise


# Upload a large file using put_object to bypass multipart upload issues with MinIO.
def save_large_file_to_s3(bucket_name: str, local_file_path: str, object_key: str):
    try:
        print(f"Uploading {local_file_path} to s3://{bucket_name}/{object_key}...")
        with open(local_file_path, "rb") as f:
            get_s3_client().put_object(Bucket=bucket_name, Key=object_key, Body=f)
        print(f"Upload succeeded to s3://{bucket_name}/{object_key}")
    except Exception as e:
        print(f"Failed to upload {local_file_path} to s3://{bucket_name}/{object_key}: {e}")
        raise