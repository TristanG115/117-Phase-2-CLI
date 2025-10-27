import boto3
import os
from typing import Dict
from botocore.exceptions import ClientError
from fastapi import HTTPException

# Loads region and S3 name from config file
def load() -> Dict[str, str]:
    config_info: Dict[str, str] = {}
    # Path to .aws/config.json
    config_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir), ".aws", "config.json"))
    if os.path.exists(config_path):
        with open(config_path, "r", encoding = "utf-8") as i:
            data = json.load(i)
            config_info["AWS_REGION"] = data.get("AWS_REGION")
            config_info["S3_BUCKET_NAME"] = data.get("S3_BUCKET_NAME")
            return config_info

CONFIGS = load()
AWS_REG: str = CONFIGS["AWS_REGION"]
BUCKET_NAME: str = CONFIGS["S3_BUCKET_NAME"]
s3 = boto3.client("s3", region_name = AWS_REG)

# Uploads file to S3
def upload(file: targetFile, key: str) -> Dict[str, str]:
    try:
        s3.upload_fileobj(file.targetFile, BUCKET_NAME, key)
    except ClientError as e:
        raise HTTPException(status_code = 500, details = f"S3 file failed to upload: {e}")
    return {"message": f"File successfully uploaded to {BUCKET_NAME}/{key}"}

# Download file from S3 server through generated download url
def download(key: str, expiration: int = 3000) -> str:
    try:
        download_url = s3.generate_presigned_url(
            "get_object",
            Params = {"Bucket": BUCKET_NAME, "Key": key},
            ExpiresIn = expiration
        )
    except ClientError as e:
        raise HTTPException(status_code = 500, detail = f"Could not generate download url: {e}")
    return download_url