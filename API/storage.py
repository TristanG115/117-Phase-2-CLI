from __future__ import annotations
import boto3
import json
import os
from typing import Dict, BinaryIO
from botocore.exceptions import ClientError
from fastapi import HTTPException

class S3Storage:
    # Handles all AWS S3 operations for the Trustworthy Model Registry

    def __init__(self):
        self.configs = self.load_config()
        self.region: str = self.configs["AWS_REGION"]
        self.bucket_name: str = self.configs["S3_BUCKET_NAME"]

        # Initialize S3 client
        self.s3 = boto3.client("s3", region_name=self.region)

    def load_config(self) -> Dict[str, str]:
        # Load AWS region and bucket name from .aws/config.json
        config_info: Dict[str, str] = {}
        config_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, ".aws", "config.json"))
        )

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            config_info["AWS_REGION"] = data.get("AWS_REGION")
            config_info["S3_BUCKET_NAME"] = data.get("S3_BUCKET_NAME")
        return config_info

    def upload(self, file_obj: BinaryIO, key: str) -> Dict[str, str]:
        # Upload a file object to the S3 bucket
        try:
            self.s3.upload_fileobj(file_obj, self.bucket_name, key)
        except ClientError as e:
            raise HTTPException(status_code = 500, detail = f"S3 upload failed: {e}")

        return {"message": f"File successfully uploaded to s3://{self.bucket_name}/{key}"}

    def download_url(self, key: str, expiration: int = 3000) -> str:
        # Generate a temporary pre-signed URL to download a file
        try:
            url = self.s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": key},
                ExpiresIn = expiration,
            )
        except ClientError as e:
            raise HTTPException(status_code = 500, detail = f"Could not generate download URL: {e}")
        return url