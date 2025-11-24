from __future__ import annotations

import json
import logging
import os
from typing import Any, BinaryIO, Dict, Optional

import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class StorageUnavailableError(Exception):
    """Raised when S3 is unavailable or upload fails."""
    pass


class S3Storage:
    """Handles all AWS S3 operations for the Trustworthy Model Registry"""

    def __init__(self):
        self.configs = self.load_config()
        self.region: str = self.configs["AWS_REGION"]
        self.bucket_name: str = self.configs["S3_BUCKET_NAME"]

        # Initialize S3 client
        self.s3 = boto3.client("s3", region_name=self.region)
        logger.info(f"S3Storage initialized: bucket={self.bucket_name}, region={self.region}")

    def load_config(self) -> Dict[str, str]:
        """Load AWS region and bucket name from config.json"""
        config_info: Dict[str, str] = {}

        # Try multiple possible config locations
        possible_paths = [
            os.path.join(os.path.dirname(__file__), ".aws", "config.json"),
            os.path.join(os.path.dirname(__file__), "config.json"),
            os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
                ".aws",
                "config.json",
            ),
        ]

        config_path = None
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break

        if not config_path:
            raise FileNotFoundError(f"Config file not found. Tried locations: {possible_paths}")

        with open(config_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            config_info["AWS_REGION"] = data.get("AWS_REGION")
            config_info["S3_BUCKET_NAME"] = data.get("S3_BUCKET_NAME")

        logger.info(f"Config loaded from: {config_path}")
        return config_info

    def upload(self, file_obj: BinaryIO, key: str, metadata: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Upload a file object to the S3 bucket.

        Args:
            file_obj: File-like object to upload
            key: S3 object key (path)
            metadata: Optional metadata to attach to object

        Returns:
            Dictionary with success message and S3 location
        """
        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata

            self.s3.upload_fileobj(file_obj, self.bucket_name, key, ExtraArgs=extra_args)
            logger.info(f"Uploaded file to s3://{self.bucket_name}/{key}")

            return {
                "message": "File successfully uploaded",
                "s3_location": f"s3://{self.bucket_name}/{key}",
                "key": key,
            }
        except ClientError as e:
            logger.error(f"S3 upload failed for key {key}: {e}")
            raise StorageUnavailableError("S3 unavailable — failed to upload to bucket")


    def download_url(self, key: str, expiration: int = 3600) -> str:
        """
        Generate a temporary pre-signed URL to download a file.

        Args:
            key: S3 object key
            expiration: URL expiration time in seconds (default 1 hour)

        Returns:
            Pre-signed URL string
        """
        try:
            url = self.s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": key},
                ExpiresIn=expiration,
            )
            logger.info(f"Generated download URL for {key} (expires in {expiration}s)")
            return url
        except ClientError as e:
            logger.error(f"Could not generate download URL for {key}: {e}")
            raise HTTPException(status_code=500, detail=f"Could not generate download URL: {e}")

    def download_to_file(self, key: str, local_path: str) -> None:
        """
        Download S3 object directly to a local file.

        Args:
            key: S3 object key
            local_path: Local file path to save to
        """
        try:
            self.s3.download_file(self.bucket_name, key, local_path)
            logger.info(f"Downloaded s3://{self.bucket_name}/{key} to {local_path}")
        except ClientError as e:
            logger.error(f"Download failed for {key}: {e}")
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise HTTPException(status_code=404, detail=f"File not found: {key}")
            else:
                raise HTTPException(status_code=500, detail=f"Download failed: {e}")

    def file_exists(self, key: str) -> bool:
        """
        Check if a file exists in S3.

        Args:
            key: S3 object key

        Returns:
            True if file exists, False otherwise
        """
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError:
            return False

    def delete_file(self, key: str) -> Dict[str, str]:
        """
        Delete a file from S3.

        Args:
            key: S3 object key

        Returns:
            Dictionary with success message
        """
        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info(f"Deleted s3://{self.bucket_name}/{key}")
            return {"message": f"File deleted successfully: {key}"}
        except ClientError as e:
            logger.error(f"Delete failed for {key}: {e}")
            raise HTTPException(status_code=500, detail=f"Delete failed: {e}")

    def list_files(self, prefix: str = "") -> list:
        """
        List files in S3 with optional prefix filter.

        Args:
            prefix: Optional prefix to filter results

        Returns:
            List of file keys
        """
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)

            files = []
            if "Contents" in response:
                files = [obj["Key"] for obj in response["Contents"]]

            logger.info(f"Listed {len(files)} files with prefix '{prefix}'")
            return files
        except ClientError as e:
            logger.error(f"List failed for prefix {prefix}: {e}")
            raise HTTPException(status_code=500, detail=f"List failed: {e}")

    def get_file_metadata(self, key: str) -> Dict[str, Any]:
        """
        Get metadata about an S3 object.

        Args:
            key: S3 object key

        Returns:
            Dictionary with file metadata
        """
        try:
            response = self.s3.head_object(Bucket=self.bucket_name, Key=key)
            return {
                "size": response.get("ContentLength"),
                "last_modified": response.get("LastModified"),
                "content_type": response.get("ContentType"),
                "metadata": response.get("Metadata", {}),
            }
        except ClientError as e:
            logger.error(f"Could not get metadata for {key}: {e}")
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise HTTPException(status_code=404, detail=f"File not found: {key}")
            else:
                raise HTTPException(status_code=500, detail=f"Metadata retrieval failed: {e}")
