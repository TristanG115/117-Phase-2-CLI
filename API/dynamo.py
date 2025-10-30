from __future__ import annotations
import boto3, json, os, uuid
from datetime import datetime
from typing import Any, Dict, Optional
from botocore.exceptions import ClientError


class DynamoDB:
    # Handles all DynamoDB operations for the Trustworthy Model Registry

    def __init__(self, table_name: str = "TrustworthyModelRegDB"):
        self.configs = self.load_config()
        self.region: str = self.configs["AWS_REGION"]
        self.bucket_name: str = self.configs["S3_BUCKET_NAME"]

        # Initialize database connection
        self.dynamo = boto3.resource("dynamodb", region_name = self.region)
        self.table = self.dynamo.Table(table_name)

    def load_config(self) -> Dict[str, str]:
        config_info: Dict[str, str] = {}
        config_path = os.path.join(
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), os.pardir, ".aws", "config.json")
            )
        )

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            config_info["AWS_REGION"] = data.get("AWS_REGION")
            config_info["S3_BUCKET_NAME"] = data.get("S3_BUCKET_NAME")
        return config_info

    def add_model(self, **kwargs: Any) -> Dict[str, Any]:
        # Insert a new model record into database
        model_id = str(uuid.uuid4())
        item = {
            "model_id": model_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **kwargs,
        }

        try:
            self.table.put_item(Item=item)
            return item
        except ClientError as e:
            raise RuntimeError(f"Failed to insert model: {e}")

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        # Fetch a model record by model_id
        try:
            response = self.table.get_item(Key = {"model_id": model_id})
            return response.get("Item")
        except ClientError as e:
            raise RuntimeError(f"Failed to get model: {e}")

    def update_model(self, model_id: str, updates: Dict[str, Any]) -> None:
        # Update specific fields of a model record
        update_expr = "SET " + ", ".join(f"{k} = :{k}" for k in updates)
        expr_values = {f":{k}": v for k, v in updates.items()}
        try:
            self.table.update_item(
                Key = {"model_id": model_id},
                UpdateExpression = update_expr,
                ExpressionAttributeValues = expr_values,
            )
        except ClientError as e:
            raise RuntimeError(f"Failed to update model: {e}")