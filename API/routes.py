from __future__ import annotations
from fastapi import APIRouter, HTTPException, Form
from huggingface_hub import hf_hub_download
import tempfile, os
from storage import S3Storage
from dynamo import DynamoDB

router = APIRouter(prefix = "/models", tags = ["Models"])
s3 = S3Storage()
db = DynamoDB()

@router.get("/{model_id}")
def get_model_info(model_id: str):
    model = db.get_model(model_id)
    if not model:
        raise HTTPException(status_code = 404, detail = "Model not found")
    return model

"""
@router.get("/{model_id}/download")
def download_model(model_id: str):
"""