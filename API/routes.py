from __future__ import annotations
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse
import tempfile, os, zipfile
from .storage import S3Storage
from .dynamo import DynamoDB

router = APIRouter(prefix="/models", tags=["Models"])
s3 = S3Storage()
db = DynamoDB()


@router.get("/{model_id}")
def get_model_info(model_id: str):
    model = db.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.post("/{model_id}/upload")
async def upload_model(model_id: str, file: UploadFile):
    # Uploads a model to S3 as a .zip file
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the uploaded file locally
            file_path = os.path.join(tmpdir, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            # Create a ZIP archive
            zip_path = os.path.join(tmpdir, f"{model_id}.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(file_path, arcname=file.filename)

            # Upload the ZIP file to S3
            with open(zip_path, "rb") as f:
                s3_key = f"{model_id}.zip"
                s3.upload(f, s3_key)

        return {"message": "Model uploaded successfully", "s3_key": s3_key}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/{model_id}/download")
def download_model(model_id: str):
    # Downloads model file from S3
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            s3_key = f"{model_id}.zip"
            local_path = os.path.join(tmpdir, f"{model_id}.zip")

            # Download file from S3 into local temp
            s3.s3.download_file(s3.bucket_name, s3_key, local_path)

            if not zipfile.is_zipfile(local_path):
                raise HTTPException(status_code=400, detail="Downloaded file is not a valid ZIP")

            return FileResponse(local_path, filename=f"{model_id}.zip", media_type="application/zip")

    except s3.s3.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Model not found in storage")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")
