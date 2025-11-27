"""
Helper functions for downloading, zipping, and uploading artifacts to S3.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)


def download_huggingface_artifact(url: str, artifact_name: str, temp_dir: str) -> bool:
    """
    Download a HuggingFace model or dataset.

    Args:
        url: HuggingFace URL
        artifact_name: Name of the artifact
        temp_dir: Temporary directory to download to

    Returns:
        True if successful, False otherwise
    """
    try:
        # Set HF_HOME to use larger disk (not /tmp which is too small)
        os.environ["HF_HOME"] = "/home/ec2-user/.cache/huggingface"

        # Try using huggingface_hub library for proper downloading
        try:
            from huggingface_hub import snapshot_download

            # Extract repo_id from URL
            clean_url = url.strip().rstrip("/")
            parts = clean_url.replace("https://", "").replace("http://", "").split("/")

            repo_id = None
            repo_type = "model"  # default

            if "huggingface.co" in parts[0]:
                if len(parts) > 1 and parts[1] == "datasets":
                    repo_type = "dataset"
                    if len(parts) >= 4:
                        repo_id = f"{parts[2]}/{parts[3]}"
                    elif len(parts) >= 3:
                        repo_id = parts[2]
                elif len(parts) >= 3:
                    repo_id = f"{parts[1]}/{parts[2]}"

            if repo_id:
                logger.info(
                    f"[HF DOWNLOAD] Downloading {repo_type} {repo_id} via huggingface_hub"
                )
                download_dir = os.path.join(temp_dir, artifact_name)
                snapshot_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    local_dir=download_dir,
                    local_dir_use_symlinks=False,
                )
                logger.info(f"[HF DOWNLOAD] Successfully downloaded {repo_id}")
                return True

        except ImportError:
            logger.warning(
                "[HF DOWNLOAD] huggingface_hub not available, trying git-lfs"
            )

        # Fallback: try git clone with git-lfs
        try:
            download_dir = os.path.join(temp_dir, artifact_name)
            os.makedirs(download_dir, exist_ok=True)

            logger.info(f"[GIT DOWNLOAD] Cloning {url} with git-lfs")
            subprocess.run(["git", "lfs", "install"], check=False, capture_output=True)
            result = subprocess.run(
                ["git", "clone", url, download_dir],
                check=True,
                capture_output=True,
                timeout=300,  # 5 minute timeout
            )
            logger.info(f"[GIT DOWNLOAD] Successfully cloned {url}")
            return True

        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ) as e:
            logger.warning(f"[GIT DOWNLOAD] Git clone failed: {e}")

        # Last resort: download what we can via HTTP
        logger.info(f"[HTTP DOWNLOAD] Attempting HTTP download from {url}")
        download_dir = os.path.join(temp_dir, artifact_name)
        os.makedirs(download_dir, exist_ok=True)

        # Download the main page and extract file links
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            # Save a copy of the page
            with open(
                os.path.join(download_dir, "index.html"), "w", encoding="utf-8"
            ) as f:
                f.write(response.text)

            # Try to download a few common files
            common_files = [
                "README.md",
                "config.json",
                "pytorch_model.bin",
                "model.safetensors",
            ]
            for filename in common_files:
                try:
                    file_url = f"{url.rstrip('/')}/resolve/main/{filename}"
                    file_response = requests.get(file_url, timeout=30)
                    if file_response.status_code == 200:
                        filepath = os.path.join(download_dir, filename)
                        with open(filepath, "wb") as f:
                            f.write(file_response.content)
                        logger.info(f"[HTTP DOWNLOAD] Downloaded {filename}")
                except Exception:
                    continue

            return True

        # If we didn't get a 200, explicitly fail the HF download path
        logger.warning(
            f"[HTTP DOWNLOAD] HTTP request returned status {response.status_code} for {url}"
        )
        return False

    except Exception as e:
        logger.error(f"[HF DOWNLOAD] Failed to download HuggingFace artifact: {e}")
        return False


def download_github_repo(url: str, artifact_name: str, temp_dir: str) -> bool:
    """
    Download a GitHub repository.

    Args:
        url: GitHub URL
        artifact_name: Name of the artifact
        temp_dir: Temporary directory to download to

    Returns:
        True if successful, False otherwise
    """
    try:
        download_dir = os.path.join(temp_dir, artifact_name)
        os.makedirs(download_dir, exist_ok=True)

        # Try git clone first
        try:
            logger.info(f"[GITHUB DOWNLOAD] Cloning {url}")
            result = subprocess.run(
                ["git", "clone", "--depth", "1", url, download_dir],
                check=True,
                capture_output=True,
                timeout=300,  # 5 minute timeout
            )
            logger.info(f"[GITHUB DOWNLOAD] Successfully cloned {url}")

            # Remove .git directory to save space
            git_dir = os.path.join(download_dir, ".git")
            if os.path.exists(git_dir):
                shutil.rmtree(git_dir)

            return True

        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ) as e:
            logger.warning(
                f"[GITHUB DOWNLOAD] Git clone failed: {e}, trying HTTP download"
            )

        # Fallback: Download as ZIP from GitHub
        try:
            # Convert URL to download URL
            # https://github.com/owner/repo -> https://github.com/owner/repo/archive/refs/heads/main.zip
            clean_url = url.rstrip("/").replace(".git", "")
            zip_url = f"{clean_url}/archive/refs/heads/main.zip"

            logger.info(f"[GITHUB DOWNLOAD] Downloading ZIP from {zip_url}")
            response = requests.get(zip_url, timeout=60, stream=True)

            if response.status_code == 200:
                zip_path = os.path.join(temp_dir, f"{artifact_name}.zip")
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Extract the zip
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(download_dir)

                os.remove(zip_path)
                logger.info(
                    f"[GITHUB DOWNLOAD] Successfully downloaded and extracted {url}"
                )
                return True
            else:
                logger.error(
                    f"[GITHUB DOWNLOAD] HTTP download failed: {response.status_code}"
                )
                return False

        except Exception as e:
            logger.error(f"[GITHUB DOWNLOAD] ZIP download failed: {e}")
            return False

    except Exception as e:
        logger.error(f"[GITHUB DOWNLOAD] Failed to download GitHub repo: {e}")
        return False


def download_and_zip_artifact(
    url: str, artifact_name: str, artifact_type: str
) -> Optional[str]:
    """
    Download an artifact from a URL and zip it TO A FILE ON DISK (not memory),
    with detailed logging, throughput, ETA estimation, and per-file progress.
    """

    temp_dir = None
    zip_file_path = None

    try:
        logger.info(f"[DOWNLOAD] Starting REAL download for {artifact_name} from {url}")

        # Use large filesystem, not /tmp
        base_temp_dir = "/home/ec2-user/temp"
        os.makedirs(base_temp_dir, exist_ok=True)

        # Create temporary working directory
        temp_dir = tempfile.mkdtemp(
            prefix=f"artifact_{artifact_type}_", dir=base_temp_dir
        )
        logger.info(f"[DOWNLOAD] Using temp directory: {temp_dir}")

        # Download artifact content
        if "huggingface.co" in url:
            download_success = download_huggingface_artifact(
                url, artifact_name, temp_dir
            )
        elif "github.com" in url:
            download_success = download_github_repo(url, artifact_name, temp_dir)
        else:
            logger.error(f"[DOWNLOAD] Unsupported URL: {url}")
            return None

        if not download_success:
            logger.error(f"[DOWNLOAD] Failed to download {artifact_name}")
            return None

        # Begin ZIP phase
        logger.info(f"[DOWNLOAD] Creating zip archive for {artifact_name}")
        zip_file_path = os.path.join(base_temp_dir, f"{artifact_name}.zip")
        artifact_dir = os.path.join(temp_dir, artifact_name)

        # ================= ZIP PROGRESS LOGGING ====================
        logger.info(f"[ZIP] Preparing to zip {artifact_name}...")

        # Count files and compute total size
        total_files = 0
        total_bytes = 0
        for root, dirs, files in os.walk(artifact_dir):
            for f in files:
                total_files += 1
                fp = os.path.join(root, f)
                try:
                    total_bytes += os.path.getsize(fp)
                except:
                    pass

        total_mb = total_bytes / (1024 * 1024)
        logger.info(
            f"[ZIP] {total_files} files detected. Total size: {total_mb:.2f} MB"
        )

        processed = 0
        processed_bytes = 0
        start_time = time.time()

        with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(artifact_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)

                    try:
                        fsize = os.path.getsize(file_path)
                    except:
                        fsize = 0

                    try:
                        zipf.write(file_path, arcname)
                    except Exception as e:
                        logger.warning(f"[ZIP] Failed to add file {file_path}: {e}")
                        continue

                    processed += 1
                    processed_bytes += fsize

                    # Throughput and ETA calculation
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        mbps = (processed_bytes / (1024 * 1024)) / elapsed
                    else:
                        mbps = 0

                    if processed_bytes > 0 and total_bytes > 0:
                        pct = (processed_bytes / total_bytes) * 100
                        remaining_bytes = total_bytes - processed_bytes
                        if mbps > 0:
                            eta_seconds = remaining_bytes / (mbps * 1024 * 1024)
                        else:
                            eta_seconds = 0
                    else:
                        pct = 0
                        eta_seconds = 0

                    # Log progress every file ≤ 50 files, otherwise every 10
                    if (
                        total_files <= 50
                        or processed % 10 == 0
                        or processed == total_files
                    ):
                        logger.info(
                            f"[ZIP] {processed}/{total_files} files "
                            f"({pct:.1f}%) | "
                            f"Throughput: {mbps:.2f} MB/s | "
                            f"ETA: {eta_seconds:.1f}s | "
                            f"Last file: {file} ({fsize/1024/1024:.2f} MB)"
                        )

        # Final ZIP complete log
        elapsed_total = time.time() - start_time
        logger.info(
            f"[ZIP] Completed zip for {artifact_name} in {elapsed_total:.2f}s "
            f"({total_mb:.2f} MB total, avg {total_mb/elapsed_total:.2f} MB/s)."
        )

        zip_size_mb = os.path.getsize(zip_file_path) / (1024 * 1024)
        logger.info(
            f"[DOWNLOAD] Successfully created zip file at {zip_file_path} "
            f"(size: {zip_size_mb:.2f} MB)"
        )

        return zip_file_path

    except Exception as e:
        logger.error(f"[DOWNLOAD] Error downloading/zipping {artifact_name}: {e}")
        return None

    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"[DOWNLOAD] Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"[DOWNLOAD] Failed to clean up temp directory: {e}")


def get_s3_key_for_artifact(artifact_name: str, artifact_type: str) -> str:
    """
    Generate a consistent S3 key for an artifact.

    Args:
        artifact_name: Name of the artifact
        artifact_type: Type (model, dataset, code)

    Returns:
        S3 key string
    """
    # Use a consistent naming scheme: artifacts/{type}/{name}.zip
    safe_name = artifact_name.replace("/", "_").replace(" ", "_")
    return f"artifacts/{artifact_type}/{safe_name}.zip"


def upload_artifact_to_s3(
    s3_storage, artifact_name: str, artifact_type: str, zip_file_path: str
) -> Optional[str]:
    """
    Upload a zipped artifact to S3 FROM A FILE ON DISK.

    Args:
        s3_storage: S3Storage instance
        artifact_name: Name of the artifact
        artifact_type: Type (model, dataset, code)
        zip_file_path: Path to the zip file on disk

    Returns:
        S3 key if successful, None otherwise
    """
    try:
        s3_key = get_s3_key_for_artifact(artifact_name, artifact_type)

        logger.info(f"[S3 UPLOAD] Uploading {artifact_name} to S3: {s3_key}")

        # Upload file from disk to S3
        metadata = {"artifact_name": artifact_name, "artifact_type": artifact_type}

        # Open file and upload
        with open(zip_file_path, "rb") as f:
            result = s3_storage.upload(f, s3_key, metadata=metadata)

        logger.info(f"[S3 UPLOAD] Successfully uploaded {artifact_name} to S3")

        # Clean up the zip file after successful upload
        try:
            os.remove(zip_file_path)
            logger.info(f"[S3 UPLOAD] Cleaned up zip file: {zip_file_path}")
        except Exception as e:
            logger.warning(f"[S3 UPLOAD] Failed to clean up zip file: {e}")

        return s3_key

    except Exception as e:
        logger.error(f"[S3 UPLOAD] Error uploading {artifact_name} to S3: {e}")
        # Try to clean up zip file even on failure
        try:
            if os.path.exists(zip_file_path):
                os.remove(zip_file_path)
        except:
            pass
        return None


def check_artifact_exists_in_s3(
    s3_storage, artifact_name: str, artifact_type: str
) -> bool:
    """
    Check if an artifact already exists in S3.

    Args:
        s3_storage: S3Storage instance
        artifact_name: Name of the artifact
        artifact_type: Type (model, dataset, code)

    Returns:
        True if artifact exists in S3, False otherwise
    """
    try:
        s3_key = get_s3_key_for_artifact(artifact_name, artifact_type)
        exists = s3_storage.file_exists(s3_key)

        if exists:
            logger.info(f"[S3 CHECK] Artifact {artifact_name} already exists in S3")
        else:
            logger.info(f"[S3 CHECK] Artifact {artifact_name} not found in S3")

        return exists

    except Exception as e:
        logger.error(f"[S3 CHECK] Error checking if {artifact_name} exists in S3: {e}")
        return False


def get_artifact_download_url(
    s3_storage, artifact_name: str, artifact_type: str
) -> Optional[str]:
    """
    Get a pre-signed download URL for an artifact in S3.

    Args:
        s3_storage: S3Storage instance
        artifact_name: Name of the artifact
        artifact_type: Type (model, dataset, code)

    Returns:
        Pre-signed URL if successful, None otherwise
    """
    try:
        s3_key = get_s3_key_for_artifact(artifact_name, artifact_type)

        # Generate a pre-signed URL (valid for 1 hour)
        download_url = s3_storage.download_url(s3_key, expiration=3600)

        logger.info(f"[S3 DOWNLOAD URL] Generated download URL for {artifact_name}")
        return download_url

    except Exception as e:
        logger.error(
            f"[S3 DOWNLOAD URL] Error generating download URL for {artifact_name}: {e}"
        )
        return None


def process_artifact_for_s3(
    s3_storage,
    url: str,
    artifact_name: str,
    artifact_type: str,
    force_redownload: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Complete workflow: Check S3 cache, download/zip/upload if needed, return download URL.

    Args:
        s3_storage: S3Storage instance
        url: Source URL
        artifact_name: Name of the artifact
        artifact_type: Type (model, dataset, code)
        force_redownload: If True, re-download even if exists in S3

    Returns:
        Tuple of (s3_key, download_url) or (None, None) if failed
    """
    try:
        # Check if artifact already exists in S3
        if not force_redownload and check_artifact_exists_in_s3(
            s3_storage, artifact_name, artifact_type
        ):
            logger.info(f"[S3 CACHE HIT] Using existing S3 object for {artifact_name}")
            s3_key = get_s3_key_for_artifact(artifact_name, artifact_type)
            download_url = get_artifact_download_url(
                s3_storage, artifact_name, artifact_type
            )
            return s3_key, download_url

        # Download and zip the artifact (returns file path on disk)
        zip_file_path = download_and_zip_artifact(url, artifact_name, artifact_type)
        if not zip_file_path:
            logger.error(f"[S3 PROCESS] Failed to download/zip {artifact_name}")
            return None, None

        # Upload to S3 (this will clean up the zip file after upload)
        s3_key = upload_artifact_to_s3(
            s3_storage, artifact_name, artifact_type, zip_file_path
        )
        if not s3_key:
            logger.error(f"[S3 PROCESS] Failed to upload {artifact_name} to S3")
            return None, None

        # Generate download URL
        download_url = get_artifact_download_url(
            s3_storage, artifact_name, artifact_type
        )
        if not download_url:
            logger.error(
                f"[S3 PROCESS] Failed to generate download URL for {artifact_name}"
            )
            return None, None

        logger.info(f"[S3 PROCESS] Successfully processed {artifact_name}")
        return s3_key, download_url

    except Exception as e:
        logger.error(f"[S3 PROCESS] Error processing {artifact_name}: {e}")
        return None, None
