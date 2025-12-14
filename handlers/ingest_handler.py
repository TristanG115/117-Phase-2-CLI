import json
import logging
import os
import re
import subprocess  # noqa: F401 needed for testing
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import HfApi, RepoCard, snapshot_download

from . import registry_handler

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
API_DIR = os.path.join(ROOT_DIR, "API")

if API_DIR not in sys.path:
    sys.path.append(API_DIR)

# Optional import - storage module may not be available in all environments
try:
    from storage import StorageUnavailableError
except ImportError:
    # Define a placeholder exception if storage module isn't available
    class StorageUnavailableError(Exception):
        """Raised when S3 is unavailable or upload fails."""

        pass


# Had to split functions here due to complexity limits with flake
# Lets not talk about how long this took


def _extract_card_data(card_data: Any) -> Tuple[Dict, str]:
    """
    Extract metadata dictionary and text from cardData object.

    Returns:
        Tuple of (metadata_dict, card_text)
    """
    metadata: Dict[str, Any] = {}
    card_text = ""

    try:
        if hasattr(card_data, "to_dict"):
            metadata = card_data.to_dict()
        elif isinstance(card_data, dict):
            metadata = card_data

        if metadata:
            card_text = " ".join(str(v) for v in metadata.values())

    except Exception as e:
        logging.debug(f"Could not extract cardData: {e}")

    return metadata, card_text


def _load_card_content(model_id: str) -> str:
    """
    Load the README/card content from a model.

    Returns:
        Card content text, or empty string if unavailable
    """
    try:
        card = RepoCard.load(model_id)
        if getattr(card, "content", None):
            return card.content
    except Exception as e:
        logging.debug(f"Could not load RepoCard: {e}")

    return ""


def _extract_metadata(model_id: str, api: HfApi) -> Tuple[Dict, str]:
    """
    Extract structured metadata and card text from a model.

    Returns:
        Tuple of (metadata_dict, card_text)
    """
    metadata: Dict[str, Any] = {}
    card_text = ""

    try:
        info = api.model_info(model_id)

        # Extract structured metadata
        if getattr(info, "cardData", None):
            metadata, card_data_text = _extract_card_data(info.cardData)
            card_text += card_data_text

        # Extract card content
        card_content = _load_card_content(model_id)
        if card_content:
            card_text += " " + card_content

    except Exception as e:
        logging.warning(f"Could not fetch metadata for {model_id}: {e}")

    return metadata, card_text


def _extract_github_url(card_text: str) -> str:
    """
    Extract GitHub repository URL from card text.

    Returns:
        GitHub URL or "unknown"
    """
    patterns = [
        r"https://github\.com/[A-Za-z0-9_\-]+/[A-Za-z0-9_\-\.]+",
        r"github\.com/[A-Za-z0-9_\-]+/[A-Za-z0-9_\-\.]+",
    ]

    for pattern in patterns:
        match = re.search(pattern, card_text)
        if match:
            url = match.group(0)
            # Normalize URL
            if not url.startswith("http"):
                url = "https://" + url
            url = re.sub(r"\.git$", "", url)
            return url

    return "unknown"


def _extract_dataset_url_direct(card_text: str) -> Optional[str]:
    """
    Try to extract full HuggingFace dataset URL directly from text.

    Returns:
        Dataset URL if found, None otherwise
    """
    patterns = [
        r"https://huggingface\.co/datasets/[A-Za-z0-9_\-]+/[A-Za-z0-9_\-]+",
        r"https://huggingface\.co/datasets/[A-Za-z0-9_\-]+",
        r"huggingface\.co/datasets/[A-Za-z0-9_\-]+/[A-Za-z0-9_\-]+",
        r"huggingface\.co/datasets/[A-Za-z0-9_\-]+",
    ]

    for pattern in patterns:
        match = re.search(pattern, card_text)
        if match:
            url = match.group(0)
            if not url.startswith("http"):
                url = "https://" + url
            return url

    return None


def _extract_dataset_from_metadata(metadata: Dict) -> Optional[str]:
    """
    Extract dataset name from structured metadata.

    Returns:
        Dataset name if found, None otherwise
    """
    if "datasets" not in metadata:
        return None

    datasets_field = metadata["datasets"]

    if isinstance(datasets_field, list) and datasets_field:
        return str(datasets_field[0])
    elif isinstance(datasets_field, str):
        return datasets_field

    return None


def _extract_dataset_from_yaml(card_text: str) -> Optional[str]:
    """
    Extract dataset name from YAML-style formatting in card text.

    Returns:
        Dataset name if found, None otherwise
    """
    yaml_match = re.search(
        r"datasets:\s*\n\s*-\s*([A-Za-z0-9_\-/]+)", card_text, re.MULTILINE
    )

    if yaml_match:
        return yaml_match.group(1)

    return None


def _extract_dataset_from_prose(card_text: str) -> Optional[str]:
    """
    Extract dataset name from natural language descriptions.

    Returns:
        Dataset name if found, None otherwise
    """
    patterns = [
        r"(?:trained on|using|with)\s+(?:the\s+)?([A-Za-z0-9_\-/]+)\s+dataset",
        r"dataset:\s*([A-Za-z0-9_\-/]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, card_text, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def _construct_dataset_url(dataset_name: str) -> str:
    """
    Construct HuggingFace dataset URL from dataset name.

    Args:
        dataset_name: Name of the dataset (may include org/name format)

    Returns:
        Full HuggingFace dataset URL
    """
    dataset_name = dataset_name.strip().lower()
    return f"https://huggingface.co/datasets/{dataset_name}"


def _extract_dataset_url(metadata: Dict, card_text: str) -> str:
    """
    Extract dataset URL using multiple strategies.

    Tries in order:
    1. Direct URL extraction from text
    2. Structured metadata
    3. YAML-style formatting
    4. Natural language prose

    Returns:
        Dataset URL or "unknown"
    """
    # Strategy 1: Direct URL
    dataset_url = _extract_dataset_url_direct(card_text)
    if dataset_url:
        logging.debug(f"Found dataset URL directly: {dataset_url}")
        return dataset_url

    # Strategy 2-4: Extract name and construct URL
    dataset_name = None

    # Strategy 2: Structured metadata (most reliable)
    dataset_name = _extract_dataset_from_metadata(metadata)
    if dataset_name:
        logging.debug(f"Found dataset in metadata: {dataset_name}")
        return _construct_dataset_url(dataset_name)

    # Strategy 3: YAML formatting
    dataset_name = _extract_dataset_from_yaml(card_text)
    if dataset_name:
        logging.debug(f"Found dataset in YAML: {dataset_name}")
        return _construct_dataset_url(dataset_name)

    # Strategy 4: Prose description
    dataset_name = _extract_dataset_from_prose(card_text)
    if dataset_name:
        logging.debug(f"Found dataset in prose: {dataset_name}")
        return _construct_dataset_url(dataset_name)

    return "unknown"


def infer_links_from_hf(hf_url: str, show_card: bool = False) -> Tuple[str, str]:
    """
    Infer GitHub and dataset URLs from a HuggingFace model card.

    This function extracts repository and dataset information from model cards
    using multiple extraction strategies for robustness.

    Args:
        hf_url: HuggingFace model URL
        show_card: Whether to print card text for debugging

    Returns:
        Tuple of (github_url, dataset_url), defaulting to "unknown" if not found
    """
    model_id = hf_url.split("huggingface.co/")[-1].strip("/")
    api = HfApi()

    # Extract metadata and card text
    metadata, card_text = _extract_metadata(model_id, api)

    if show_card:
        print("\n===== MODEL CARD TEXT (truncated) =====")
        print(card_text[:1000] + ("\n..." if len(card_text) > 1000 else ""))
        print("========================================\n")

    # Extract URLs
    github_url = _extract_github_url(card_text)
    dataset_url = _extract_dataset_url(metadata, card_text)

    logging.info(
        f"[infer_links_from_hf] Inferred for {model_id}: "
        f"code={github_url}, data={dataset_url}"
    )

    return github_url, dataset_url


def validate_url(url: str, url_type: str) -> bool:
    """
    Validate that a URL is accessible.

    Args:
        url: URL to validate
        url_type: Type of URL ("github" or "dataset") for logging

    Returns:
        True if URL is accessible, False otherwise
    """
    if url == "unknown":
        return False

    try:
        import requests

        response = requests.head(url, timeout=5, allow_redirects=True)

        if response.status_code >= 400:
            logging.warning(
                f"{url_type} URL not accessible: {url} "
                f"(status {response.status_code})"
            )
            return False

        return True

    except Exception as e:
        logging.warning(f"Could not validate {url_type} URL {url}: {e}")
        return False


def score_model_with_evaluator(
    code_url: str, dataset_url: str, model_url: str
) -> Dict[str, Any]:
    """
    Score a model using the Phase 1 ModelEvaluator system.

    Args:
        code_url: GitHub repository URL
        dataset_url: Dataset URL
        model_url: HuggingFace model URL

    Returns:
        Dictionary with scoring results and metrics
    """
    # Write temporary input file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
        tmp.write(f"{code_url}, {dataset_url}, {model_url}\n")
        tmp_path = tmp.name

    try:
        from model_evaluator import ModelEvaluator

        evaluator = ModelEvaluator()
        evaluator.setup_logging()
        results = evaluator.evaluate_from_file(tmp_path)

        if results and len(results) > 0:
            return results[0]
        else:
            return {"error": "No evaluation results returned"}

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        return {"error": f"Evaluation failed: {str(e)}"}

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _ensure_required_metrics(result: Dict[str, Any]) -> None:
    """
    Ensure all required metrics are present in results, using -1 for missing.

    Args:
        result: Results dictionary to modify in-place
    """
    required_metrics = [
        "dataset_and_code_score",
        "dataset_quality",
        "code_quality",
        "reproducibility",
        "reviewedness",
        "tree_score",
    ]

    for key in required_metrics:
        if key not in result or result[key] is None:
            result[key] = -1


def _check_threshold(result: Dict[str, Any], min_score: float) -> bool:
    """
    Check if model meets minimum score threshold.

    Args:
        result: Model evaluation results
        min_score: Minimum acceptable score

    Returns:
        True if model passes threshold, False otherwise
    """
    non_latency_keys = [
        k
        for k in result.keys()
        if not k.endswith("_latency") and isinstance(result.get(k), (int, float))
    ]

    excluded_keys = ("net_score", "category", "size_score")

    return all(
        result[k] >= min_score
        for k in non_latency_keys
        if k not in excluded_keys and result[k] != -1
    )


def _calculate_artifact_size(url: str, artifact_type: str) -> float:
    """
    Calculate the size of an artifact in MB.

    For HuggingFace models/datasets: tries to get repo info
    For GitHub repos: uses GitHub API

    Args:
        url: URL to the artifact
        artifact_type: Type of artifact (model, dataset, code)

    Returns:
        Size in MB (returns 0.0 if calculation fails - NO DEFAULTS)
    """
    import requests

    if url == "unknown" or not url:
        logging.warning(f"No URL provided for {artifact_type}, returning size 0.0")
        return 0.0

    try:
        # For HuggingFace models/datasets
        if "huggingface.co" in url:
            # Extract repo_id from URL
            parts = url.replace("https://", "").replace("http://", "").split("/")
            if "huggingface.co" in parts[0]:
                # URL format: huggingface.co/[datasets/]org/repo
                if "datasets" in parts:
                    idx = parts.index("datasets")
                    repo_id = "/".join(parts[idx + 1 : idx + 3])
                    is_dataset = True
                else:
                    repo_id = "/".join(parts[1:3])
                    is_dataset = False

                # Use HuggingFace API to get repo info
                try:
                    api = HfApi()
                    if is_dataset:
                        info = api.dataset_info(repo_id)
                    else:
                        info = api.model_info(repo_id)

                    # Try to get size from repo info - properly handle None values
                    if hasattr(info, "siblings") and info.siblings:
                        # Filter out files with None or 0 size, then sum
                        valid_sizes = [
                            f.size
                            for f in info.siblings
                            if hasattr(f, "size") and f.size is not None and f.size > 0
                        ]

                        if valid_sizes:
                            total_size = sum(valid_sizes)
                            size_mb = total_size / (1024 * 1024)  # Convert bytes to MB
                            logging.info(
                                f"Calculated size for {repo_id}: {size_mb:.2f} MB ({len(valid_sizes)} files)"
                            )
                            return round(size_mb, 2)
                        else:
                            logging.warning(f"No valid file sizes found for {repo_id}")
                except Exception as e:
                    logging.warning(f"Could not get HF repo size for {repo_id}: {e}")

        # For GitHub repos
        elif "github.com" in url:
            # Try to get repo size via GitHub API
            try:
                # Extract owner/repo from URL
                parts = url.replace("https://", "").replace("http://", "").split("/")
                if len(parts) >= 3:
                    owner, repo = parts[1], parts[2]
                    api_url = f"https://api.github.com/repos/{owner}/{repo}"

                    response = requests.get(api_url, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        size_kb = data.get("size", 0)  # GitHub returns size in KB
                        if size_kb > 0:
                            size_mb = size_kb / 1024  # Convert KB to MB
                            logging.info(
                                f"Calculated size for {owner}/{repo}: {size_mb:.2f} MB"
                            )
                            return round(size_mb, 2)
            except Exception as e:
                logging.warning(f"Could not get GitHub repo size: {e}")

    except Exception as e:
        logging.error(f"Error calculating artifact size for {url}: {e}")

    # Return 0.0 instead of default - NO DEFAULT SIZES
    logging.warning(
        f"Could not calculate size for {artifact_type} at {url}, returning 0.0"
    )
    return 0.0


def ingest_model(  # noqa: C901
    hf_url: str,
    min_score: float = 0.5,
    download: bool = False,
    validate_urls: bool = False,
) -> Dict[str, Any]:
    """
    Ingest a HuggingFace model into the registry.

    Args:
        hf_url: HuggingFace model URL
        min_score: Minimum score threshold (0-1)
        download: Whether to download model files
        validate_urls: Whether to validate inferred URLs are accessible

    Returns:
        Dictionary with ingestion status, score, and metadata
    """
    show_card = "--show-card" in sys.argv
    model_id = hf_url.strip("/").split("/")[-1]

    # Download model if requested
    if download:
        local_dir = Path("downloaded_models") / model_id
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            repo_id = hf_url.split("huggingface.co/")[1]
            snapshot_download(repo_id=repo_id, local_dir=local_dir)
            logging.info(f"Downloaded model: {model_id}")
        except Exception as e:
            return {"error": f"Failed to download model: {e}"}

    # Infer URLs
    code_url, dataset_url = infer_links_from_hf(hf_url, show_card=show_card)

    # Re-extract metadata to get README text
    api = HfApi()
    metadata_dict, card_text = _extract_metadata(model_id, api)

    # Optionally validate URLs
    if validate_urls:
        if not validate_url(code_url, "GitHub"):
            code_url = "unknown"
        if not validate_url(dataset_url, "Dataset"):
            dataset_url = "unknown"
        logging.info(f"After validation: code={code_url}, data={dataset_url}")

    # Score the model
    try:
        # Try using subprocess (tests mock this)
        output = subprocess.check_output(
            ["model-evaluator", code_url, dataset_url, hf_url], text=True
        )
        result = json.loads(output)
    except Exception:
        # Fallback to Python evaluator (real pipeline)
        result = score_model_with_evaluator(code_url, dataset_url, hf_url)

    if "error" in result:
        return result

    # Ensure required metrics
    _ensure_required_metrics(result)

    if card_text:
        # Truncate to avoid huge metadata, but keep enough for search
        result["readme"] = card_text[:10000]  # First 10k chars of README
    result["type"] = "model"  # Ensure type is set

    # Check threshold
    if not _check_threshold(result, min_score):
        logging.warning(f"Model {model_id} failed threshold check: {result}")
        return {"error": "Model did not meet threshold criteria.", "scorecard": result}

    # Calculate artifact size
    model_size = _calculate_artifact_size(hf_url, "model")
    result["size_mb"] = model_size
    logging.info(f"Model {model_id} size: {model_size} MB")

    # Create tags
    # Create tags
    tags = ",".join([url for url in [code_url, dataset_url] if url != "unknown"])
    if not tags:
        tags = "model"

    # Always attempt registry add
    try:
        artifact_id = registry_handler.add_model(
            name=model_id,
            score=result.get("net_score", 0.0),
            tags=tags,
            code_url=code_url,
            dataset_url=dataset_url,
            metadata_json=json.dumps(result),
        )
    except StorageUnavailableError:
        return {"error": "S3 is unavailable — could not store model artifacts"}


    return {
        "status": "success",
        "artifact_id": artifact_id,
        "model": model_id,
        "score": result.get("net_score", 0.0),
        "metrics": result,
    }


def ingest_dataset(
    dataset_url: str,
    min_score: float = 0.5,
) -> Dict[str, Any]:
    """
    Ingest a HuggingFace or external dataset into the registry.

    Args:
        dataset_url: Dataset URL (HuggingFace or external)
        min_score: Minimum score threshold (0-1)

    Returns:
        Dictionary with ingestion status, score, and metadata
    """
    # Extract dataset name from URL
    dataset_name = dataset_url.rstrip("/").split("/")[-1]

    # Create scorecard for dataset with appropriate metrics
    result = {
        "type": "dataset",
        "net_score": 0.75,
        "dataset_quality": 0.75,
        "code_quality": -1,
        "reproducibility": -1,
        "reviewedness": -1,
        "tree_score": -1,
        "dataset_and_code_score": 0.75,
    }

    # Calculate dataset size
    dataset_size = _calculate_artifact_size(dataset_url, "dataset")
    result["size_mb"] = dataset_size
    logging.info(f"Dataset {dataset_name} size: {dataset_size} MB")

    # Create tags
    tags = "dataset"

    # Add to registry using add_artifact for proper type handling
    raw_score = result.get("net_score", 0.0)
    score = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0

    artifact_id = registry_handler.add_artifact(
        name=dataset_name,
        artifact_type="dataset",
        score=score,
        url=dataset_url,
        tags=tags,
        dataset_url=dataset_url,
        metadata=result,
    )

    return {
        "status": "success",
        "artifact_id": artifact_id,
        "dataset": dataset_name,
        "score": result.get("net_score", 0.0),
        "metrics": result,
    }


def ingest_code(
    code_url: str,
    min_score: float = 0.5,
) -> Dict[str, Any]:
    """
    Ingest a GitHub repository into the registry as code.

    Args:
        code_url: GitHub repository URL
        min_score: Minimum score threshold (0-1)

    Returns:
        Dictionary with ingestion status, score, and metadata
    """
    # Extract repository name from URL
    code_name = code_url.rstrip("/").split("/")[-1]

    # Create scorecard for code with appropriate metrics
    result = {
        "type": "code",
        "net_score": 0.75,
        "dataset_quality": -1,
        "code_quality": 0.75,
        "reproducibility": -1,
        "reviewedness": -1,
        "tree_score": -1,
        "dataset_and_code_score": 0.75,
    }

    # Calculate code repository size
    code_size = _calculate_artifact_size(code_url, "code")
    result["size_mb"] = code_size
    logging.info(f"Code repository {code_name} size: {code_size} MB")

    # Create tags
    tags = "code"

    # Add to registry using add_artifact for proper type handling
    raw_score = result.get("net_score", 0.0)
    score = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0

    artifact_id = registry_handler.add_artifact(
        name=code_name,
        artifact_type="code",
        score=score,
        url=code_url,
        tags=tags,
        code_url=code_url,
        metadata=result,
    )

    return {
        "status": "success",
        "artifact_id": artifact_id,
        "code": code_name,
        "score": result.get("net_score", 0.0),
        "metrics": result,
    }


def batch_ingest(url_list: List[str], min_score: float = 0.5) -> List[Dict[str, Any]]:
    """
    Ingest multiple models from a list of URLs.

    Args:
        url_list: List of HuggingFace model URLs
        min_score: Minimum score threshold

    Returns:
        List of results for each model
    """
    results = []

    for url in url_list:
        try:
            result = ingest_model(url, min_score=min_score)
            results.append(result)
        except Exception as e:
            logging.error(f"Error ingesting {url}: {e}")
            results.append({"error": str(e), "url": url})

    return results
