import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, RepoCard, snapshot_download

from .registry_handler import add_model


def infer_links_from_hf(hf_url: str, show_card: bool = False):
    """
    Try to infer GitHub and dataset URLs from the Hugging Face model card.
    - Reads both structured metadata (cardData) and README content.
    - Extracts links via regex.
    - If only dataset names are listed, attempts to construct Hugging Face dataset URLs.
    Returns ("unknown", "unknown") if nothing can be inferred.
    """
    model_id = hf_url.split("huggingface.co/")[-1].strip("/")
    api = HfApi()
    github_url, dataset_url = "unknown", "unknown"

    try:
        # === STEP 1: Collect model card text ===
        info = api.model_info(model_id)
        card_text = ""

        # Extract structured metadata if available
        if getattr(info, "cardData", None):
            try:
                if hasattr(info.cardData, "to_dict"):
                    meta_dict = info.cardData.to_dict()
                elif isinstance(info.cardData, dict):
                    meta_dict = info.cardData
                else:
                    meta_dict = {}
                card_text += " ".join(str(v) for v in meta_dict.values())
            except Exception:
                pass

        # Append README / full model card content
        try:
            card = RepoCard.load(model_id)
            if getattr(card, "content", None):
                card_text += " " + card.content
        except Exception:
            pass

        if show_card:
            print("\n===== MODEL CARD TEXT (truncated) =====")
            print(card_text[:1000] + ("\n..." if len(card_text) > 1000 else ""))
            print("========================================\n")

        # === STEP 2: Extract links ===
        github_match = re.search(r"https://github\.com/[^\s)]+", card_text)
        dataset_match = re.search(
            r"https://huggingface\.co/datasets/[^\s)]+", card_text
        )

        if github_match:
            github_url = github_match.group(0)
        if dataset_match:
            dataset_url = dataset_match.group(0)

        # === STEP 3: Fallback to dataset name parsing ===
        if dataset_url == "unknown":
            # Look for YAML-style datasets: entries like "- bookcorpus" or "- wikipedia"
            dataset_names = re.findall(
                r"datasets:\s*(?:- )?([A-Za-z0-9_\-/]+)", card_text
            )
            if not dataset_names:
                dataset_names = re.findall(r"-\s*([A-Za-z0-9_\-/]+)", card_text)

            if dataset_names:
                ds_name = dataset_names[0].strip()
                # If dataset appears as "bookcorpus" → /datasets/bookcorpus/bookcorpus
                if "/" not in ds_name:
                    dataset_url = f"https://huggingface.co/datasets/{ds_name}/{ds_name}"
                else:
                    dataset_url = f"https://huggingface.co/datasets/{ds_name}"

    except Exception as e:
        logging.warning(
            f"[infer_links_from_hf] Could not fetch metadata for {model_id}: {e}"
        )

    logging.info(
        f"[infer_links_from_hf] Inferred for {model_id}: code={github_url}, data={dataset_url}"
    )
    return github_url, dataset_url


def ingest_model(hf_url: str, min_score: float = 0.5):
    """
    Download and score a HuggingFace model, then add it to the local registry
    if it meets quality thresholds. Automatically infers missing code/dataset links.
    Metrics that cannot be calculated are marked as -1.
    """
    show_card = "--show-card" in sys.argv
    model_id = hf_url.strip("/").split("/")[-1]
    local_dir = Path("downloaded_models") / model_id
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=hf_url.split("huggingface.co/")[1], local_dir=local_dir
        )
        logging.info(f"Downloaded model: {model_id}")
    except Exception as e:
        return {"error": f"Failed to download model: {e}"}

    code_url, dataset_url = infer_links_from_hf(hf_url, show_card=show_card)

    # Write temporary input file for scoring, score needs a file input
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        tmp.write(f"{code_url}, {dataset_url}, {hf_url}\n")
        tmp_path = tmp.name

    cmd = ["python", "run.py", tmp_path]
    try:
        output = subprocess.check_output(cmd, text=True)
        result = json.loads(output.strip().split("\n")[-1])
    except Exception as e:
        return {"error": f"Scoring failed: {e}"}

    for key in [
        "dataset_and_code_score",
        "dataset_quality",
        "code_quality",
        "reproducibility",
        "reviewedness",
        "treescore",
    ]:
        if key not in result or result[key] is None:
            result[key] = -1

    non_latency = [
        k
        for k in result.keys()
        if not k.endswith("_latency") and isinstance(result[k], (int, float))
    ]
    passed = all(
        result[k] >= min_score
        for k in non_latency
        if k not in ("net_score", "category", "size_score") and result[k] != -1
    )

    if not passed:
        logging.warning(f"Model {model_id} failed threshold check: {result}")
        return {"error": "Model did not meet threshold criteria.", "scorecard": result}

    tags = (
        ",".join([url for url in [code_url, dataset_url] if url != "unknown"]) or "none"
    )
    add_model(
        name=model_id,
        score=result.get("net_score", 0.0),
        tags=tags,
        code_url=code_url,
        dataset_url=dataset_url,
        metadata_json=json.dumps(result),
    )

    return {
        "status": "success",
        "model": model_id,
        "score": result.get("net_score", 0.0),
    }
