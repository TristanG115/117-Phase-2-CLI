"""
Artifact Lineage Extraction Module using Purdue GenAI Studio

This module uses the Purdue GenAI Studio LLM API to parse README files
and extract dependencies between models, datasets, and code repositories.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class LineageExtractor:
    """Extract artifact lineage from README content using Purdue GenAI Studio LLM"""

    def __init__(self):
        """Initialize the lineage extractor"""
        self.api_available = True
        self.api_key = os.getenv("GEN_AI_STUDIO_API_KEY")

        if not self.api_key:
            logger.warning("GEN_AI_STUDIO_API_KEY not set, will use regex fallback")
            self.api_available = False

    def _call_genai_api(self, prompt: str, readme_content: str) -> Optional[Dict]:
        """
        Call Purdue GenAI Studio API to analyze README content

        Args:
            prompt: The instruction prompt for the LLM
            readme_content: The README text to analyze

        Returns:
            Dictionary with extracted information, or None if error
        """
        if not self.api_key:
            logger.warning("No API key available for GenAI Studio")
            return None

        try:
            logger.info("Calling GenAI Studio API for lineage extraction")

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": "llama4:latest",
                "messages": [
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nREADME Content:\n{readme_content}",
                    }
                ],
                "temperature": 0.0,
            }

            response = requests.post(
                "https://genai.api.purdue.edu/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"].strip()

                # Try to parse as JSON
                try:
                    # Remove markdown code blocks if present
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]

                    result = json.loads(content.strip())
                    logger.info("Successfully parsed LLM response as JSON")
                    return result

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse LLM response as JSON: {e}")
                    logger.debug(f"Raw response: {content}")
                    return None

            else:
                logger.warning(f"GenAI API returned status {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.error("GenAI API call timed out")
            return None
        except Exception as e:
            logger.error(f"Error calling GenAI API: {e}", exc_info=True)
            return None

    def extract_lineage(
        self, readme_content: str, model_name: str
    ) -> Dict[str, List[str]]:
        """
        Extract artifact lineage from README content

        Args:
            readme_content: The README text content
            model_name: Name of the model being analyzed

        Returns:
            Dictionary containing:
            - datasets: List of dataset URLs/names
            - code_repos: List of code repository URLs
            - parent_models: List of parent/base model names
            - evaluation_datasets: List of evaluation dataset names
        """
        if not readme_content or len(readme_content.strip()) < 50:
            logger.warning(f"README content too short for {model_name}")
            return self._fallback_extraction(readme_content)

        # Truncate very long READMEs to avoid token limits
        if len(readme_content) > 10000:
            readme_content = readme_content[:10000] + "\n...[truncated]"

        prompt = """You are analyzing a machine learning model's README file to extract artifact lineage information.

Please analyze the README and extract the following information in JSON format:

{
  "training_datasets": ["list of dataset names or URLs used for training"],
  "code_repositories": ["list of GitHub or code repository URLs"],
  "parent_models": ["list of parent/base model names that this model was fine-tuned from"],
  "evaluation_datasets": ["list of dataset names used for evaluation/testing"]
}

IMPORTANT INSTRUCTIONS:
1. For datasets: Look for mentions of training data, datasets used, or data sources
2. For code repositories: Look for GitHub links, implementation code, or training code references
3. For parent models: Look for mentions of base models, fine-tuned from, or model inheritance
4. For evaluation datasets: Look for evaluation benchmarks, test datasets, or validation data
5. Extract only EXPLICIT mentions - do not infer or guess
6. Use full URLs when available (e.g., https://huggingface.co/datasets/..., https://github.com/...)
7. For HuggingFace datasets without full URLs, include just the dataset name (e.g., "squad", "imagenet-1k")
8. Return empty lists for categories with no explicit mentions
9. DO NOT include any explanatory text, ONLY output valid JSON
10. DO NOT use markdown code blocks in your response

Your entire response must be a single, valid JSON object and nothing else."""

        # Try LLM extraction first
        if self.api_available:
            result = self._call_genai_api(prompt, readme_content)

            if result and isinstance(result, dict):
                # Validate and normalize the result
                return {
                    "datasets": result.get("training_datasets", []),
                    "code_repos": result.get("code_repositories", []),
                    "parent_models": result.get("parent_models", []),
                    "evaluation_datasets": result.get("evaluation_datasets", []),
                }

        # Fall back to regex-based extraction
        logger.info(f"Using fallback regex extraction for {model_name}")
        return self._fallback_extraction(readme_content)

    def _fallback_extraction(self, readme_content: str) -> Dict[str, List[str]]:
        """
        Fallback regex-based extraction when LLM is unavailable

        Args:
            readme_content: README text content

        Returns:
            Dictionary with extracted information
        """
        result = {
            "datasets": [],
            "code_repos": [],
            "parent_models": [],
            "evaluation_datasets": [],
        }

        if not readme_content:
            return result

        # Extract HuggingFace dataset URLs
        hf_dataset_pattern = (
            r"https?://huggingface\.co/datasets/([A-Za-z0-9_\-]+(?:/[A-Za-z0-9_\-]+)?)"
        )
        hf_datasets = re.findall(hf_dataset_pattern, readme_content)
        result["datasets"].extend(
            [f"https://huggingface.co/datasets/{ds}" for ds in hf_datasets]
        )

        # Extract GitHub repository URLs
        github_pattern = r"https?://github\.com/([A-Za-z0-9_\-]+/[A-Za-z0-9_\-\.]+)"
        github_repos = re.findall(github_pattern, readme_content)
        result["code_repos"].extend(
            [f"https://github.com/{repo}" for repo in set(github_repos)]
        )

        # Extract dataset mentions from YAML frontmatter
        yaml_dataset_pattern = r"datasets:\s*\n\s*-\s*([A-Za-z0-9_\-/]+)"
        yaml_datasets = re.findall(yaml_dataset_pattern, readme_content, re.MULTILINE)
        for ds in yaml_datasets:
            dataset_url = f"https://huggingface.co/datasets/{ds}"
            if dataset_url not in result["datasets"]:
                result["datasets"].append(dataset_url)

        # Extract parent model mentions
        parent_patterns = [
            r"fine[- ]?tuned (?:from|on) ([A-Za-z0-9_\-/]+)",
            r"based on ([A-Za-z0-9_\-/]+)",
            r"parent[_ ]model:\s*([A-Za-z0-9_\-/]+)",
            r"model[_ ]name[_ ]or[_ ]path:\s*([A-Za-z0-9_\-/]+)",
        ]
        for pattern in parent_patterns:
            matches = re.findall(pattern, readme_content, re.IGNORECASE)
            result["parent_models"].extend(matches)

        # Extract evaluation dataset mentions
        eval_patterns = [
            r"evaluated on ([A-Za-z0-9_\-/]+)",
            r"evaluation[_ ]dataset:\s*([A-Za-z0-9_\-/]+)",
            r"tested on ([A-Za-z0-9_\-/]+)",
        ]
        for pattern in eval_patterns:
            matches = re.findall(pattern, readme_content, re.IGNORECASE)
            result["evaluation_datasets"].extend(matches)

        # Deduplicate lists
        for key in result:
            result[key] = list(set(result[key]))

        return result

    def normalize_urls(
        self, lineage_data: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Normalize URLs in lineage data

        Args:
            lineage_data: Raw lineage data with potentially incomplete URLs

        Returns:
            Normalized lineage data with complete URLs
        """
        normalized = {
            "datasets": [],
            "code_repos": [],
            "parent_models": [],
            "evaluation_datasets": [],
        }

        for dataset in lineage_data.get("datasets", []):
            if dataset.startswith("http"):
                normalized["datasets"].append(dataset)
            else:
                # Assume HuggingFace dataset
                normalized["datasets"].append(
                    f"https://huggingface.co/datasets/{dataset}"
                )

        for repo in lineage_data.get("code_repos", []):
            if repo.startswith("http"):
                # Clean up GitHub URLs
                repo = repo.rstrip("/")
                repo = re.sub(r"\.git$", "", repo)
                normalized["code_repos"].append(repo)
            else:
                # Assume GitHub repo
                normalized["code_repos"].append(f"https://github.com/{repo}")

        # Parent models and eval datasets are typically just names
        normalized["parent_models"] = lineage_data.get("parent_models", [])
        normalized["evaluation_datasets"] = lineage_data.get("evaluation_datasets", [])

        return normalized

    def extract_from_model_metadata(
        self, readme_content: str, metadata_dict: Dict, model_name: str
    ) -> Tuple[str, str, List[str], List[str]]:
        """
        Extract lineage information and return in the format expected by ingest_handler

        Args:
            readme_content: README text content
            metadata_dict: Structured metadata from model card
            model_name: Name of the model

        Returns:
            Tuple of (code_url, dataset_url, parent_models, eval_datasets)
        """
        # Extract lineage using LLM or fallback
        lineage = self.extract_lineage(readme_content, model_name)
        lineage = self.normalize_urls(lineage)

        # Get primary code repository
        code_url = "unknown"
        if lineage["code_repos"]:
            code_url = lineage["code_repos"][0]  # Use first repo as primary

        # Get primary dataset
        dataset_url = "unknown"
        if lineage["datasets"]:
            dataset_url = lineage["datasets"][0]  # Use first dataset as primary

        # Also check metadata for dataset
        if dataset_url == "unknown" and "datasets" in metadata_dict:
            datasets_field = metadata_dict["datasets"]
            if isinstance(datasets_field, list) and datasets_field:
                dataset_name = str(datasets_field[0])
                dataset_url = f"https://huggingface.co/datasets/{dataset_name}"
            elif isinstance(datasets_field, str):
                dataset_url = f"https://huggingface.co/datasets/{datasets_field}"

        return (
            code_url,
            dataset_url,
            lineage["parent_models"],
            lineage["evaluation_datasets"],
        )


# Global instance
_extractor = None


def get_lineage_extractor() -> LineageExtractor:
    """Get or create the global lineage extractor instance"""
    global _extractor
    if _extractor is None:
        _extractor = LineageExtractor()
    return _extractor
