import os
import time
from typing import Any, Dict, List, Tuple

import requests

from url_classifier import URLType

from .base_metric import BaseMetric


class DatasetQualityMetric(BaseMetric):
    """Metric for dataset quality assessment using Purdue GenAI Studio (LLM) with fallback"""

    def required_url_types(self) -> List[URLType]:
        return [URLType.DATASET]

    def calculate(self, resources: Dict[URLType, List[Any]]) -> Tuple[float, int]:
        start_time = time.time()
        self.logger.info("Starting dataset quality metric calculation")

        if not resources.get(URLType.DATASET):
            self.logger.warning("No dataset resources available")
            return 0.0, int((time.time() - start_time) * 1000)

        dataset = resources[URLType.DATASET][0]

        # Try LLM-based evaluation first
        api_key = os.getenv("GEN_AI_STUDIO_API_KEY")
        if api_key:
            quality_score = self._evaluate_with_llm(dataset, api_key)
            if quality_score is not None:
                end_time = time.time()
                latency_ms = int((end_time - start_time) * 1000)
                self.logger.info(f"LLM-based dataset quality score={quality_score:.2f}, latency={latency_ms}ms")
                return quality_score, latency_ms

        # Fallback to heuristic evaluation
        self.logger.info("Falling back to heuristic dataset quality evaluation")
        quality_score = self._evaluate_dataset_quality(dataset)

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        self.logger.info(f"Heuristic dataset quality score={quality_score:.2f}, latency={latency_ms}ms")

        return quality_score, latency_ms

    def _evaluate_with_llm(self, dataset: Any, api_key: str) -> float:
        """Evaluate dataset quality using Purdue GenAI Studio LLM"""
        try:
            self.logger.info("Calling GenAI Studio API for dataset quality evaluation")

            # Get dataset information
            dataset_url = dataset.url if hasattr(dataset, "url") else "N/A"
            api_data = {}
            if hasattr(dataset, "get_huggingface_api_data"):
                api_data = dataset.get_huggingface_api_data()

            description = api_data.get("description", "N/A")
            tags = api_data.get("tags", [])
            downloads = api_data.get("downloads", 0)

            prompt = f"""You are a Software Engineer evaluating dataset resources.

Dataset URL: {dataset_url}
Description: {description}
Tags: {', '.join(tags) if tags else 'N/A'}
Downloads: {downloads}

Rate the dataset quality from 0.0 to 1.0 based on:
- Dataset clarity and documentation quality
- Completeness of metadata and descriptions
- Usefulness for developers and researchers
- Community engagement (downloads, tags)

Respond with only a number between 0.0 and 1.0."""

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": "llama4:latest",
                "messages": [{"role": "user", "content": prompt}],
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
                score = float(content)
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                self.logger.info(f"LLM returned dataset quality score: {score:.2f}")
                return score
            else:
                self.logger.warning(f"GenAI API returned status {response.status_code}")
                return 0.0

        except Exception as e:
            self.logger.error(f"Error during GenAI API call for dataset quality: {e}", exc_info=True)
            return 0.0

    def _evaluate_dataset_quality(self, dataset: Any) -> float:
        """Heuristic fallback for dataset quality evaluation"""
        try:
            # Use the handler's built-in quality score method if available
            if hasattr(dataset, "get_quality_score"):
                return dataset.get_quality_score()

            # Fallback: manual calculation if get_quality_score not available
            api_data = {}
            if hasattr(dataset, "get_huggingface_api_data"):
                api_data = dataset.get_huggingface_api_data() or {}

            score = 0.0

            # Check for documentation
            if api_data.get("cardData"):
                score += 0.3
                self.logger.debug("Dataset has card data: +0.3")

            # Check for downloads (popularity proxy for quality)
            downloads = api_data.get("downloads", 0)
            if downloads > 1000:
                score += 0.3
                self.logger.debug(f"Dataset has {downloads} downloads (>1000): +0.3")
            elif downloads > 100:
                score += 0.2
                self.logger.debug(f"Dataset has {downloads} downloads (>100): +0.2")
            elif downloads > 10:
                score += 0.1
                self.logger.debug(f"Dataset has {downloads} downloads (>10): +0.1")

            # Check for tags (well-categorized datasets)
            tags = api_data.get("tags", [])
            if len(tags) > 2:
                score += 0.2
                self.logger.debug(f"Dataset has {len(tags)} tags: +0.2")

            # Check for multiple configs (versatility)
            siblings = api_data.get("siblings", [])
            if len(siblings) > 1:
                score += 0.2
                self.logger.debug(f"Dataset has {len(siblings)} files: +0.2")

            return min(score, 1.0)

        except Exception as e:
            self.logger.error(f"Error in heuristic dataset quality evaluation: {e}")
            return 0.0
