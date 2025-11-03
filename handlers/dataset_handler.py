# NEW CODE
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests

from .base_resource_handler import BaseResourceHandler


class DatasetHandler(BaseResourceHandler):
    """Enhanced handler for Hugging Face dataset resources"""

    def __init__(self, url: str):
        super().__init__(url)
        self.dataset_id = self._extract_dataset_id()
        self._readme_content = None

    def _extract_dataset_id(self) -> str:
        """Extract dataset ID from Hugging Face URL"""
        parsed = urlparse(self.url)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 3 and path_parts[0] == "datasets":
            return f"{path_parts[1]}/{path_parts[2]}"
        elif len(path_parts) >= 2 and path_parts[0] == "datasets":
            return path_parts[1]
        return ""

    def get_huggingface_api_data(self) -> Dict[str, Any]:
        """Get comprehensive data from Hugging Face API"""
        cached = self._cache_get("hf_api_data")
        if cached:
            return cached

        try:
            api_url = f"https://huggingface.co/api/datasets/{self.dataset_id}"
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self._cache_set("hf_api_data", data)
                self.logger.info(
                    f"Fetched HuggingFace dataset API data for {self.dataset_id}: downloads={data.get('downloads', 0)}"
                )
                return data
            else:
                self.logger.warning(f"HuggingFace dataset API returned {response.status_code} for {self.dataset_id}")
        except Exception as e:
            self.logger.error(f"Error fetching dataset API data: {e}")

        return {}

    def get_readme_content(self) -> str:
        """Get README content (cached)"""
        if self._readme_content is not None:
            return self._readme_content

        try:
            readme_url = f"https://huggingface.co/datasets/{self.dataset_id}/raw/main/README.md"
            response = requests.get(readme_url, timeout=10)
            if response.status_code == 200:
                self._readme_content = response.text
                self.logger.debug(f"Fetched README for dataset {self.dataset_id}, length={len(self._readme_content)}")
                return self._readme_content
        except Exception as e:
            self.logger.error(f"Error fetching dataset README: {e}")

        self._readme_content = ""
        return ""

    def has_evaluation_dataset(self) -> bool:
        """Check if dataset is suitable for evaluation"""
        api_data = self.get_huggingface_api_data()
        tags = api_data.get("tags", [])

        eval_indicators = ["evaluation", "benchmark", "test", "eval"]
        is_eval = any(indicator in str(tag).lower() for tag in tags for indicator in eval_indicators)

        if is_eval:
            self.logger.info(f"Dataset {self.dataset_id} identified as evaluation dataset")

        return is_eval

    def get_quality_score(self) -> float:
        """
        Evaluate dataset quality comprehensively.
        This now includes README strength as a first-class signal (objective, no special-casing),
        which better distinguishes high-quality, well-documented datasets.
        """
        cached = self._cache_get("quality_score")
        if cached is not None:
            return cached

        api_data = self.get_huggingface_api_data()
        score = 0.0

        # Documentation quality (card—structured metadata)
        card_data = api_data.get("cardData", {})
        if card_data:
            score += 0.25
            # Structured dataset info indicates stronger metadata discipline
            if card_data.get("dataset_info"):
                score += 0.1

        # Description quality
        description = api_data.get("description", "")
        if len(description) > 200:
            score += 0.15
        elif len(description) > 100:
            score += 0.1
        elif len(description) > 50:
            score += 0.05

        # README strength
        readme = self.get_readme_content()
        if len(readme) > 1000:
            score += 0.10
        elif len(readme) > 500:
            score += 0.05

        # Downloads (popularity = quality proxy)
        downloads = api_data.get("downloads", 0)
        if downloads > 10000:
            score += 0.2
        elif downloads > 1000:
            score += 0.15
        elif downloads > 100:
            score += 0.1
        elif downloads > 10:
            score += 0.05

        # Tags (well-categorized)
        tags = api_data.get("tags", [])
        if len(tags) > 5:
            score += 0.15
        elif len(tags) > 2:
            score += 0.1
        elif len(tags) > 0:
            score += 0.05

        # Files/siblings (multiple configs = versatile)
        siblings = api_data.get("siblings", [])
        if len(siblings) > 5:
            score += 0.15
        elif len(siblings) > 1:
            score += 0.1

        final_score = min(score, 1.0)
        self._cache_set("quality_score", final_score)
        self.logger.info(f"Dataset quality score for {self.dataset_id}: {final_score:.2f}")
        return final_score

    def get_license_score(self) -> float:
        """Get license compatibility score from API metadata"""
        api_data = self.get_huggingface_api_data()

        # Check license field
        license_value = api_data.get("license")
        if license_value:
            score = self._parse_license_identifier(license_value)
            if score > 0:
                self.logger.info(f"Dataset license for {self.dataset_id}: {license_value} (score={score})")
                return score

        # Check cardData
        card_data = api_data.get("cardData", {})
        if card_data.get("license"):
            score = self._parse_license_identifier(card_data["license"])
            if score > 0:
                self.logger.info(
                    f"Dataset license in cardData for {self.dataset_id}: {card_data['license']} (score={score})"
                )
                return score

        # Check tags
        tags = api_data.get("tags", [])
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("license:"):
                license_name = tag.replace("license:", "").strip()
                score = self._parse_license_identifier(license_name)
                if score > 0:
                    self.logger.info(f"Dataset license in tags for {self.dataset_id}: {license_name} (score={score})")
                    return score

        self.logger.warning(f"No license found for dataset {self.dataset_id}")
        return 0.0

    def get_documentation_score(self) -> float:
        """Evaluate documentation quality comprehensively"""
        cached = self._cache_get("doc_score")
        if cached is not None:
            return cached

        api_data = self.get_huggingface_api_data()
        readme = self.get_readme_content()

        score = 0.0

        # README content
        if len(readme) > 1000:
            score += 0.3
        elif len(readme) > 500:
            score += 0.2
        elif len(readme) > 100:
            score += 0.1

        # Card data
        if api_data.get("cardData"):
            score += 0.2

        # Description
        if api_data.get("description"):
            desc_len = len(api_data["description"])
            if desc_len > 200:
                score += 0.2
            elif desc_len > 100:
                score += 0.15
            elif desc_len > 50:
                score += 0.1

        # Tags indicate categorization
        tags = api_data.get("tags", [])
        if len(tags) > 3:
            score += 0.15
        elif len(tags) > 0:
            score += 0.05

        # Check for structured dataset info
        card_data = api_data.get("cardData", {})
        if card_data.get("dataset_info"):
            score += 0.15

        final_score = min(score, 1.0)
        self._cache_set("doc_score", final_score)
        self.logger.info(f"Dataset documentation score for {self.dataset_id}: {final_score:.2f}")
        return final_score

    def get_contributor_count(self) -> int:
        """Get number of contributors (approximation using downloads)"""
        api_data = self.get_huggingface_api_data()
        downloads = api_data.get("downloads", 0)

        # Improved heuristic for datasets
        if downloads > 50000:
            return 10
        elif downloads > 10000:
            return 5
        elif downloads > 1000:
            return 3
        elif downloads > 100:
            return 2
        else:
            return 1

    def get_tags(self) -> List[str]:
        """Get dataset tags"""
        api_data = self.get_huggingface_api_data()
        return api_data.get("tags", [])

    def get_downloads(self) -> int:
        """Get download count"""
        api_data = self.get_huggingface_api_data()
        return api_data.get("downloads", 0)

    def get_description(self) -> str:
        """Get dataset description"""
        api_data = self.get_huggingface_api_data()
        return api_data.get("description", "")

    def get_siblings(self) -> List[Dict[str, Any]]:
        """Get dataset files/siblings"""
        api_data = self.get_huggingface_api_data()
        return api_data.get("siblings", [])
