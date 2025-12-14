from typing import Any, Dict, List
from urllib.parse import urlparse

import requests

from .base_resource_handler import BaseResourceHandler


class DatasetHandler(BaseResourceHandler):
    """Enhanced handler for Hugging Face dataset resources"""

    def __init__(self, url: str):
        super().__init__(url)
        self.dataset_id = self._extract_dataset_id()
        self._readme_content = ""

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

    # Quality Score
    def get_quality_score(self) -> float:
        """Evaluate dataset quality comprehensively"""
        cached = self._cache_get("quality_score")
        if cached is not None:
            return cached

        api_data = self.get_huggingface_api_data()
        score = (
            self._quality_doc_score(api_data)
            + self._quality_description_score(api_data)
            + self._quality_readme_score()
            + self._quality_downloads_score(api_data)
            + self._quality_tags_score(api_data)
            + self._quality_siblings_score(api_data)
        )

        final_score = min(score, 1.0)
        self._cache_set("quality_score", final_score)
        self.logger.info(f"Dataset quality score for {self.dataset_id}: {final_score:.2f}")
        return final_score

    def _quality_doc_score(self, api_data: dict) -> float:
        card_data = api_data.get("cardData", {})
        score = 0.0
        if card_data:
            score += 0.25
            if card_data.get("dataset_info"):
                score += 0.1
        return score

    def _quality_description_score(self, api_data: dict) -> float:
        description = api_data.get("description", "")
        if len(description) > 200:
            return 0.15
        if len(description) > 100:
            return 0.1
        if len(description) > 50:
            return 0.05
        return 0.0

    def _quality_readme_score(self) -> float:
        readme = self.get_readme_content()
        if len(readme) > 1000:
            return 0.10
        if len(readme) > 500:
            return 0.05
        return 0.0

    def _quality_downloads_score(self, api_data: dict) -> float:
        downloads = api_data.get("downloads", 0)
        if downloads > 10000:
            return 0.2
        if downloads > 1000:
            return 0.15
        if downloads > 100:
            return 0.1
        if downloads > 10:
            return 0.05
        return 0.0

    def _quality_tags_score(self, api_data: dict) -> float:
        tags = api_data.get("tags", [])
        if len(tags) > 5:
            return 0.15
        if len(tags) > 1:
            return 0.1
        return 0.0

    def _quality_siblings_score(self, api_data: dict) -> float:
        siblings = api_data.get("siblings", [])
        if len(siblings) > 5:
            return 0.15
        if len(siblings) > 1:
            return 0.1
        return 0.0

    # Documentation Score
    def get_documentation_score(self) -> float:
        """Evaluate documentation quality comprehensively"""
        cached = self._cache_get("doc_score")
        if cached is not None:
            return cached

        api_data = self.get_huggingface_api_data()
        readme = self.get_readme_content()

        score = (
            self._doc_readme_score(readme)
            + self._doc_card_data_score(api_data)
            + self._doc_description_score(api_data)
            + self._doc_tags_score(api_data)
            + self._doc_structured_info_score(api_data)
        )

        final_score = min(score, 1.0)
        self._cache_set("doc_score", final_score)
        self.logger.info(f"Dataset documentation score for {self.dataset_id}: {final_score:.2f}")
        return final_score

    def _doc_readme_score(self, readme: str) -> float:
        if len(readme) > 1000:
            return 0.3
        if len(readme) > 500:
            return 0.2
        if len(readme) > 100:
            return 0.1
        return 0.0

    def _doc_card_data_score(self, api_data: dict) -> float:
        return 0.2 if api_data.get("cardData") else 0.0

    def _doc_description_score(self, api_data: dict) -> float:
        description = api_data.get("description", "")
        desc_len = len(description)
        if desc_len > 200:
            return 0.2
        if desc_len > 100:
            return 0.15
        if desc_len > 50:
            return 0.1
        return 0.0

    def _doc_tags_score(self, api_data: dict) -> float:
        tags = api_data.get("tags", [])
        if len(tags) > 3:
            return 0.15
        if len(tags) > 0:
            return 0.05
        return 0.0

    def _doc_structured_info_score(self, api_data: dict) -> float:
        card_data = api_data.get("cardData", {})
        return 0.15 if card_data.get("dataset_info") else 0.0

    # License
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

    # Contributors
    def get_contributor_count(self) -> int:
        """Get number of contributors (approximation using downloads)"""
        api_data = self.get_huggingface_api_data()
        downloads = api_data.get("downloads", 0)
        if downloads > 50000:
            return 10
        if downloads > 10000:
            return 5
        if downloads > 1000:
            return 3
        if downloads > 100:
            return 2
        return 1

    #  Helpers
    def get_tags(self) -> List[str]:
        api_data = self.get_huggingface_api_data()
        return api_data.get("tags", [])

    def get_downloads(self) -> int:
        api_data = self.get_huggingface_api_data()
        return api_data.get("downloads", 0)

    def get_description(self) -> str:
        api_data = self.get_huggingface_api_data()
        return api_data.get("description", "")

    def get_siblings(self) -> List[Dict[str, Any]]:
        api_data = self.get_huggingface_api_data()
        return api_data.get("siblings", [])

    def get_hf_dataset_info(self, dataset_url: str):
        """
        Minimal stub method used ONLY for tests.
        """
        return {"url": dataset_url, "status": "ok"}
