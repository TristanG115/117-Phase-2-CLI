from typing import Dict, Any
import requests
from urllib.parse import urlparse

from .base_resource_handler import BaseResourceHandler


class DatasetHandler(BaseResourceHandler):
    """Handler for Hugging Face dataset resources"""

    def __init__(self, url: str):
        super().__init__(url)
        self.dataset_id = self._extract_dataset_id()

    def _extract_dataset_id(self) -> str:
        """Extract dataset ID from Hugging Face URL"""
        parsed = urlparse(self.url)
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) >= 2 and path_parts[0] == 'datasets':
            return f"{path_parts[1]}/{path_parts[2]}" if len(path_parts) > 2 else path_parts[1]
        return ""

    def get_huggingface_api_data(self) -> Dict[str, Any]:
        """Get data from Hugging Face API"""
        cached = self._cache_get('hf_api_data')
        if cached:
            return cached

        try:
            api_url = f"https://huggingface.co/api/datasets/{self.dataset_id}"
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self._cache_set('hf_api_data', data)
                return data
        except Exception as e:
            self.logger.error(f"Error fetching dataset API data: {e}")

        return {}

    def has_evaluation_dataset(self) -> bool:
        """Check if dataset is suitable for evaluation"""
        api_data = self.get_huggingface_api_data()
        tags = api_data.get('tags', [])
        return 'evaluation' in tags or 'benchmark' in tags

    def get_quality_score(self) -> float:
        """Evaluate dataset quality"""
        api_data = self.get_huggingface_api_data()

        score = 0.0

        # Check for documentation
        if api_data.get('cardData'):
            score += 0.3

        # Check for downloads (popularity proxy for quality)
        downloads = api_data.get('downloads', 0)
        if downloads > 1000:
            score += 0.3
        elif downloads > 100:
            score += 0.2
        elif downloads > 10:
            score += 0.1

        # Check for tags (well-categorized datasets)
        tags = api_data.get('tags', [])
        if len(tags) > 2:
            score += 0.2

        # Check for multiple configs (versatility)
        if len(api_data.get('siblings', [])) > 1:
            score += 0.2

        return min(score, 1.0)

    def get_license_score(self) -> float:
        """License evaluation not applicable for datasets"""
        return 0.0

    def get_documentation_score(self) -> float:
        """Evaluate documentation quality"""
        api_data = self.get_huggingface_api_data()
        card_data = api_data.get('cardData', {})

        score = 0.0
        if card_data:
            score += 0.5
        if api_data.get('description'):
            score += 0.3
        if len(api_data.get('tags', [])) > 0:
            score += 0.2

        return min(score, 1.0)

    def get_contributor_count(self) -> int:
        """Get number of contributors"""
        api_data = self.get_huggingface_api_data()

        # Use downloads as proxy
        downloads = api_data.get('downloads', 0)
        if downloads > 10000:
            return 5
        elif downloads > 1000:
            return 3
        else:
            return 1