from typing import Dict, Any, List
import requests
import tempfile
import os
from urllib.parse import urlparse

from .base_resource_handler import BaseResourceHandler


class ModelHandler(BaseResourceHandler):
    """Handler for Hugging Face model resources"""

    def __init__(self, url: str):
        super().__init__(url)
        self.model_id = self._extract_model_id()

    def _extract_model_id(self) -> str:
        """Extract model ID from Hugging Face URL"""
        parsed = urlparse(self.url)
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) >= 2:
            return f"{path_parts[0]}/{path_parts[1]}"
        return ""

    def get_huggingface_api_data(self) -> Dict[str, Any]:
        """Get data from Hugging Face API"""
        cached = self._cache_get('hf_api_data')
        if cached:
            return cached

        try:
            api_url = f"https://huggingface.co/api/models/{self.model_id}"
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self._cache_set('hf_api_data', data)
                return data
        except Exception as e:
            self.logger.error(f"Error fetching HF API data: {e}")

        return {}

    def get_model_files(self) -> List[Dict[str, Any]]:
        """Get model files from repository"""
        try:
            files_url = f"https://huggingface.co/api/models/{self.model_id}/tree/main"
            response = requests.get(files_url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            self.logger.error(f"Error fetching model files: {e}")

        return []

    def get_size_mb(self) -> float:
        """Calculate total model size in MB"""
        cached = self._cache_get('size_mb')
        if cached is not None:
            return cached

        total_size = 0
        files = self.get_model_files()

        for file_info in files:
            if isinstance(file_info, dict) and 'size' in file_info:
                total_size += file_info['size']

        size_mb = total_size / (1024 * 1024)  # Convert to MB
        self._cache_set('size_mb', size_mb)
        return size_mb

    def has_performance_benchmarks(self) -> bool:
        """Check if model has performance benchmarks"""
        try:
            # Check README for benchmark information
            readme_url = f"https://huggingface.co/{self.model_id}/raw/main/README.md"
            response = requests.get(readme_url, timeout=10)
            if response.status_code == 200:
                readme_content = response.text.lower()
                benchmark_keywords = ['benchmark', 'evaluation', 'performance', 'score', 'metric']
                return any(keyword in readme_content for keyword in benchmark_keywords)
        except Exception as e:
            self.logger.error(f"Error checking benchmarks: {e}")

        return False

    def get_license_score(self) -> float:
        """Get license compatibility score by downloading README.md and checking metadata"""
        cached = self._cache_get('license_score')
        if cached is not None:
            return cached

        score = self._get_license_from_local_readme()
        self._cache_set('license_score', score)
        return score

    def _get_license_from_local_readme(self) -> float:
        """Download README.md locally, check metadata, then delete file"""
        temp_file = None
        try:
            # Use HuggingFace's raw file HTTP API
            readme_url = f"https://huggingface.co/{self.model_id}/raw/main/README.md"

            # Get HF_API_TOKEN from environment if available
            hf_token = os.environ.get('HF_API_TOKEN')
            headers = {}
            if hf_token:
                headers['Authorization'] = f'Bearer {hf_token}'

            response = requests.get(readme_url, headers=headers, timeout=10)

            if response.status_code == 200:
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.md', delete=False)
                temp_file.write(response.text)
                temp_file.close()

                # Parse metadata from the downloaded file
                with open(temp_file.name, 'r', encoding='utf-8') as f:
                    readme_content = f.read()

                # Check for YAML frontmatter first
                license_score, found_in_metadata = self._parse_license_from_metadata(readme_content)

                # Only fallback to text parsing if no YAML frontmatter exists at all
                if not found_in_metadata and not readme_content.strip().startswith('---'):
                    license_score = self._parse_license_from_text(readme_content)
                # If YAML frontmatter exists but no license field, default to 0.0
                elif not found_in_metadata:
                    license_score = 0.0

                return license_score
            else:
                self.logger.warning(f"Could not fetch README.md from {readme_url}: HTTP {response.status_code}")
                return 0.0

        except Exception as e:
            self.logger.error(f"Error downloading README.md: {e}")
            return 0.0
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    self.logger.warning(f"Failed to delete temp README file: {e}")

    def _parse_license_from_metadata(self, content: str) -> tuple[float, bool]:
        """Parse license from YAML frontmatter metadata, returns (score, found)"""
        try:
            # Check if content starts with YAML frontmatter
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 2:
                    yaml_content = parts[1].strip()

                    # Simple parsing for license field (avoiding yaml dependency)
                    for line in yaml_content.split('\n'):
                        line = line.strip()
                        if line.lower().startswith('license:'):
                            license_value = line.split(':', 1)[1].strip().strip('"\'')
                            score = self._parse_license_identifier(license_value)
                            return score, True  # Found license in metadata

            return 0.0, False  # No license found in metadata
        except Exception as e:
            self.logger.error(f"Error parsing metadata: {e}")
            return 0.0, False

    def get_documentation_score(self) -> float:
        """Evaluate documentation quality"""
        try:
            readme_url = f"https://huggingface.co/{self.model_id}/raw/main/README.md"
            response = requests.get(readme_url, timeout=10)
            if response.status_code == 200:
                readme_content = response.text

                # Simple scoring based on README length and sections
                score = 0.0
                if len(readme_content) > 500:
                    score += 0.3
                if 'usage' in readme_content.lower():
                    score += 0.3
                if 'example' in readme_content.lower():
                    score += 0.2
                if 'training' in readme_content.lower():
                    score += 0.2

                return min(score, 1.0)
        except Exception as e:
            self.logger.error(f"Error evaluating documentation: {e}")

        return 0.0

    def get_contributor_count(self) -> int:
        """Get number of contributors (approximation using downloads/likes)"""
        api_data = self.get_huggingface_api_data()

        # Use downloads and likes as proxy for community size
        downloads = api_data.get('downloads', 0)
        likes = api_data.get('likes', 0)

        # Rough approximation: more popular models likely have more contributors
        if downloads > 10000 or likes > 100:
            return 10
        elif downloads > 1000 or likes > 20:
            return 5
        elif downloads > 100 or likes > 5:
            return 2
        else:
            return 1