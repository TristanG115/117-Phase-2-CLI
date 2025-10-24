# NEW CODE
import re
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests

from .base_resource_handler import BaseResourceHandler


class ModelHandler(BaseResourceHandler):
    """Enhanced handler for Hugging Face model resources with comprehensive data extraction"""

    def __init__(self, url: str):
        super().__init__(url)
        self.model_id = self._extract_model_id()
        self._readme_content = None
        self._api_data_fetched = False

    def _extract_model_id(self) -> str:
        """Extract model ID from Hugging Face URL"""
        parsed = urlparse(self.url)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 2:
            return f"{path_parts[0]}/{path_parts[1]}"
        return ""

    def get_huggingface_api_data(self) -> Dict[str, Any]:
        """Get comprehensive data from Hugging Face API"""
        cached = self._cache_get("hf_api_data")
        if cached:
            return cached

        try:
            api_url = f"https://huggingface.co/api/models/{self.model_id}"
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self._cache_set("hf_api_data", data)
                self._api_data_fetched = True
                self.logger.info(
                    f"Fetched HuggingFace API data for {self.model_id}: "
                    f"downloads={data.get('downloads', 0)}, likes={data.get('likes', 0)}"
                )
                return data
            else:
                self.logger.warning(
                    f"HuggingFace API returned {response.status_code} for {self.model_id}"
                )
        except Exception as e:
            self.logger.error(f"Error fetching HF API data: {e}")

        return {}

    def get_readme_content(self) -> str:
        """Get README content (cached)"""
        if self._readme_content is not None:
            return self._readme_content

        try:
            readme_url = f"https://huggingface.co/{self.model_id}/raw/main/README.md"
            response = requests.get(readme_url, timeout=10)
            if response.status_code == 200:
                self._readme_content = response.text
                self.logger.debug(
                    f"Fetched README for {self.model_id}, length={len(self._readme_content)}"
                )
                return self._readme_content
        except Exception as e:
            self.logger.error(f"Error fetching README: {e}")

        self._readme_content = ""
        return ""

    def extract_github_urls_from_readme(self) -> List[str]:
        """Extract GitHub repository URLs from README"""
        readme = self.get_readme_content()
        if not readme:
            return []

        # Pattern to match GitHub URLs
        github_pattern = r"https?://github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)"
        matches = re.findall(github_pattern, readme, re.IGNORECASE)

        # Clean and deduplicate
        github_urls = []
        seen = set()
        for match in matches:
            url = f"https://github.com/{match}"
            # Filter out blob/tree/issues URLs
            if (
                "/blob/" not in url
                and "/tree/" not in url
                and "/issues" not in url
                and url not in seen
            ):
                github_urls.append(url)
                seen.add(url)

        if github_urls:
            self.logger.info(
                f"Found {len(github_urls)} GitHub URLs in README for {self.model_id}"
            )

        return github_urls

    def extract_dataset_urls_from_readme(self) -> List[str]:
        """Extract HuggingFace dataset URLs from README"""
        readme = self.get_readme_content()
        if not readme:
            return []

        # Pattern to match HuggingFace dataset URLs
        dataset_pattern = (
            r"https?://huggingface\.co/datasets/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)"
        )
        matches = re.findall(dataset_pattern, readme, re.IGNORECASE)

        dataset_urls = []
        seen = set()
        for match in matches:
            url = f"https://huggingface.co/datasets/{match}"
            if url not in seen:
                dataset_urls.append(url)
                seen.add(url)

        if dataset_urls:
            self.logger.info(
                f"Found {len(dataset_urls)} dataset URLs in README for {self.model_id}"
            )

        return dataset_urls

    def get_model_files(self) -> List[Dict[str, Any]]:
        """Get model files from repository"""
        cached = self._cache_get("model_files")
        if cached:
            return cached

        try:
            files_url = f"https://huggingface.co/api/models/{self.model_id}/tree/main"
            response = requests.get(files_url, timeout=10)
            if response.status_code == 200:
                files = response.json()
                self._cache_set("model_files", files)
                self.logger.debug(f"Fetched {len(files)} files for {self.model_id}")
                return files
        except Exception as e:
            self.logger.error(f"Error fetching model files: {e}")

        return []

    def _resolve_file_size_via_head(self, filename: str) -> int:
        """
        Resolve a single file size using a HEAD request on the /resolve endpoint.
        Returns size in bytes or 0 on failure.
        """
        try:
            # Try `main` first, then fall back to `master`
            for branch in ("main", "master"):
                url = f"https://huggingface.co/{self.model_id}/resolve/{branch}/{filename}"
                r = requests.head(url, timeout=10, allow_redirects=True)
                if r.status_code == 200 and "Content-Length" in r.headers:
                    sz = int(r.headers["Content-Length"])
                    if sz > 0:
                        return sz
        except Exception as e:
            self.logger.debug(f"HEAD size probe failed for {filename}: {e}")
        return 0

    def get_size_mb(self) -> float:
        """
        Calculate total model size in MB with robust fallbacks.
        This improves accuracy for models that store large LFS blobs (common for BERT),
        which can be under-counted by naive tree listings.
        """
        cached = self._cache_get("size_mb")
        if cached is not None:
            return cached

        # 1) Try to use safetensors total reported by API
        api_data = self.get_huggingface_api_data()
        if api_data.get("safetensors"):
            total_size = api_data["safetensors"].get("total", 0)
            if total_size > 0:
                size_mb = total_size / (1024 * 1024)
                self._cache_set("size_mb", size_mb)
                self.logger.info(f"Model size from safetensors: {size_mb:.2f} MB")
                return size_mb

        # 2) Sum sizes from the API 'tree' if present
        total_size = 0
        files = self.get_model_files()
        for file_info in files:
            if isinstance(file_info, dict) and "size" in file_info:
                try:
                    total_size += int(file_info["size"])
                except Exception:
                    pass

        # 3) If still suspiciously low, try the 'siblings' list from API (if present)
        if total_size == 0 and isinstance(api_data, dict):
            siblings = api_data.get("siblings", [])
            for sib in siblings:
                if isinstance(sib, dict) and "rsize" in sib:
                    try:
                        total_size += int(sib["rsize"])
                    except Exception:
                        # some entries label as 'size' instead of 'rsize'
                        sz = sib.get("size", 0)
                        try:
                            total_size += int(sz)
                        except Exception:
                            pass

        # 4) If still low, probe common weight files via HEAD on /resolve
        if total_size == 0:
            common_weight_names = [
                # pytorch
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
                # safetensors
                "model.safetensors",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                # TF / ONNX (less common for BERT base)
                "tf_model.h5",
                "model.onnx",
            ]
            probed = 0
            for fname in common_weight_names:
                sz = self._resolve_file_size_via_head(fname)
                if sz > 0:
                    total_size += sz
                    probed += 1
            if probed:
                self.logger.info(
                    f"Resolved {probed} weight file sizes via HEAD for {self.model_id}"
                )

        size_mb = total_size / (1024 * 1024) if total_size > 0 else 0.0
        self._cache_set("size_mb", size_mb)
        self.logger.info(
            f"Model size calculated: {size_mb:.2f} MB (aggregated from API/tree/HEAD)"
        )
        return size_mb

    def has_performance_benchmarks(self) -> bool:
        """Check if model has performance benchmarks in README or metadata"""
        cached = self._cache_get("has_benchmarks")
        if cached is not None:
            return cached

        # Check API metadata first
        api_data = self.get_huggingface_api_data()

        # Check for model-index (structured benchmark data)
        card_data = api_data.get("cardData", {})
        if card_data.get("model-index"):
            self.logger.info(f"Found model-index benchmarks for {self.model_id}")
            self._cache_set("has_benchmarks", True)
            return True

        # Check README content
        readme = self.get_readme_content()
        if readme:
            readme_lower = readme.lower()
            benchmark_keywords = [
                "benchmark",
                "evaluation",
                "performance",
                "score",
                "accuracy",
                "f1",
                "metric",
                "results",
            ]
            has_benchmark = any(
                keyword in readme_lower for keyword in benchmark_keywords
            )
            self._cache_set("has_benchmarks", has_benchmark)
            if has_benchmark:
                self.logger.info(
                    f"Found benchmark keywords in README for {self.model_id}"
                )
            return has_benchmark

        self._cache_set("has_benchmarks", False)
        return False

    def get_license_score(self) -> float:
        """Get license compatibility score from API metadata and README"""
        cached = self._cache_get("license_score")
        if cached is not None:
            return cached

        score = 0.0

        # First try: Check API metadata
        api_data = self.get_huggingface_api_data()

        # Check license field
        license_value = api_data.get("license")
        if license_value:
            score = self._parse_license_identifier(license_value)
            if score > 0:
                self.logger.info(
                    f"License found in API metadata for {self.model_id}: {license_value} (score={score})"
                )
                self._cache_set("license_score", score)
                return score

        # Check cardData
        card_data = api_data.get("cardData", {})
        if card_data.get("license"):
            score = self._parse_license_identifier(card_data["license"])
            if score > 0:
                self.logger.info(
                    f"License found in cardData for {self.model_id}: {card_data['license']} (score={score})"
                )
                self._cache_set("license_score", score)
                return score

        # Check tags for license
        tags = api_data.get("tags", [])
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("license:"):
                license_name = tag.replace("license:", "").strip()
                score = self._parse_license_identifier(license_name)
                if score > 0:
                    self.logger.info(
                        f"License found in tags for {self.model_id}: {license_name} (score={score})"
                    )
                    self._cache_set("license_score", score)
                    return score

        # Second try: Check README
        readme = self.get_readme_content()
        if readme:
            # Check for YAML frontmatter
            if readme.startswith("---"):
                parts = readme.split("---", 2)
                if len(parts) >= 2:
                    yaml_content = parts[1].strip()
                    for line in yaml_content.split("\n"):
                        line = line.strip()
                        if line.lower().startswith("license:"):
                            license_value = line.split(":", 1)[1].strip().strip("\"'")
                            score = self._parse_license_identifier(license_value)
                            if score > 0:
                                self.logger.info(
                                    f"License found in README YAML for {self.model_id}: {license_value} (score={score})"
                                )
                                self._cache_set("license_score", score)
                                return score

            # Fallback: Search README text
            score = self._parse_license_from_text(readme)
            if score > 0:
                self.logger.info(
                    f"License found in README text for {self.model_id} (score={score})"
                )

        self._cache_set("license_score", score)
        if score == 0:
            self.logger.warning(f"No license found for {self.model_id}")
        return score

    def get_documentation_score(self) -> float:
        """Evaluate documentation quality comprehensively"""
        cached = self._cache_get("doc_score")
        if cached is not None:
            return cached

        readme = self.get_readme_content()
        api_data = self.get_huggingface_api_data()

        score = 0.0

        # README length and content
        if len(readme) > 1000:
            score += 0.25
        elif len(readme) > 500:
            score += 0.15
        elif len(readme) > 100:
            score += 0.05

        # Check for key sections
        readme_lower = readme.lower()
        if "usage" in readme_lower or "how to use" in readme_lower:
            score += 0.2
        if "example" in readme_lower or "code example" in readme_lower:
            score += 0.2
        if (
            "training" in readme_lower
            or "fine-tuning" in readme_lower
            or "fine-tune" in readme_lower
        ):
            score += 0.15
        if "installation" in readme_lower or "requirements" in readme_lower:
            score += 0.1
        if "citation" in readme_lower or "bibtex" in readme_lower:
            score += 0.1

        # Check API metadata
        if api_data.get("cardData"):
            score += 0.1

        final_score = min(score, 1.0)
        self._cache_set("doc_score", final_score)
        self.logger.info(f"Documentation score for {self.model_id}: {final_score:.2f}")
        return final_score

    def get_contributor_count(self) -> int:
        """Get number of contributors (approximation using downloads/likes)"""
        api_data = self.get_huggingface_api_data()

        downloads = api_data.get("downloads", 0)
        likes = api_data.get("likes", 0)

        # Improved heuristic
        if downloads > 100000 or likes > 500:
            return 20
        elif downloads > 10000 or likes > 100:
            return 10
        elif downloads > 1000 or likes > 20:
            return 5
        elif downloads > 100 or likes > 5:
            return 2
        else:
            return 1

    def get_tags(self) -> List[str]:
        """Get model tags"""
        api_data = self.get_huggingface_api_data()
        return api_data.get("tags", [])

    def get_downloads(self) -> int:
        """Get download count"""
        api_data = self.get_huggingface_api_data()
        return api_data.get("downloads", 0)

    def get_likes(self) -> int:
        """Get likes count"""
        api_data = self.get_huggingface_api_data()
        return api_data.get("likes", 0)
