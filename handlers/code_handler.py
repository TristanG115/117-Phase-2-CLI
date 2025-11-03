import os
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests

from .base_resource_handler import BaseResourceHandler


class CodeHandler(BaseResourceHandler):
    """Enhanced handler for GitHub code repository resources"""

    def __init__(self, url: str):
        super().__init__(url)
        self.repo_path = self._extract_repo_path()
        self._repo_tree = None

    def _extract_repo_path(self) -> str:
        """Extract owner/repo from GitHub URL"""
        parsed = urlparse(self.url)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 2:
            return f"{path_parts[0]}/{path_parts[1]}"
        return ""

    def _get_headers(self) -> Dict[str, str]:
        """Get headers with GitHub token if available"""
        headers = {"Accept": "application/vnd.github.v3+json"}
        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token:
            headers["Authorization"] = f"token {github_token}"
        return headers

    def get_github_api_data(self) -> Dict[str, Any]:
        """Get comprehensive data from GitHub API"""
        cached = self._cache_get("github_api_data")
        if cached:
            return cached

        try:
            api_url = f"https://api.github.com/repos/{self.repo_path}"
            response = requests.get(api_url, headers=self._get_headers(), timeout=10)

            if response.status_code == 200:
                data = response.json()
                self._cache_set("github_api_data", data)
                self.logger.info(
                    (
                        f"Fetched GitHub API data for {self.repo_path}: "
                        f"stars={data.get('stargazers_count', 0)}, "
                        f"forks={data.get('forks_count', 0)}"
                    )
                )
                return data
            elif response.status_code == 401:
                self.logger.error("GitHub API authentication failed - invalid token")
            elif response.status_code == 404:
                self.logger.warning(f"GitHub repository not found: {self.repo_path}")
            else:
                self.logger.warning(f"GitHub API returned {response.status_code} for {self.repo_path}")
        except Exception as e:
            self.logger.error(f"Error fetching GitHub API data: {e}")

        return {}

    def get_repo_tree(self) -> List[Dict[str, Any]]:
        """Get repository file tree (recursive)"""
        if self._repo_tree is not None:
            return self._repo_tree

        cached = self._cache_get("repo_tree")
        if cached:
            self._repo_tree = cached
            return cached

        try:
            # Get default branch first
            api_data = self.get_github_api_data()
            default_branch = api_data.get("default_branch", "main")

            tree_url = f"https://api.github.com/repos/{self.repo_path}/git/trees/{default_branch}?recursive=1"
            response = requests.get(tree_url, headers=self._get_headers(), timeout=10)

            if response.status_code == 200:
                tree_data = response.json()
                tree = tree_data.get("tree", [])
                self._repo_tree = tree
                self._cache_set("repo_tree", tree)
                self.logger.info(f"Fetched repository tree for {self.repo_path}: {len(tree)} items")
                return tree
            else:
                self.logger.warning(f"GitHub tree API returned {response.status_code} for {self.repo_path}")
        except Exception as e:
            self.logger.error(f"Error fetching repository tree: {e}")

        self._repo_tree = []
        return []

    def has_tests(self) -> bool:
        """Check if repository has test files"""
        tree = self.get_repo_tree()

        test_indicators = [
            "test/",
            "tests/",
            "/test/",
            "/tests/",
            "test_",
            "_test.py",
            "_test.js",
            "spec/",
            "/spec/",
            "_spec.",
            "__tests__/",
            "unittest",
            "pytest",
        ]

        for item in tree:
            path = item.get("path", "").lower()
            if any(indicator in path for indicator in test_indicators):
                self.logger.info(f"Found test files in {self.repo_path}")
                return True

        return False

    def has_ci_cd(self) -> bool:
        """Check if repository has CI/CD configuration"""
        tree = self.get_repo_tree()

        ci_indicators = [
            ".github/workflows/",
            ".travis.yml",
            ".circleci/",
            "azure-pipelines",
            "jenkinsfile",
            ".gitlab-ci.yml",
            "circle.yml",
        ]

        for item in tree:
            path = item.get("path", "").lower()
            if any(indicator in path for indicator in ci_indicators):
                self.logger.info(f"Found CI/CD config in {self.repo_path}")
                return True

        return False

    def has_linting_config(self) -> bool:
        """Check if repository has linting/formatting configuration"""
        tree = self.get_repo_tree()

        lint_files = [
            ".flake8",
            ".pylintrc",
            "pylint.cfg",
            "pyproject.toml",
            "setup.cfg",
            "tox.ini",
            ".eslintrc",
            ".prettierrc",
            ".pre-commit-config.yaml",
            "black.toml",
            ".isort.cfg",
        ]

        for item in tree:
            path = item.get("path", "").lower()
            if any(lint_file in path for lint_file in lint_files):
                self.logger.info(f"Found linting config in {self.repo_path}")
                return True

        return False

    def get_python_file_count(self) -> int:
        """Count Python files in repository"""
        tree = self.get_repo_tree()
        count = sum(1 for item in tree if item.get("path", "").endswith(".py"))
        self.logger.debug(f"Found {count} Python files in {self.repo_path}")
        return count

    def has_evaluation_code(self) -> bool:
        """Check if repository has evaluation code"""
        tree = self.get_repo_tree()

        eval_indicators = ["eval", "evaluation", "benchmark", "test"]

        for item in tree:
            path = item.get("path", "").lower()
            if any(indicator in path for indicator in eval_indicators):
                if path.endswith(".py") or path.endswith(".ipynb"):
                    self.logger.info(f"Found evaluation code in {self.repo_path}")
                    return True

        return False

    def get_code_quality_score(self) -> float:
        """Evaluate code quality comprehensively"""
        cached = self._cache_get("code_quality_score")
        if cached is not None:
            return cached

        api_data = self.get_github_api_data()
        if not api_data:
            return 0.0

        score = 0.0

        # README (30%)
        if api_data.get("has_readme"):
            score += 0.3

        # Stars - community validation (25%)
        stars = api_data.get("stargazers_count", 0)
        if stars > 1000:
            score += 0.25
        elif stars > 100:
            score += 0.2
        elif stars > 10:
            score += 0.15
        elif stars > 0:
            score += 0.1

        # Recent activity (20%)
        updated_at = api_data.get("updated_at", "")
        if "2025" in updated_at or "2024" in updated_at:
            score += 0.2
        elif "2023" in updated_at:
            score += 0.1

        # Tests (10%)
        if self.has_tests():
            score += 0.1

        # CI/CD (10%)
        if self.has_ci_cd():
            score += 0.1

        # Linting (5%)
        if self.has_linting_config():
            score += 0.05

        final_score = min(score, 1.0)
        self._cache_set("code_quality_score", final_score)
        self.logger.info(f"Code quality score for {self.repo_path}: {final_score:.2f}")
        return final_score

    def get_license_score(self) -> float:
        """Get license compatibility score from GitHub"""
        api_data = self.get_github_api_data()

        license_info = api_data.get("license")
        if license_info and isinstance(license_info, dict):
            spdx_id = license_info.get("spdx_id")
            if spdx_id and spdx_id != "NOASSERTION":
                score = self._parse_license_identifier(spdx_id)
                if score > 0:
                    self.logger.info(f"License for {self.repo_path}: {spdx_id} (score={score})")
                    return score

        self.logger.warning(f"No license found for {self.repo_path}")
        return 0.0

    def get_documentation_score(self) -> float:
        """Evaluate documentation quality comprehensively"""
        cached = self._cache_get("doc_score")
        if cached is not None:
            return cached

        api_data = self.get_github_api_data()
        if not api_data:
            return 0.0

        score = 0.0

        # README (40%)
        if api_data.get("has_readme"):
            score += 0.4

        # Description (20%)
        description = api_data.get("description", "")
        if len(description) > 100:
            score += 0.2
        elif len(description) > 50:
            score += 0.15
        elif len(description) > 0:
            score += 0.1

        # Wiki (20%)
        if api_data.get("has_wiki"):
            score += 0.2

        # Homepage/docs link (10%)
        if api_data.get("homepage"):
            score += 0.1

        # Check for docs folder
        tree = self.get_repo_tree()
        has_docs_folder = any("docs/" in item.get("path", "").lower() for item in tree)
        if has_docs_folder:
            score += 0.1

        final_score = min(score, 1.0)
        self._cache_set("doc_score", final_score)
        self.logger.info(f"Documentation score for {self.repo_path}: {final_score:.2f}")
        return final_score

    def get_contributor_count(self) -> int:
        """Get actual number of contributors from GitHub API"""
        cached = self._cache_get("contributor_count")
        if cached is not None:
            return cached

        try:
            contributors_url = f"https://api.github.com/repos/{self.repo_path}/contributors?per_page=100"
            response = requests.get(contributors_url, headers=self._get_headers(), timeout=10)

            if response.status_code == 200:
                contributors = response.json()
                count = len(contributors)
                self._cache_set("contributor_count", count)
                self.logger.info(f"Found {count} contributors for {self.repo_path}")
                return count
        except Exception as e:
            self.logger.error(f"Error getting contributor count: {e}")

        # Fallback: estimate from stars
        api_data = self.get_github_api_data()
        stars = api_data.get("stargazers_count", 0)
        if stars > 1000:
            return 10
        elif stars > 100:
            return 5
        elif stars > 10:
            return 2
        return 1

    def get_stars(self) -> int:
        """Get star count"""
        api_data = self.get_github_api_data()
        return api_data.get("stargazers_count", 0)

    def get_forks(self) -> int:
        """Get fork count"""
        api_data = self.get_github_api_data()
        return api_data.get("forks_count", 0)
