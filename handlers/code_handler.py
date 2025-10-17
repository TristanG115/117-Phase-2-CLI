from typing import Dict, Any
import requests
import os
from urllib.parse import urlparse

from .base_resource_handler import BaseResourceHandler


class CodeHandler(BaseResourceHandler):
    """Handler for GitHub code repository resources"""

    def __init__(self, url: str):
        super().__init__(url)
        self.repo_path = self._extract_repo_path()

    def _extract_repo_path(self) -> str:
        """Extract owner/repo from GitHub URL"""
        parsed = urlparse(self.url)
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) >= 2:
            return f"{path_parts[0]}/{path_parts[1]}"
        return ""

    def get_github_api_data(self) -> Dict[str, Any]:
        """Get data from GitHub API"""
        cached = self._cache_get('github_api_data')
        if cached:
            return cached

        try:
            api_url = f"https://api.github.com/repos/{self.repo_path}"
            headers = {}

            # Add GitHub token if available
            github_token = os.environ.get('GITHUB_TOKEN')
            if github_token:
                headers['Authorization'] = f'token {github_token}'

            response = requests.get(api_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self._cache_set('github_api_data', data)
                return data
            elif response.status_code == 401:
                self.logger.error("GitHub API authentication failed - invalid token")
                # Continue without authentication for rate-limited access
                return {}
        except Exception as e:
            self.logger.error(f"Error fetching GitHub API data: {e}")

        return {}

    def has_evaluation_code(self) -> bool:
        """Check if repository has evaluation code"""
        try:
            # Search for evaluation-related files
            search_url = f"https://api.github.com/search/code?q=repo:{self.repo_path}+evaluation+test+benchmark"
            headers = {}

            # Add GitHub token if available
            github_token = os.environ.get('GITHUB_TOKEN')
            if github_token:
                headers['Authorization'] = f'token {github_token}'

            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code == 200:
                results = response.json()
                return results.get('total_count', 0) > 0
            elif response.status_code == 401:
                self.logger.error("GitHub API authentication failed - invalid token")
                return False
        except Exception as e:
            self.logger.error(f"Error checking evaluation code: {e}")

        return False

    def get_code_quality_score(self) -> float:
        """Evaluate code quality"""
        api_data = self.get_github_api_data()

        score = 0.0

        # Check for README
        if api_data.get('has_readme'):
            score += 0.3

        # Check stars
        stars = api_data.get('stargazers_count', 0)
        if stars > 100:
            score += 0.4
        elif stars > 10:
            score += 0.3
        elif stars > 0:
            score += 0.2

        # Check for recent activity
        updated_at = api_data.get('updated_at', '')
        if updated_at and '2024' in updated_at or '2023' in updated_at:
            score += 0.3

        # Check for issues/community engagement
        if api_data.get('has_issues'):
            score += 0.2

        return min(score, 1.0)

    def get_license_score(self) -> float:
        """License evaluation not applicable for code repositories"""
        return 0.0

    def get_documentation_score(self) -> float:
        """Evaluate documentation quality"""
        api_data = self.get_github_api_data()

        score = 0.0
        if api_data.get('description'):
            score += 0.3
        if api_data.get('has_readme'):
            score += 0.4
        if api_data.get('has_wiki'):
            score += 0.2
        if api_data.get('homepage'):
            score += 0.1

        return min(score, 1.0)

    def get_contributor_count(self) -> int:
        """Get number of contributors"""
        try:
            contributors_url = f"https://api.github.com/repos/{self.repo_path}/contributors"
            response = requests.get(contributors_url, timeout=10)
            if response.status_code == 200:
                contributors = response.json()
                return len(contributors)
        except Exception as e:
            self.logger.error(f"Error getting contributor count: {e}")

        return 1
