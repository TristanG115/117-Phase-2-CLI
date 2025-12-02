import time
from typing import Any, Dict, List, Tuple

from url_classifier import URLType

from .base_metric import BaseMetric


class ReviewednessMetric(BaseMetric):
    """Metric for calculating fraction of code introduced through reviewed PRs"""

    def required_url_types(self) -> List[URLType]:
        return [URLType.CODE]

    def calculate(
        self, resources: Dict[URLType, List[Any]], **kwargs
    ) -> Tuple[float, int]:
        start_time = time.time()

        code_resources = resources.get(URLType.CODE, [])
        if not code_resources:
            # No GitHub repository linked - return 0.0
            return 0.0, int((time.time() - start_time) * 1000)

        code_repo = code_resources[0]
        reviewedness_score = self._calculate_reviewedness(code_repo)

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        return round(reviewedness_score, 2), latency_ms

    def _calculate_reviewedness(self, code_repo: Any) -> float:
        """
        Calculate the fraction of code introduced through reviewed PRs

        Uses GitHub API data to estimate review coverage based on:
        - Number of contributors (more contributors = more likely to have review process)
        - Stars and forks (popular repos tend to have better review practices)
        - Presence of CI/CD (indicates formal development process)

        Returns:
            0.0 to 1.0 representing estimated fraction of code with reviews
        """
        try:
            score = 0.0

            # Try the direct method first (if CodeHandler implements it)
            if hasattr(code_repo, "get_reviewedness_score"):
                direct_score = code_repo.get_reviewedness_score()
                if direct_score > 0:
                    return max(0.0, direct_score)

            # Otherwise, calculate from available GitHub data
            # Get GitHub API data if available
            if hasattr(code_repo, "get_github_api_data"):
                try:
                    api_data = code_repo.get_github_api_data()

                    # Base score from repository characteristics
                    stars = api_data.get("stargazers_count", 0)
                    forks = api_data.get("forks_count", 0)

                    # Popular repos (1000+ stars) likely have review processes
                    if stars > 1000:
                        score += 0.3
                    elif stars > 100:
                        score += 0.2
                    elif stars > 10:
                        score += 0.1

                    # Active forks indicate collaborative development
                    if forks > 100:
                        score += 0.2
                    elif forks > 10:
                        score += 0.1

                except Exception as e:
                    self.logger.debug(f"Could not get GitHub API data: {e}")

            # Check for contributors (more contributors = more likely reviews)
            if hasattr(code_repo, "get_contributors"):
                try:
                    contributors = code_repo.get_contributors()
                    num_contributors = len(contributors) if contributors else 0

                    # Multiple contributors suggest peer review
                    if num_contributors > 10:
                        score += 0.3
                    elif num_contributors > 5:
                        score += 0.2
                    elif num_contributors > 2:
                        score += 0.1

                except Exception as e:
                    self.logger.debug(f"Could not get contributors: {e}")

            # Check for CI/CD (indicates formal review process)
            if hasattr(code_repo, "has_ci_cd"):
                try:
                    if code_repo.has_ci_cd():
                        score += 0.2
                except Exception as e:
                    self.logger.debug(f"Could not check CI/CD: {e}")

            # If we couldn't gather any indicators, give minimal score for having a repo
            if score == 0.0 and hasattr(code_repo, "repo_url"):
                score = 0.1  # Minimal score for having a code repository

            return min(score, 1.0)  # Cap at 1.0

        except Exception as e:
            self.logger.error(f"Error calculating reviewedness: {e}")
            return 0.0
