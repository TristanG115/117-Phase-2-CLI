import time
from typing import Any, Dict, List, Tuple

from url_classifier import URLType

from .base_metric import BaseMetric


class CodeQualityMetric(BaseMetric):
    """Metric for code quality assessment"""

    def required_url_types(self) -> List[URLType]:
        return [URLType.CODE]

    def calculate(
        self, resources: Dict[URLType, List[Any]], **kwargs
    ) -> Tuple[float, int]:
        start_time = time.time()

        if not resources.get(URLType.CODE):
            return 0.0, int((time.time() - start_time) * 1000)

        code_repo = resources[URLType.CODE][0]
        quality_score = self._evaluate_code_quality(code_repo)

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        return round(quality_score, 2), latency_ms

    def _evaluate_code_quality(self, code_repo: Any) -> float:
        """Evaluate code quality with comprehensive fallback logic"""
        try:
            # Try the handler's built-in method first
            if hasattr(code_repo, "get_code_quality_score"):
                return code_repo.get_code_quality_score()
        except Exception as e:
            self.logger.debug(f"Handler's quality score failed: {e}")

        # Fallback: calculate from available indicators
        try:
            score = 0.3  # Base score for having a code repository

            # Get GitHub API data if available
            api_data = {}
            if hasattr(code_repo, "get_github_api_data"):
                api_data = code_repo.get_github_api_data() or {}

            # README indicates documentation (0.15)
            if api_data.get("has_readme"):
                score += 0.15
                self.logger.debug("Has README: +0.15")

            # Tests indicate quality practices (0.2)
            if hasattr(code_repo, "has_tests") and code_repo.has_tests():
                score += 0.2
                self.logger.debug("Has tests: +0.2")

            # CI/CD indicates automation (0.15)
            if hasattr(code_repo, "has_ci_cd") and code_repo.has_ci_cd():
                score += 0.15
                self.logger.debug("Has CI/CD: +0.15")

            # Stars indicate community validation (0.1)
            stars = 0
            if hasattr(code_repo, "get_stars"):
                stars = code_repo.get_stars()
            elif api_data:
                stars = api_data.get("stargazers_count", 0)

            if stars > 1000:
                score += 0.1
                self.logger.debug(f"Has {stars} stars (>1000): +0.1")
            elif stars > 100:
                score += 0.07
                self.logger.debug(f"Has {stars} stars (>100): +0.07")
            elif stars > 10:
                score += 0.04
                self.logger.debug(f"Has {stars} stars (>10): +0.04")

            # Issues and activity indicate maintenance (0.1)
            if api_data:
                open_issues = api_data.get("open_issues_count", 0)
                # Having some open issues is normal; too many is bad
                if 1 <= open_issues <= 50:
                    score += 0.1
                    self.logger.debug("Healthy issue count: +0.1")
                elif open_issues == 0:
                    score += 0.05
                    self.logger.debug("No open issues: +0.05")

            return min(score, 1.0)

        except Exception as e:
            self.logger.error(f"Error in fallback code quality evaluation: {e}")
            # Minimal baseline for having code
            return 0.3
