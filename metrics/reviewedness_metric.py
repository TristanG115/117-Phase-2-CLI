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
            # No GitHub repository linked - return 0.0 instead of -1.0
            return 0.0, int((time.time() - start_time) * 1000)

        code_repo = code_resources[0]
        reviewedness_score = self._calculate_reviewedness(code_repo)

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        return round(reviewedness_score, 2), latency_ms

    def _calculate_reviewedness(self, code_repo: Any) -> float:
        """
        Calculate the fraction of code introduced through reviewed PRs

        Returns:
            0.0 if no GitHub repository or reviewedness not supported
            0.0 to 1.0 representing fraction of code with reviews
        """
        try:
            # Check if this is a GitHub repository
            if not hasattr(code_repo, "get_reviewedness_score"):
                self.logger.warning(
                    "Code repository does not support reviewedness calculation"
                )
                return 0.0

            score = code_repo.get_reviewedness_score()
            # Ensure we never return -1.0, convert to 0.0
            return max(0.0, score)

        except Exception as e:
            self.logger.error(f"Error calculating reviewedness: {e}")
            return 0.0
