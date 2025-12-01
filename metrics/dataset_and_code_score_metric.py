import time
from typing import Any, Dict, List, Tuple

from url_classifier import URLType

from .base_metric import BaseMetric


class DatasetAndCodeScoreMetric(BaseMetric):
    """Metric for availability and quality of training dataset and code"""

    def required_url_types(self) -> List[URLType]:
        return [URLType.DATASET, URLType.CODE]

    def calculate(
        self, resources: Dict[URLType, List[Any]], **kwargs
    ) -> Tuple[float, int]:
        start_time = time.time()

        dataset_available = URLType.DATASET in resources and resources[URLType.DATASET]
        code_available = URLType.CODE in resources and resources[URLType.CODE]

        score = 0.0

        # Dataset component (60% of total)
        if dataset_available:
            dataset = resources[URLType.DATASET][0]
            # Get dataset quality indicators
            try:
                downloads = (
                    dataset.get_downloads() if hasattr(dataset, "get_downloads") else 0
                )
                tags = dataset.get_tags() if hasattr(dataset, "get_tags") else []

                # Base score for having a dataset
                dataset_score = 0.3

                # Quality bonuses
                if downloads > 1000:
                    dataset_score += 0.2
                elif downloads > 100:
                    dataset_score += 0.1

                if len(tags) > 2:
                    dataset_score += 0.1

                score += min(dataset_score, 0.6)
            except Exception as e:
                self.logger.warning(f"Error evaluating dataset quality: {e}")
                # Fallback to basic score
                score += 0.4

        # Code component (40% of total)
        if code_available:
            code = resources[URLType.CODE][0]
            # Get code quality indicators
            try:
                has_tests = code.has_tests() if hasattr(code, "has_tests") else False
                has_ci = code.has_ci_cd() if hasattr(code, "has_ci_cd") else False
                stars = code.get_stars() if hasattr(code, "get_stars") else 0

                # Base score for having code
                code_score = 0.2

                # Quality bonuses
                if has_tests:
                    code_score += 0.1
                if has_ci:
                    code_score += 0.05
                if stars > 100:
                    code_score += 0.05

                score += min(code_score, 0.4)
            except Exception as e:
                self.logger.warning(f"Error evaluating code quality: {e}")
                # Fallback to basic score
                score += 0.25

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        return round(score, 2), latency_ms
