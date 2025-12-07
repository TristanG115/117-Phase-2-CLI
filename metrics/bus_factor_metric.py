import time
from typing import Any, Dict, List, Tuple

from url_classifier import URLType

from .base_metric import BaseMetric


class BusFactorMetric(BaseMetric):
    """Metric for knowledge concentration risk"""

    def required_url_types(self) -> List[URLType]:
        # Bus factor should consider all related resources
        return [URLType.MODEL, URLType.DATASET, URLType.CODE]

    def calculate(
        self, resources: Dict[URLType, List[Any]], **kwargs
    ) -> Tuple[float, int]:
        start_time = time.time()

        contributor_counts = []

        for url_type in self.required_url_types():
            if url_type in resources and resources[url_type]:
                for resource in resources[url_type]:
                    contributor_count = self._get_contributor_count(resource)
                    contributor_counts.append(contributor_count)

        # Calculate bus factor based on contributor diversity
        avg_contributors = (
            sum(contributor_counts) / len(contributor_counts)
            if contributor_counts
            else 0
        )

        # More realistic scoring based on actual contributor patterns
        # Models on HuggingFace typically have 1-2 main contributors
        # Popular ones might have 5-20 based on downloads/likes
        if avg_contributors >= 15:
            final_score = 1.0
        elif avg_contributors >= 8:
            final_score = 0.9
        elif avg_contributors >= 5:
            final_score = 0.8
        elif avg_contributors >= 3:
            final_score = 0.7
        elif avg_contributors >= 2:
            final_score = 0.6
        else:
            # Single contributor gets 0.5 (not terrible, just concentrated)
            final_score = 0.5

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        return round(final_score, 2), latency_ms

    def _get_contributor_count(self, resource: Any) -> int:
        try:
            return resource.get_contributor_count()
        except Exception as e:
            self.logger.error(f"Error getting contributor count: {e}")
            return 1
