from typing import Tuple, Dict, List, Any
import time
from .base_metric import BaseMetric
from url_classifier import URLType


class BusFactorMetric(BaseMetric):
    """Metric for knowledge concentration risk"""

    def required_url_types(self) -> List[URLType]:
        # Bus factor should consider all related resources
        return [URLType.MODEL, URLType.DATASET, URLType.CODE]

    def calculate(self, resources: Dict[URLType, List[Any]]) -> Tuple[float, int]:
        start_time = time.time()

        contributor_counts = []

        for url_type in self.required_url_types():
            if url_type in resources and resources[url_type]:
                for resource in resources[url_type]:
                    contributor_count = self._get_contributor_count(resource)
                    contributor_counts.append(contributor_count)

        # Calculate bus factor based on contributor diversity
        avg_contributors = sum(contributor_counts) / len(contributor_counts) if contributor_counts else 0

        # Convert to 0-1 score (more contributors = higher score)
        if avg_contributors >= 10:
            final_score = 1.0
        elif avg_contributors >= 5:
            final_score = 0.8
        elif avg_contributors >= 2:
            final_score = 0.5
        else:
            final_score = 0.2

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        return final_score, latency_ms

    def _get_contributor_count(self, resource: Any) -> int:
        try:
            return resource.get_contributor_count()
        except Exception as e:
            self.logger.error(f"Error getting contributor count: {e}")
            return 1