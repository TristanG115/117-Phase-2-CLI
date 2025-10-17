from typing import Tuple, Dict, List, Any
import time
from .base_metric import BaseMetric
from url_classifier import URLType


class RampUpTimeMetric(BaseMetric):
    """Metric for ease of getting started with the model"""

    def required_url_types(self) -> List[URLType]:
        # Ramp-up depends on documentation quality across all resources
        return [URLType.MODEL, URLType.DATASET, URLType.CODE]

    def calculate(self, resources: Dict[URLType, List[Any]]) -> Tuple[float, int]:
        start_time = time.time()

        documentation_scores = []

        for url_type in self.required_url_types():
            if url_type in resources and resources[url_type]:
                for resource in resources[url_type]:
                    doc_score = self._evaluate_documentation_quality(resource)
                    documentation_scores.append(doc_score)

        # Average documentation quality across all resources
        final_score = sum(documentation_scores) / len(documentation_scores) if documentation_scores else 0.0

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        return final_score, latency_ms

    def _evaluate_documentation_quality(self, resource: Any) -> float:
        try:
            return resource.get_documentation_score()
        except Exception as e:
            self.logger.error(f"Error evaluating documentation: {e}")
            return 0.0