from typing import Tuple, Dict, List, Any
import time
from .base_metric import BaseMetric
from url_classifier import URLType


class LicenseMetric(BaseMetric):
    """Metric for license compatibility with LGPLv2.1"""

    def required_url_types(self) -> List[URLType]:
        # Only check MODEL license
        return [URLType.MODEL]

    def calculate(self, resources: Dict[URLType, List[Any]]) -> Tuple[float, int]:
        start_time = time.time()

        # Check license compatibility across all available resources
        scores = []

        for url_type in self.required_url_types():
            if url_type in resources and resources[url_type]:
                for resource in resources[url_type]:
                    license_score = self._evaluate_resource_license(resource)
                    scores.append(license_score)

        # Since we only check one model, take the first score
        final_score = scores[0] if scores else 0.0

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        return final_score, latency_ms

    def _evaluate_resource_license(self, resource: Any) -> float:
        # This would call the resource's license evaluation method
        try:
            return resource.get_license_score()
        except Exception as e:
            self.logger.error(f"Error evaluating license: {e}")
            return 0.0