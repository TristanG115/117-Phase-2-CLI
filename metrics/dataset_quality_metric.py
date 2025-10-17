from typing import Tuple, Dict, List, Any
import time
from .base_metric import BaseMetric
from url_classifier import URLType


class DatasetQualityMetric(BaseMetric):
    """Metric for dataset quality assessment"""

    def required_url_types(self) -> List[URLType]:
        return [URLType.DATASET]

    def calculate(self, resources: Dict[URLType, List[Any]]) -> Tuple[float, int]:
        start_time = time.time()

        if not resources.get(URLType.DATASET):
            return 0.0, int((time.time() - start_time) * 1000)

        dataset = resources[URLType.DATASET][0]
        quality_score = self._evaluate_dataset_quality(dataset)

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        return quality_score, latency_ms

    def _evaluate_dataset_quality(self, dataset: Any) -> float:
        try:
            return dataset.get_quality_score()
        except Exception as e:
            self.logger.error(f"Error evaluating dataset quality: {e}")
            return 0.0