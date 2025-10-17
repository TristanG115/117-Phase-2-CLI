from typing import Tuple, Dict, List, Any
import time
from .base_metric import BaseMetric
from url_classifier import URLType


class DatasetAndCodeScoreMetric(BaseMetric):
    """Metric for availability of training dataset and code"""

    def required_url_types(self) -> List[URLType]:
        return [URLType.DATASET, URLType.CODE]

    def calculate(self, resources: Dict[URLType, List[Any]]) -> Tuple[float, int]:
        start_time = time.time()

        dataset_available = URLType.DATASET in resources and resources[URLType.DATASET]
        code_available = URLType.CODE in resources and resources[URLType.CODE]

        score = 0.0
        if dataset_available:
            score += 0.6
        if code_available:
            score += 0.4

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        return score, latency_ms