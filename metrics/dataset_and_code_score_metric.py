import time
from typing import Any, Dict, List, Tuple

from url_classifier import URLType

from .base_metric import BaseMetric


class DatasetAndCodeScoreMetric(BaseMetric):
    """Metric for availability of training dataset and code"""

    def required_url_types(self) -> List[URLType]:
        return [URLType.DATASET, URLType.CODE]

    def calculate(
        self, resources: Dict[URLType, List[Any]], **kwargs
    ) -> Tuple[float, int]:
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

        return round(score, 2), latency_ms
