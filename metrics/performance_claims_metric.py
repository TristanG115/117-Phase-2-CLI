from typing import Tuple, Dict, List, Any
import time
from .base_metric import BaseMetric
from url_classifier import URLType


class PerformanceClaimsMetric(BaseMetric):
    """Metric for evidence of performance claims"""

    def required_url_types(self) -> List[URLType]:
        # Performance claims need model + dataset + evaluation code
        return [URLType.MODEL, URLType.DATASET, URLType.CODE]

    def calculate(self, resources: Dict[URLType, List[Any]]) -> Tuple[float, int]:
        start_time = time.time()

        has_benchmarks = False
        has_evaluation_code = False
        has_dataset_info = False

        # Check for benchmarks in model
        if URLType.MODEL in resources and resources[URLType.MODEL]:
            has_benchmarks = resources[URLType.MODEL][0].has_performance_benchmarks()

        # Check for evaluation code
        if URLType.CODE in resources and resources[URLType.CODE]:
            has_evaluation_code = resources[URLType.CODE][0].has_evaluation_code()

        # Check for dataset information
        if URLType.DATASET in resources and resources[URLType.DATASET]:
            has_dataset_info = resources[URLType.DATASET][0].has_evaluation_dataset()

        # Score based on available evidence
        score = 0.0
        if has_benchmarks:
            score += 0.5
        if has_evaluation_code:
            score += 0.3
        if has_dataset_info:
            score += 0.2

        final_score = min(score, 1.0)

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        return final_score, latency_ms