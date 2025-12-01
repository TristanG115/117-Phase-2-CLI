import time
from typing import Any, Dict, List, Tuple

from url_classifier import URLType

from .base_metric import BaseMetric


class PerformanceClaimsMetric(BaseMetric):
    """Metric for evidence of performance claims"""

    def required_url_types(self) -> List[URLType]:
        # Performance claims need model + dataset + evaluation code
        return [URLType.MODEL, URLType.DATASET, URLType.CODE]

    def calculate(
        self, resources: Dict[URLType, List[Any]], **kwargs
    ) -> Tuple[float, int]:
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

        # More lenient scoring: benchmarks alone can give a good score
        score = 0.0
        if has_benchmarks:
            # Benchmarks in model README are most important
            score += 0.7
        if has_evaluation_code:
            # Having eval code is a bonus
            score += 0.2
        if has_dataset_info:
            # Having eval dataset is also a bonus
            score += 0.1

        # Even without benchmarks, having eval code/dataset shows some effort
        if not has_benchmarks and (has_evaluation_code or has_dataset_info):
            score = max(score, 0.4)

        final_score = min(score, 1.0)

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        return round(final_score, 2), latency_ms
