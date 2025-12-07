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
            try:
                has_benchmarks = resources[URLType.MODEL][
                    0
                ].has_performance_benchmarks()
            except:
                pass

        # Check for evaluation code
        if URLType.CODE in resources and resources[URLType.CODE]:
            try:
                has_evaluation_code = resources[URLType.CODE][0].has_evaluation_code()
            except:
                pass

        # Check for dataset information
        if URLType.DATASET in resources and resources[URLType.DATASET]:
            try:
                has_dataset_info = resources[URLType.DATASET][
                    0
                ].has_evaluation_dataset()
            except:
                pass

        # Stricter scoring - require actual evidence of performance claims
        score = 0.0

        if has_benchmarks:
            # Benchmarks in model README are excellent evidence
            score = 0.9
        elif has_evaluation_code:
            # Evaluation code alone is good evidence
            score = 0.65
        elif has_dataset_info:
            # Evaluation dataset info is moderate evidence
            score = 0.5

        # Bonus points for multiple types of evidence
        if has_benchmarks and has_evaluation_code:
            score = 1.0
        elif has_benchmarks and has_dataset_info:
            score = 0.95
        elif has_evaluation_code and has_dataset_info:
            score = 0.75

        # Only give minimal baseline if we have a model with some documentation
        # but no explicit performance evidence
        if score == 0.0 and URLType.MODEL in resources and resources[URLType.MODEL]:
            try:
                model = resources[URLType.MODEL][0]
                # Check if model has ANY documentation at all
                if hasattr(model, "get_huggingface_api_data"):
                    api_data = model.get_huggingface_api_data()
                    if api_data and (
                        api_data.get("cardData") or api_data.get("description")
                    ):
                        score = 0.3  # Minimal score for having documentation
            except:
                pass

        final_score = min(score, 1.0)

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        return round(final_score, 2), latency_ms
