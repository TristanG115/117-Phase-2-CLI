import time
from typing import Any, Dict, List, Tuple

from url_classifier import URLType

from .base_metric import BaseMetric


class CodeQualityMetric(BaseMetric):
    """Metric for code quality assessment"""

    def required_url_types(self) -> List[URLType]:
        return [URLType.CODE]

    def calculate(
        self, resources: Dict[URLType, List[Any]], **kwargs
    ) -> Tuple[float, int]:
        start_time = time.time()

        if not resources.get(URLType.CODE):
            return 0.0, int((time.time() - start_time) * 1000)

        code_repo = resources[URLType.CODE][0]
        quality_score = self._evaluate_code_quality(code_repo)

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        return round(quality_score, 2), latency_ms

    def _evaluate_code_quality(self, code_repo: Any) -> float:
        try:
            return code_repo.get_code_quality_score()
        except Exception as e:
            self.logger.error(f"Error evaluating code quality: {e}")
            return 0.0
