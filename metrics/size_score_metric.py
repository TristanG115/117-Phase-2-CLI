import time
from typing import Any, Dict, List, Tuple

from url_classifier import URLType

from .base_metric import BaseMetric


class SizeScoreMetric(BaseMetric):
    """Metric for model size compatibility with different hardware"""

    def required_url_types(self) -> List[URLType]:
        # Size is primarily a model concern
        return [URLType.MODEL]

    def calculate(self, resources, **kwargs) -> Tuple[Dict[str, float], int]:  # type: ignore[override]
        start_time = time.time()

        if not resources.get(URLType.MODEL):
            return {}, int((time.time() - start_time) * 1000)

        model = resources[URLType.MODEL][0]  # Assume one model
        size_dict = self._calculate_hardware_compatibility(model)

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        return size_dict, latency_ms

    def _calculate_hardware_compatibility(self, model: Any) -> Dict[str, float]:
        try:
            model_size_mb = model.get_size_mb()

            # Hardware compatibility thresholds (example)
            return {
                "raspberry_pi": (
                    1.0 if model_size_mb < 100 else 0.5 if model_size_mb < 500 else 0.0
                ),
                "jetson_nano": (
                    1.0
                    if model_size_mb < 1000
                    else 0.7 if model_size_mb < 2000 else 0.3
                ),
                "desktop_pc": (
                    1.0
                    if model_size_mb < 5000
                    else 0.8 if model_size_mb < 10000 else 0.5
                ),
                "aws_server": 1.0 if model_size_mb < 20000 else 0.9,
            }
        except Exception as e:
            self.logger.error(f"Error calculating size compatibility: {e}")
            return {
                "raspberry_pi": 0.0,
                "jetson_nano": 0.0,
                "desktop_pc": 0.5,
                "aws_server": 0.5,
            }
