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
        """
        Calculate hardware compatibility based on model size.

        Updated thresholds based on realistic hardware constraints:
        - Raspberry Pi 4: 4-8GB RAM, can handle models up to ~2GB
        - Jetson Nano: 4GB RAM, optimized for ML, can handle up to ~3GB
        - Desktop PC: 16-32GB RAM typical, can handle up to ~8GB comfortably
        - AWS Server: Variable, but assume large instances can handle anything
        """
        try:
            model_size_mb = model.get_size_mb()
            self.logger.info(
                f"Calculating hardware compatibility for model size: {model_size_mb:.2f} MB"
            )

            # Convert MB to GB for clearer thresholds
            model_size_gb = model_size_mb / 1024

            return {
                "raspberry_pi": self._raspberry_pi_score(model_size_gb),
                "jetson_nano": self._jetson_nano_score(model_size_gb),
                "desktop_pc": self._desktop_pc_score(model_size_gb),
                "aws_server": self._aws_server_score(model_size_gb),
            }
        except Exception as e:
            self.logger.error(f"Error calculating size compatibility: {e}")
            # Return conservative scores on error
            return {
                "raspberry_pi": 0.3,
                "jetson_nano": 0.4,
                "desktop_pc": 0.6,
                "aws_server": 0.8,
            }

    def _raspberry_pi_score(self, size_gb: float) -> float:
        """Raspberry Pi 4 with 4-8GB RAM"""
        if size_gb < 0.5:  # <500MB - very small models
            return 1.0
        elif size_gb < 1.0:  # <1GB - small models
            return 0.9
        elif size_gb < 2.0:  # <2GB - might work with optimization
            return 0.7
        elif size_gb < 3.0:  # <3GB - challenging but possible
            return 0.4
        else:  # >=3GB - likely too large
            return 0.2

    def _jetson_nano_score(self, size_gb: float) -> float:
        """Jetson Nano with 4GB RAM, optimized for ML"""
        if size_gb < 1.0:  # <1GB - excellent
            return 1.0
        elif size_gb < 2.0:  # <2GB - very good
            return 0.9
        elif size_gb < 3.0:  # <3GB - good
            return 0.8
        elif size_gb < 4.0:  # <4GB - possible with optimization
            return 0.6
        elif size_gb < 6.0:  # <6GB - challenging
            return 0.4
        else:  # >=6GB - likely too large
            return 0.2

    def _desktop_pc_score(self, size_gb: float) -> float:
        """Desktop PC with 16-32GB RAM"""
        if size_gb < 2.0:  # <2GB - trivial
            return 1.0
        elif size_gb < 4.0:  # <4GB - easy
            return 1.0
        elif size_gb < 8.0:  # <8GB - comfortable
            return 0.9
        elif size_gb < 12.0:  # <12GB - manageable
            return 0.8
        elif size_gb < 16.0:  # <16GB - possible on high-end systems
            return 0.7
        else:  # >=16GB - requires server-class hardware
            return 0.5

    def _aws_server_score(self, size_gb: float) -> float:
        """AWS server with flexible resources"""
        if size_gb < 10.0:  # <10GB - any instance
            return 1.0
        elif size_gb < 20.0:  # <20GB - medium instances
            return 1.0
        elif size_gb < 50.0:  # <50GB - large instances
            return 0.95
        elif size_gb < 100.0:  # <100GB - xlarge instances
            return 0.9
        else:  # >=100GB - very large, but still possible
            return 0.85
