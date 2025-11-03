from .base_metric import BaseMetric
from .bus_factor_metric import BusFactorMetric
from .code_quality_metric import CodeQualityMetric
from .dataset_and_code_score_metric import DatasetAndCodeScoreMetric
from .dataset_quality_metric import DatasetQualityMetric
from .license_metric import LicenseMetric
from .performance_claims_metric import PerformanceClaimsMetric
from .ramp_up_time_metric import RampUpTimeMetric
from .size_score_metric import SizeScoreMetric

# Metric registry for easy access
METRIC_CLASSES = {
    "license": LicenseMetric,
    "size_score": SizeScoreMetric,
    "ramp_up_time": RampUpTimeMetric,
    "bus_factor": BusFactorMetric,
    "performance_claims": PerformanceClaimsMetric,
    "dataset_and_code_score": DatasetAndCodeScoreMetric,
    "dataset_quality": DatasetQualityMetric,
    "code_quality": CodeQualityMetric,
}
