from .base_metric import BaseMetric
from .license_metric import LicenseMetric
from .size_score_metric import SizeScoreMetric
from .ramp_up_time_metric import RampUpTimeMetric
from .bus_factor_metric import BusFactorMetric
from .performance_claims_metric import PerformanceClaimsMetric
from .dataset_and_code_score_metric import DatasetAndCodeScoreMetric
from .dataset_quality_metric import DatasetQualityMetric
from .code_quality_metric import CodeQualityMetric


# Metric registry for easy access
METRIC_CLASSES = {
    'license': LicenseMetric,
    'size_score': SizeScoreMetric,
    'ramp_up_time': RampUpTimeMetric,
    'bus_factor': BusFactorMetric,
    'performance_claims': PerformanceClaimsMetric,
    'dataset_and_code_score': DatasetAndCodeScoreMetric,
    'dataset_quality': DatasetQualityMetric,
    'code_quality': CodeQualityMetric
}