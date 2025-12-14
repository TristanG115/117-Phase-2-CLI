from metrics.dataset_and_code_score_metric import DatasetAndCodeScoreMetric
from metrics.bus_factor_metric import BusFactorMetric
from url_classifier import URLType
from unittest.mock import Mock


def test_dataset_and_code_score_metric_basic():
    m = DatasetAndCodeScoreMetric()
    dataset = Mock(); dataset.get_downloads.return_value = 2000; dataset.get_tags.return_value = [1,2,3]
    code = Mock(); code.has_tests.return_value = True; code.has_ci_cd.return_value = True; code.get_stars.return_value = 200

    score, _ = m.calculate({URLType.DATASET: [dataset], URLType.CODE: [code]})
    assert score > 0

    # Error branch in dataset
    bad_dataset = Mock()
    bad_dataset.get_downloads.side_effect = Exception("boom")
    score2, _ = m.calculate({URLType.DATASET: [bad_dataset]})
    assert score2 >= 0


def test_bus_factor_metric_thresholds():
    m = BusFactorMetric()
    a = Mock(); a.get_contributor_count.return_value = 20
    b = Mock(); b.get_contributor_count.return_value = 1

    score, _ = m.calculate({URLType.MODEL: [a], URLType.DATASET: [a], URLType.CODE: [a]})
    assert score >= 0.9

    # Error getting contributor count logs error and falls back
    bad = Mock(); bad.get_contributor_count.side_effect = Exception("boom")
    score2, _ = m.calculate({URLType.MODEL: [bad]})
    assert score2 >= 0.5
