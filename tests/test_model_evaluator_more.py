import json
import os
import sys
import tempfile

import pytest

from model_evaluator import ModelEvaluator
from url_classifier import URLType


def test_create_resource_handlers():
    me = ModelEvaluator(max_workers=1)
    grouped = {URLType.MODEL: ["m1"], URLType.DATASET: ["d1"], URLType.CODE: ["c1"]}
    resources = me._create_resource_handlers(grouped)
    assert URLType.MODEL in resources and len(resources[URLType.MODEL]) == 1
    assert URLType.DATASET in resources and len(resources[URLType.DATASET]) == 1
    assert URLType.CODE in resources and len(resources[URLType.CODE]) == 1


def test_evaluate_single_model_success_and_failure(monkeypatch):
    me = ModelEvaluator(max_workers=1)

    class FakeModelHandler:
        def __init__(self, url):
            self.model_id = "owner/model"

    monkeypatch.setattr("model_evaluator.ModelHandler", FakeModelHandler)

    # Monkeypatch metric calculation to return predictable metrics
    monkeypatch.setattr(me, "_calculate_metrics_parallel", lambda resources, aid=None: {"license": {"score": 0.8, "latency": 10}})

    res = me._evaluate_single_model("https://huggingface.co/owner/model", {}, artifact_id=None)
    assert res is not None
    assert res["name"] == "model"

    # Simulate ModelHandler raising
    def bad_init(url):
        raise Exception("bad")

    monkeypatch.setattr("model_evaluator.ModelHandler", bad_init)
    assert me._evaluate_single_model("x", {}, None) is None


def test_safe_calculate_metric_and_treescore(monkeypatch):
    me = ModelEvaluator(max_workers=1)

    class BadMetric:
        def calculate(self, resources):
            raise Exception("boom")

    class TreeMetric:
        def calculate(self, resources, artifact_id=None):
            return 0.5, 10

    assert me._safe_calculate_metric(BadMetric(), {}) == (0.0, 0)
    assert me._safe_calculate_treescore(TreeMetric(), {}, artifact_id=123) == (0.5, 10)


def test_calculate_net_score_various():
    me = ModelEvaluator(max_workers=1)
    metric_results = {
        "license": {"score": 0.8, "latency": 10},
        "reviewedness": {"score": -1.0, "latency": 0},
        "size_score": {"score": {"a": 0.9, "b": 0.7}, "latency": 5},
    }

    net, latency = me._calculate_net_score(metric_results)
    assert isinstance(net, float)
    assert latency == 15


def test_evaluate_from_file_and_main(monkeypatch, tmp_path):
    p = tmp_path / "urls.txt"
    p.write_text("https://huggingface.co/owner/model\n")

    me = ModelEvaluator(max_workers=1)
    # Monkeypatch evaluate_urls to return a predictable result
    monkeypatch.setattr(me, "evaluate_urls", lambda urls: [{"name": "m", "category": "MODEL"}])
    results = me.evaluate_from_file(str(p))
    assert len(results) == 1

    # File not found
    assert me.evaluate_from_file(str(p) + ".nope") == []

    # main usage: wrong args
    monkeypatch.setattr(sys, "argv", ["model_evaluator.py"])
    with pytest.raises(SystemExit) as se:
        from model_evaluator import main

        main()
    assert se.value.code == 1

    # main with file but no results -> exit 1
    monkeypatch.setattr(sys, "argv", ["model_evaluator.py", str(p)])
    monkeypatch.setattr("model_evaluator.ModelEvaluator.evaluate_from_file", lambda self, pth: [])
    with pytest.raises(SystemExit) as se2:
        from model_evaluator import main

        main()
    assert se2.value.code == 1


def test_setup_logging(tmp_path, monkeypatch):
    me = ModelEvaluator(max_workers=1)
    monkeypatch.setenv("LOG_LEVEL", "1")
    log_file = tmp_path / "logs" / "out.log"
    monkeypatch.setenv("LOG_FILE", str(log_file))
    me.setup_logging()
    # log file should be created
    assert log_file.exists()
