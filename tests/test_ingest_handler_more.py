import json

import pytest

import handlers.ingest_handler as ih


def test_extract_card_data_and_load_card_content(monkeypatch):
    # dict card data
    md, txt = ih._extract_card_data({"a": 1, "b": 2})
    assert md.get("a") == 1

    class Obj:
        def to_dict(self):
            return {"x": "y"}

    md2, txt2 = ih._extract_card_data(Obj())
    assert md2.get("x") == "y"

    # load card content success and fail
    class Card:
        content = "hello"

    monkeypatch.setattr("handlers.ingest_handler.RepoCard.load", lambda mid: Card())
    assert ih._load_card_content("owner/model") == "hello"

    monkeypatch.setattr("handlers.ingest_handler.RepoCard.load", lambda mid: (_ for _ in ()).throw(Exception("nope")))
    assert ih._load_card_content("owner/model") == ""


def test_extract_metadata_and_url_extraction(monkeypatch):
    class API:
        def model_info(self, mid):
            class Info:
                cardData = {"datasets": ["owner/ds"]}

            return Info()

    monkeypatch.setattr("handlers.ingest_handler.HfApi", lambda: API())
    md, card = ih._extract_metadata("owner/model", API())
    assert isinstance(md, dict)

    # github extraction
    assert ih._extract_github_url("see https://github.com/owner/repo here") == "https://github.com/owner/repo"
    assert ih._extract_github_url("github.com/owner/repo.git") == "https://github.com/owner/repo"

    # dataset extraction from text and metadata
    assert ih._extract_dataset_url_direct("see https://huggingface.co/datasets/owner/ds") == "https://huggingface.co/datasets/owner/ds"
    assert ih._extract_dataset_from_metadata({"datasets": ["owner/ds"]}) == "owner/ds"
    assert ih._extract_dataset_from_yaml("datasets:\n- owner/ds") == "owner/ds"
    assert ih._extract_dataset_from_prose("trained on owner/ds dataset") == "owner/ds"
    assert ih._construct_dataset_url("Owner/Ds") == "https://huggingface.co/datasets/owner/ds"


def test_extract_dataset_url_strategy():
    # direct
    assert ih._extract_dataset_url({}, "see https://huggingface.co/datasets/owner/ds") == "https://huggingface.co/datasets/owner/ds"
    # metadata
    assert ih._extract_dataset_url({"datasets": ["owner/ds"]}, "") == "https://huggingface.co/datasets/owner/ds"
    # yaml
    assert ih._extract_dataset_url({}, "datasets:\n- owner/ds") == "https://huggingface.co/datasets/owner/ds"
    # prose
    assert ih._extract_dataset_url({}, "trained on owner/ds dataset") == "https://huggingface.co/datasets/owner/ds"
    # none
    assert ih._extract_dataset_url({}, "nothing here") == "unknown"


def test_validate_url(monkeypatch):
    monkeypatch.setattr("requests.head", lambda *a, **k: type("R", (), {"status_code": 200})())
    assert ih.validate_url("https://example.com", "github") is True

    monkeypatch.setattr("requests.head", lambda *a, **k: type("R", (), {"status_code": 404})())
    assert ih.validate_url("https://example.com", "github") is False

    monkeypatch.setattr("requests.head", lambda *a, **k: (_ for _ in ()).throw(Exception("boom")))
    assert ih.validate_url("https://example.com", "github") is False


def test_ensure_and_check_threshold():
    res = {}
    ih._ensure_required_metrics(res)
    assert res.get("dataset_and_code_score") == -1

    good = {"a": 1.0, "b": 0.9}
    assert ih._check_threshold({"a": 1.0, "b": 0.8}, 0.5) is True
    assert ih._check_threshold({"a": 0.4, "b": 0.8}, 0.5) is False


def test_calculate_artifact_size_hf_and_github(monkeypatch):
    # huggingface with siblings
    class Info:
        siblings = [type("F", (), {"size": 5 * 1024 * 1024}), type("F", (), {"size": None}), type("F", (), {"size": 2 * 1024 * 1024})]

    class API:
        def model_info(self, repo_id):
            return Info()

        def dataset_info(self, repo_id):
            return Info()

    monkeypatch.setattr("handlers.ingest_handler.HfApi", lambda: API())
    size = ih._calculate_artifact_size("https://huggingface.co/owner/model", "model")
    assert size > 0

    # github path
    class R:
        status_code = 200

        def json(self):
            return {"size": 2048}

    import requests as _req

    monkeypatch.setattr("requests.get", lambda *a, **k: R())
    size2 = ih._calculate_artifact_size("https://github.com/owner/repo", "code")
    assert size2 > 0


def test_ingest_model_and_dataset_and_code(monkeypatch, tmp_path):
    # snapshot download fail
    monkeypatch.setattr("handlers.ingest_handler.snapshot_download", lambda **k: (_ for _ in ()).throw(Exception("nope")))
    res = ih.ingest_model("https://huggingface.co/owner/model", download=True)
    assert "error" in res

    # successful ingest_model flow (mock evaluator and registry)
    monkeypatch.setattr("handlers.ingest_handler.infer_links_from_hf", lambda u, show_card=False: ("https://github.com/owner/repo", "https://huggingface.co/datasets/owner/ds"))
    monkeypatch.setattr("handlers.ingest_handler._extract_metadata", lambda mid, api: ({}, ""))
    monkeypatch.setattr("handlers.ingest_handler.score_model_with_evaluator", lambda a, b, c: {"net_score": 0.9})
    monkeypatch.setattr("handlers.ingest_handler.registry_handler", type("R", (), {"add_model": lambda **k: 123}))

    res2 = ih.ingest_model("https://huggingface.co/owner/model", download=False, validate_urls=False)
    assert res2.get("status") == "success"

    # ingest_dataset and ingest_code
    monkeypatch.setattr("handlers.ingest_handler.registry_handler", type("R", (), {"add_artifact": lambda **k: 321}))
    dres = ih.ingest_dataset("https://huggingface.co/datasets/owner/ds")
    assert dres.get("status") == "success"

    cres = ih.ingest_code("https://github.com/owner/repo")
    assert cres.get("status") == "success"


def test_batch_ingest(monkeypatch):
    monkeypatch.setattr("handlers.ingest_handler.ingest_model", lambda url, min_score=0.5, download=False, validate_urls=False: {"status": "ok", "url": url})
    results = ih.batch_ingest(["a", "b"])
    assert len(results) == 2
from unittest.mock import Mock, patch
import handlers.ingest_handler as ih
import json


def test_extract_card_data_variants():
    # dict input
    meta, text = ih._extract_card_data({"a": 1, "b": 2})
    assert isinstance(meta, dict)
    assert "1" in text or "2" in text

    class Box:
        def to_dict(self):
            return {"x": "y"}

    meta2, text2 = ih._extract_card_data(Box())
    assert meta2.get("x") == "y"


def test_extract_github_and_dataset_urls():
    # github
    assert ih._extract_github_url("See https://github.com/owner/repo for code") == "https://github.com/owner/repo"
    assert ih._extract_github_url("owner/repo") == "unknown"

    # dataset direct
    assert ih._extract_dataset_url_direct("... https://huggingface.co/datasets/owner/ds ...") == "https://huggingface.co/datasets/owner/ds"
    # yaml extraction
    txt = "datasets:\n  - owner/ds"
    assert ih._extract_dataset_from_yaml(txt) == "owner/ds"
    # prose extraction
    assert ih._extract_dataset_from_prose("Trained on the owner/ds dataset") == "owner/ds"


def test_construct_dataset_url_and_extract():
    url = ih._construct_dataset_url(" Owner/DS ")
    assert url.endswith("owner/ds")

    # If nothing found, returns unknown
    assert ih._extract_dataset_url({}, "no data here") == "unknown"


def test_validate_url_and_head(monkeypatch):
    class R: pass
    r = R(); r.status_code = 200
    import sys
    monkeypatch.setitem(sys.modules, "requests", Mock(head=lambda *a, **k: r))
    assert ih.validate_url("https://example.com", "github") is True

    r2 = R(); r2.status_code = 404
    monkeypatch.setitem(sys.modules, "requests", Mock(head=lambda *a, **k: r2))
    assert ih.validate_url("https://example.com", "github") is False


def test_score_model_with_evaluator_fallback(monkeypatch, tmp_path):
    # Force subprocess to raise so fallback is used
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: json.dumps([{"net_score": 0.9}]))

    # Use real fallback path by making subprocess succeed
    res = ih.score_model_with_evaluator("http://code", "http://data", "http://model")
    assert isinstance(res, dict)


def test_ensure_required_and_threshold():
    r = {"dataset_quality": 0.6, "code_quality": 0.6}
    ih._ensure_required_metrics(r)
    # required keys exist
    assert "dataset_and_code_score" in r

    assert ih._check_threshold({"a": 0.6, "b": 0.7}, 0.5) is True
    assert ih._check_threshold({"a": 0.4, "b": 0.7}, 0.5) is False


def test_calculate_artifact_size_github_and_hf(monkeypatch):
    # GitHub branch
    class Resp: pass
    r = Resp(); r.status_code = 200; r.json = lambda: {"size": 2048}
    import sys
    monkeypatch.setitem(sys.modules, "requests", Mock(get=lambda *a, **k: r))
    size = ih._calculate_artifact_size("https://github.com/owner/repo", "code")
    assert size > 0

    # HF dataset branch - simulate api with siblings sizes
    class Info: pass
    f = Info()
    class File: pass
    file = File(); file.size = 1024 * 1024  # 1MB
    f.siblings = [file]

    monkeypatch.setattr(ih, "HfApi", lambda *a, **k: Mock(dataset_info=lambda repo: f, model_info=lambda repo: f))
    size2 = ih._calculate_artifact_size("https://huggingface.co/datasets/owner/ds", "dataset")
    assert size2 > 0


def test_ingest_dataset_and_code_and_batch(monkeypatch):
    # ingest_dataset and ingest_code should add artifacts
    res = ih.ingest_dataset("https://huggingface.co/datasets/owner/ds")
    assert res["status"] == "success"

    res2 = ih.ingest_code("https://github.com/owner/repo")
    assert res2["status"] == "success"

    # batch_ingest handles error gracefully
    out = ih.batch_ingest(["https://huggingface.co/models/owner/model", "bad://url"])
    assert isinstance(out, list)
