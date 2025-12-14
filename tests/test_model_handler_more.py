import json

import pytest

import handlers.model_handler as mh


class FakeResp:
    def __init__(self, status=200, data=None, text="", headers=None):
        self.status_code = status
        self._data = data
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("bad")


def test_extract_model_id_and_readme_and_urls(monkeypatch):
    m = mh.ModelHandler("https://huggingface.co/owner/model")
    assert m.model_id == "owner/model"

    # readme extraction
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, None, text="See https://github.com/owner/repo and https://huggingface.co/datasets/owner/ds"))
    assert "https://github.com/owner/repo" in m.extract_github_urls_from_readme()
    assert "https://huggingface.co/datasets/owner/ds" in m.extract_dataset_urls_from_readme()


def test_get_hf_api_data_list_and_errors(monkeypatch):
    m = mh.ModelHandler("https://huggingface.co/owner/model")

    # list with exact match
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, [{"id": "owner/model", "downloads": 1}]))
    d = m.get_huggingface_api_data()
    assert d.get("downloads") == 1

    # empty list -> returns {}
    m2 = mh.ModelHandler("https://huggingface.co/owner/model")
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, []))
    assert m2.get_huggingface_api_data() == {}

    # unexpected type
    m3 = mh.ModelHandler("https://huggingface.co/owner/model")
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, "not-a-dict"))
    assert m3.get_huggingface_api_data() == {}


def test_get_model_files_and_resolve_size_and_get_size_mb(monkeypatch, tmp_path):
    m = mh.ModelHandler("https://huggingface.co/owner/model")

    # files listing
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, [{"path": "a", "size": 1024}, {"path": "b", "size": 2048}]))
    files = m.get_model_files()
    assert isinstance(files, list)

    # HEAD probe finds content-length
    monkeypatch.setattr("requests.head", lambda *a, **k: type("R", (), {"status_code": 200, "headers": {"Content-Length": "3145728"}})())
    sz = m._resolve_file_size_via_head("pytorch_model.bin")
    assert sz > 0

    # get_size_mb with safetensors
    m._cache_set("hf_api_data", {"safetensors": {"total": 10 * 1024 * 1024}})
    assert m.get_size_mb() >= 10


def test_has_performance_benchmarks_and_docs(monkeypatch):
    m = mh.ModelHandler("https://huggingface.co/owner/model")
    m._cache_set("hf_api_data", {"cardData": {"model-index": True}})
    assert m.has_performance_benchmarks() is True

    # readme keywords
    m2 = mh.ModelHandler("https://huggingface.co/owner/model")
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, None, text="This benchmark achieved high accuracy and has usage examples"))
    assert m2.has_performance_benchmarks() is True


def test_license_and_doc_and_contributors(monkeypatch):
    m = mh.ModelHandler("https://huggingface.co/owner/model")
    # license via api
    m._cache_set("hf_api_data", {"license": "MIT"})
    assert m.get_license_score() > 0

    # license via README YAML
    m2 = mh.ModelHandler("https://huggingface.co/owner/model")
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, None, text="---\nlicense: GPL-3.0\n---\n"))
    assert m2.get_license_score() >= 0

    # documentation score
    m3 = mh.ModelHandler("https://huggingface.co/owner/model")
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, None, text="Usage example: how to use this model for training and evaluation"))
    assert m3.get_documentation_score() > 0

    # contributors heuristic
    m4 = mh.ModelHandler("https://huggingface.co/owner/model")
    m4._cache_set("hf_api_data", {"downloads": 20000, "likes": 0})
    assert m4.get_contributor_count() >= 10


def test_get_hf_model_info_errors(monkeypatch):
    m = mh.ModelHandler("https://huggingface.co/owner/model")
    # success
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, {"a": 1}))
    assert isinstance(m.get_hf_model_info(), dict)

    # failure should raise (use a fresh handler to avoid cache)
    m2 = mh.ModelHandler("https://huggingface.co/owner/model")
    monkeypatch.setattr("requests.get", lambda *a, **k: (_ for _ in ()).throw(Exception("nope")))
    with pytest.raises(Exception):
        m2.get_hf_model_info()
from unittest.mock import Mock, patch
import handlers.model_handler as mh


def test_get_model_files_success_and_failure():
    m = mh.ModelHandler("https://huggingface.co/owner/model")
    mock_resp = Mock(); mock_resp.status_code = 200; mock_resp.json.return_value = [{"size": 1024}]
    with patch("handlers.model_handler.requests.get", return_value=mock_resp):
        files = m.get_model_files()
        assert isinstance(files, list)

    # Exception path
    def bad_get(*a, **k):
        raise Exception("boom")

    m_new = mh.ModelHandler("https://huggingface.co/owner/model-new")
    with patch("handlers.model_handler.requests.get", side_effect=bad_get):
        assert m_new.get_model_files() == []


def test_resolve_file_size_and_get_size_mb_branches(monkeypatch):
    m = mh.ModelHandler("https://huggingface.co/owner/model")

    # safetensors path
    monkeypatch.setattr(m, "get_huggingface_api_data", lambda: {"safetensors": {"total": 2 * 1024 * 1024}})
    assert m.get_size_mb() >= 2.0

    # tree sizes path
    m2 = mh.ModelHandler("https://huggingface.co/owner/model2")
    monkeypatch.setattr(m2, "get_huggingface_api_data", lambda: {})
    monkeypatch.setattr(m2, "get_model_files", lambda: [{"size": 1024 * 1024}, {"size": "2048"}])
    assert m2.get_size_mb() > 0

    # siblings rsize path
    m3 = mh.ModelHandler("https://huggingface.co/owner/model3")
    monkeypatch.setattr(m3, "get_huggingface_api_data", lambda: {"siblings": [{"rsize": 1024}, {"size": 1024}]})
    assert m3.get_size_mb() >= 0

    # HEAD probe path
    m4 = mh.ModelHandler("https://huggingface.co/owner/model4")
    monkeypatch.setattr(m4, "get_huggingface_api_data", lambda: {})
    monkeypatch.setattr(m4, "get_model_files", lambda: [])
    monkeypatch.setattr(m4, "_resolve_file_size_via_head", lambda fname: 1024 * 1024)
    assert m4.get_size_mb() > 0


def test_has_benchmarks_and_license_parsing():
    m = mh.ModelHandler("https://huggingface.co/owner/model")
    # cardData model-index
    monkeypatch1 = patch("handlers.model_handler.ModelHandler.get_huggingface_api_data", return_value={"cardData": {"model-index": [1]}})
    with monkeypatch1:
        assert m.has_performance_benchmarks() is True

    # README benchmark keywords
    m2 = mh.ModelHandler("https://huggingface.co/owner/model2")
    monkeypatch2 = patch("handlers.model_handler.ModelHandler.get_readme_content", return_value="This shows benchmark results and accuracy")
    with monkeypatch2:
        assert m2.has_performance_benchmarks() is True

    # License in YAML frontmatter
    m3 = mh.ModelHandler("https://huggingface.co/owner/model3")
    monkeypatch3 = patch("handlers.model_handler.ModelHandler.get_readme_content", return_value="---\nlicense: mit\n---\n rest")
    with monkeypatch3:
        assert m3.get_license_score() == 1.0


def test_get_huggingface_api_data_list_errors():
    m = mh.ModelHandler("https://huggingface.co/owner/model")
    # empty list
    resp = Mock(); resp.status_code = 200; resp.json.return_value = []
    with patch("handlers.model_handler.requests.get", return_value=resp):
        assert m.get_huggingface_api_data() == {}

    # list with non-dict first item
    resp2 = Mock(); resp2.status_code = 200; resp2.json.return_value = ["nope", 123]
    m2 = mh.ModelHandler("https://huggingface.co/owner/model2")
    with patch("handlers.model_handler.requests.get", return_value=resp2):
        assert m2.get_huggingface_api_data() == {}
