from unittest.mock import Mock, patch

import pytest

import handlers.dataset_handler as dh


class FakeResp:
    def __init__(self, status=200, data=None, text=""):
        self.status_code = status
        self._data = data
        self.text = text

    def json(self):
        return self._data


def make_api_response(**kwargs):
    r = Mock()
    r.status_code = 200
    r.json.return_value = kwargs
    return r


def test_extract_dataset_id_variants():
    d1 = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    assert d1.dataset_id == "owner/ds"

    d2 = dh.DatasetHandler("https://huggingface.co/datasets/owner")
    assert d2.dataset_id == "owner"

    d3 = dh.DatasetHandler("https://huggingface.co/notdatasets/owner")
    assert d3.dataset_id == ""


def test_get_hf_api_data_list_response(monkeypatch):
    ds = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    # Return list with matching id
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, [{"id": "owner/ds", "downloads": 5}]))
    data = ds.get_huggingface_api_data()
    assert data.get("downloads") == 5

    # Return list with dict but no id match -> should pick first dict
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, [{"foo": 1}, {"bar": 2}]))
    data2 = ds.get_huggingface_api_data()
    assert isinstance(data2, dict)


def test_get_readme_content_and_error(monkeypatch):
    ds = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    # Successful readme
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, None, text="hello"))
    assert ds.get_readme_content() == "hello"

    # Exception path
    def raise_req(*a, **k):
        raise Exception("boom")

    monkeypatch.setattr("requests.get", raise_req)
    ds2 = dh.DatasetHandler("https://huggingface.co/datasets/x/y")
    assert ds2.get_readme_content() == ""


def test_has_evaluation_dataset():
    ds = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    ds._cache_set("hf_api_data", {"tags": ["Evaluation", "other"]})
    assert ds.has_evaluation_dataset() is True


def test_quality_and_docs_and_license_and_contributors(monkeypatch):
    ds = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    api = {
        "cardData": {"dataset_info": True, "license": "MIT"},
        "description": "x" * 250,
        "downloads": 2000,
        "tags": ["a", "b", "license:Apache-2.0"],
        "siblings": [{}, {}],
        "license": None,
    }
    ds._cache_set("hf_api_data", api)
    # readme short
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(200, None, text="readme"))

    q = ds.get_quality_score()
    assert q > 0.5

    d = ds.get_documentation_score()
    assert d > 0.0

    lic = ds.get_license_score()
    assert lic > 0.0

    contrib = ds.get_contributor_count()
    assert contrib >= 2

    assert ds.get_tags() == api["tags"]
    assert ds.get_downloads() == api["downloads"]
    assert ds.get_description() == api["description"]
    assert ds.get_siblings() == api["siblings"]


def test_quality_component_scoring_and_bounds():
    h = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    api_resp = make_api_response(
        cardData={"dataset_info": {"rows": 10}},
        description="x" * 120,
        downloads=15000,
        tags=[1, 2, 3, 4, 5, 6],
        siblings=[{}, {}, {}, {}, {}, {}],
    )
    readme_resp = Mock(); readme_resp.status_code = 200; readme_resp.text = "x" * 1200

    with patch("requests.get", side_effect=[api_resp, readme_resp]):
        q = h.get_quality_score()
        assert q > 0.5

    # Description length scoring
    api_resp2 = make_api_response(description="x" * 60)
    with patch("requests.get", return_value=api_resp2):
        assert h._quality_description_score(api_resp2.json.return_value) == 0.05

    # Downloads thresholds
    assert h._quality_downloads_score({"downloads": 20000}) == 0.2
    assert h._quality_downloads_score({"downloads": 2000}) == 0.15
    assert h._quality_downloads_score({"downloads": 200}) == 0.1
    assert h._quality_downloads_score({"downloads": 20}) == 0.05

    # Tags & siblings
    assert h._quality_tags_score({"tags": [1, 2, 3, 4, 5, 6]}) == 0.15
    assert h._quality_tags_score({"tags": [1, 2]}) == 0.1
    assert h._quality_siblings_score({"siblings": [{}, {}]}) == 0.1


def test_documentation_score_readme_thresholds():
    h = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    # Small readme
    with patch("handlers.dataset_handler.DatasetHandler.get_readme_content", return_value="x" * 50):
        assert h._doc_readme_score("x" * 50) == 0.0

    # Medium readme
    with patch("handlers.dataset_handler.DatasetHandler.get_readme_content", return_value="x" * 600):
        assert h._doc_readme_score("x" * 600) == 0.2

    # Large readme
    with patch("handlers.dataset_handler.DatasetHandler.get_readme_content", return_value="x" * 1500):
        assert h._doc_readme_score("x" * 1500) == 0.3


def test_license_and_contributor_helpers():
    h = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    # License via tag
    api = make_api_response(tags=["license:mit"])
    with patch("requests.get", return_value=api):
        assert h.get_license_score() == 1.0

    # No license found (use fresh handler to avoid cache)
    h2 = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    api2 = make_api_response()
    with patch("requests.get", return_value=api2):
        assert h2.get_license_score() == 0.0

    # Contributor thresholds
    assert h.get_contributor_count() == 1
    assert h.get_contributor_count() == 1


def test_readme_fetch_exception_handling():
    h = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    def bad_get(*a, **k):
        raise Exception("boom")

    with patch("requests.get", side_effect=bad_get):
        assert h.get_readme_content() == ""


def test_cached_readme_and_api_cache():
    h = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    h._readme_content = "cached"
    # requests.get should not be called
    with patch("requests.get", side_effect=Exception("should not call")):
        assert h.get_readme_content() == "cached"

    h2 = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds2")
    h2._cache_set("hf_api_data", {"downloads": 42})
    assert h2.get_huggingface_api_data()["downloads"] == 42


def test_documentation_card_and_tag_scores():
    h = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    assert h._doc_card_data_score({"cardData": {"x": 1}}) == 0.2
    assert h._doc_description_score({"description": "x" * 250}) == 0.2
    assert h._doc_tags_score({"tags": [1]}) == 0.05
    assert h._doc_structured_info_score({"cardData": {"dataset_info": {"rows": 1}}}) == 0.15
