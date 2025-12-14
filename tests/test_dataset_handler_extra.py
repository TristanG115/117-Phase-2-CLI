from unittest.mock import Mock, patch

import handlers.dataset_handler as dh


def test_extract_dataset_id_variants():
    d1 = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    assert d1.dataset_id == "owner/ds"

    d2 = dh.DatasetHandler("https://huggingface.co/datasets/owner")
    assert d2.dataset_id == "owner"

    d3 = dh.DatasetHandler("https://huggingface.co/owner/ds")
    assert d3.dataset_id == ""


def test_get_huggingface_api_data_list_response():
    handler = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    mock_response = Mock()
    mock_response.status_code = 200
    # Return a list where one item matches expected id
    mock_response.json.return_value = [
        {"id": "other/thing"},
        {"id": "owner/ds", "downloads": 5000},
    ]

    with patch("requests.get", return_value=mock_response):
        data = handler.get_huggingface_api_data()
        assert data.get("downloads") == 5000


def test_has_evaluation_dataset_true_false():
    handler = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"tags": ["Evaluation", "nlp"]}
    with patch("requests.get", return_value=mock_response):
        assert handler.has_evaluation_dataset() is True

    mock_response2 = Mock()
    mock_response2.status_code = 200
    mock_response2.json.return_value = {"tags": ["nlp"]}
    handler2 = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    with patch("requests.get", return_value=mock_response2):
        assert handler2.has_evaluation_dataset() is False


def test_quality_score_and_components():
    handler = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    mock_api_response = Mock()
    mock_api_response.status_code = 200
    mock_api_response.json.return_value = {
        "cardData": {"dataset_info": {"rows": 100}},
        "description": "x" * 250,
        "downloads": 2000,
        "tags": ["a", "b", "c"],
        "siblings": [{}, {}],
    }

    mock_readme = Mock()
    mock_readme.status_code = 200
    mock_readme.text = "x" * 600

    with patch("requests.get", side_effect=[mock_api_response, mock_readme]):
        qscore = handler.get_quality_score()
        assert qscore > 0.5


def test_get_license_score_variants():
    handler = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    # license in root
    r1 = Mock(); r1.status_code = 200; r1.json.return_value = {"license": "mit"}
    with patch("requests.get", return_value=r1):
        assert handler.get_license_score() > 0

    # license in cardData
    r2 = Mock(); r2.status_code = 200; r2.json.return_value = {"cardData": {"license": "apache-2.0"}}
    with patch("requests.get", return_value=r2):
        assert handler.get_license_score() > 0

    # license in tags
    r3 = Mock(); r3.status_code = 200; r3.json.return_value = {"tags": ["license:bsd-3-clause"]}
    with patch("requests.get", return_value=r3):
        assert handler.get_license_score() > 0


def test_contributor_count_thresholds():
    handler = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")

    r = Mock(); r.status_code = 200; r.json.return_value = {"downloads": 150000}
    with patch("requests.get", return_value=r):
        assert handler.get_contributor_count() >= 10


def test_get_hf_dataset_info_stub():
    handler = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    out = handler.get_hf_dataset_info("some_url")
    assert out["status"] == "ok"
