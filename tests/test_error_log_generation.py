import artifact_downloader as ad
import handlers.registry_handler as rh
import handlers.model_handler as mh
import handlers.dataset_handler as dh
from unittest.mock import Mock, patch


def test_generate_many_error_logs(caplog):
    caplog.set_level("ERROR")

    # trigger registry error via Dynamo exception
    class FakeD:
        def __init__(self):
            raise Exception("boom")

    # force Dynamo available path
    patch1 = patch.object(rh, "DYNAMO_AVAILABLE", True)
    patch2 = patch.object(rh, "DynamoDB", FakeD)
    with patch1, patch2:
        for _ in range(3):
            rh._get_db()

    # Trigger dataset_handler HF API list no dict
    d = dh.DatasetHandler("https://huggingface.co/datasets/owner/ds")
    resp = Mock(); resp.status_code = 200; resp.json.return_value = [1,2,3]
    with patch("requests.get", return_value=resp):
        d.get_huggingface_api_data()

    # Trigger model handler readme error
    m = mh.ModelHandler("https://huggingface.co/owner/model")
    bad = Mock(); bad.status_code = 200; bad.text = 123  # non-str len will raise
    with patch("requests.get", return_value=bad):
        m.get_readme_content()

    # Artifact downloader upload failure
    class BadS3:
        def upload(self, f, key, metadata=None):
            raise Exception("upload fail")

        def file_exists(self, key):
            raise Exception("file check fail")

        def download_url(self, key, expiration=3600):
            raise Exception("download fail")

    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(b"x")
    tmp.flush()
    ad.upload_artifact_to_s3(BadS3(), "a", "model", tmp.name)

    # Unsupported download path will log an ERROR
    ad.download_and_zip_artifact("ftp://example", "a", "other")

    # Make requests.get raise to trigger HF downloader outer exception path
    with patch("requests.get", side_effect=Exception("boom")):
        ad.download_huggingface_artifact("https://huggingface.co/x/y", "a", "/tmp")

    # Trigger registry init failure logging
    orig_get_db = rh._get_db
    def bad_get_db():
        raise Exception("db bad")
    rh._get_db = bad_get_db
    try:
        try:
            rh.init_registry()
        except Exception:
            pass
    finally:
        rh._get_db = orig_get_db

    # Test get_artifact_download_url error
    bad_s3_2 = BadS3()
    ad.get_artifact_download_url(bad_s3_2, "a", "model")

    # Test model handler API json raise
    bad_resp = Mock(); bad_resp.status_code = 200; bad_resp.json.side_effect = Exception("bad json")
    with patch("requests.get", return_value=bad_resp):
        mh.ModelHandler("https://huggingface.co/owner/m").get_huggingface_api_data()

    # Finally assert several error records logged
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) >= 6
