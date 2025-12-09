import os
import tempfile
import zipfile
from unittest.mock import patch, MagicMock

import artifact_downloader as ad


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def touch_zip(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("file.txt", "hello")


# ----------------------------------------------------------------------
# SIMPLE HELPERS
# ----------------------------------------------------------------------

def test_get_s3_key_for_artifact():
    assert ad.get_s3_key_for_artifact("my/model", "dataset") == \
           "artifacts/dataset/my_model.zip"


def test_check_artifact_exists():
    s3 = MagicMock()
    s3.file_exists.return_value = True
    assert ad.check_artifact_exists_in_s3(s3, "x", "y") is True


def test_download_url():
    s3 = MagicMock()
    s3.download_url.return_value = "http://url"
    assert ad.get_artifact_download_url(s3, "x", "y") == "http://url"


# ----------------------------------------------------------------------
# HF DOWNLOAD (HTTP fallback)
# ----------------------------------------------------------------------

@patch("artifact_downloader.requests.get")
@patch("artifact_downloader.subprocess.run")
def test_download_hf_fallback(mock_run, mock_http):
    mock_run.side_effect = FileNotFoundError()  # disables git-lfs

    mock_http.return_value.status_code = 200
    mock_http.return_value.text = "<html></html>"
    mock_http.return_value.content = b"data"

    with tempfile.TemporaryDirectory() as td:
        ok = ad.download_huggingface_artifact(
            "https://huggingface.co/models/x", "artifact", td
        )
        assert ok
        assert os.path.exists(os.path.join(td, "artifact", "index.html"))


# ----------------------------------------------------------------------
# GITHUB DOWNLOAD (ZIP fallback)
# ----------------------------------------------------------------------

@patch("artifact_downloader.requests.get")
@patch("artifact_downloader.subprocess.run")
def test_download_github_zip(mock_run, mock_http):
    mock_run.side_effect = FileNotFoundError()

    def fake_resp(*args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.iter_content.return_value = [b"123"]
        return resp

    mock_http.side_effect = fake_resp

    with tempfile.TemporaryDirectory() as td:
        ok = ad.download_github_repo("https://github.com/a/b", "artifact", td)
        assert ok


# ----------------------------------------------------------------------
# ZIP CREATION
# ----------------------------------------------------------------------

@patch("artifact_downloader.download_huggingface_artifact", return_value=True)
def test_download_and_zip(mock_dl):
    with tempfile.TemporaryDirectory() as tmp:
        # patch mkdtemp → our own temp folder
        with patch("artifact_downloader.tempfile.mkdtemp") as mk:
            work = os.path.join(tmp, "w")
            os.makedirs(work, exist_ok=True)

            # create expected artifact folder
            art = os.path.join(work, "modelX")
            os.makedirs(art, exist_ok=True)
            with open(os.path.join(art, "x.txt"), "w") as f:
                f.write("hi")

            mk.return_value = work

            z = ad.download_and_zip_artifact(
                "https://huggingface.co/models/x", "modelX", "model"
            )
            assert os.path.exists(z)
            assert zipfile.is_zipfile(z)


# ----------------------------------------------------------------------
# UPLOAD
# ----------------------------------------------------------------------

def test_upload_artifact():
    s3 = MagicMock()
    s3.upload.return_value = True

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"abc")
        path = f.name

    key = ad.upload_artifact_to_s3(s3, "abc", "model", path)
    assert key.endswith("abc.zip")
    s3.upload.assert_called()


# ----------------------------------------------------------------------
# PROCESS FLOW
# ----------------------------------------------------------------------

def test_process_artifact_for_s3():
    s3 = MagicMock()
    s3.file_exists.return_value = False
    s3.upload.return_value = True
    s3.download_url.return_value = "http://signed"

    with tempfile.TemporaryDirectory() as td:
        fake_zip = os.path.join(td, "fake.zip")
        touch_zip(fake_zip)

        with patch("artifact_downloader.download_and_zip_artifact", return_value=fake_zip):
            key, url = ad.process_artifact_for_s3(
                s3, "https://huggingface.co/x", "nameX", "model"
            )

    assert key.endswith("nameX.zip")
    assert url == "http://signed"
