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

<<<<<<< HEAD
    class FileResp:
        status_code = 200
        content = b"data"

    monkeypatch.setattr("subprocess.run", fake_run_fail)
    # requests.get: first call is page, then file downloads
    monkeypatch.setattr("requests.get", lambda *a, **k: PageResp() if a[0] == "https://huggingface.co/owner/model" else FileResp())
    assert ad.download_huggingface_artifact("https://huggingface.co/owner/model", "artifact", str(tmp_path)) is True


def test_download_github_repo_zip_fail(monkeypatch):
    # simulate git clone failing and ZIP non-200
    def fake_run_fail(*a, **k):
        raise Exception("git fail")

    monkeypatch.setattr("subprocess.run", fake_run_fail)

    class R: status_code = 404
    monkeypatch.setattr("requests.get", lambda *a, **k: R())

    assert ad.download_github_repo("https://github.com/owner/repo", "artifact", ".") is False


def test_download_and_zip_artifact_full_flow(tmp_path, monkeypatch):
    # Create an artifact dir with files and ensure download_and_zip_artifact zips it
    base_temp = tmp_path / "base"
    base_temp.mkdir()
    artifact_dir = base_temp / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "f1.txt").write_text("hello")

    # Monkeypatch tempfile.mkdtemp to return our base_temp
    monkeypatch.setattr("tempfile.mkdtemp", lambda prefix, dir=None: str(base_temp))
    # Monkeypatch download to be successful
    monkeypatch.setattr(ad, "download_huggingface_artifact", lambda url, name, tdir: True)

    zip_path = ad.download_and_zip_artifact("https://huggingface.co/owner/artifact", "artifact", "dataset")
    assert zip_path is not None
    assert zipfile.is_zipfile(zip_path)


def test_download_huggingface_dataset_variants(tmp_path, monkeypatch):
    # dataset URL with two and three segments
    monkeypatch.setitem(__import__('sys').modules, 'huggingface_hub', Mock(snapshot_download=lambda **k: None))
    assert ad.download_huggingface_artifact("https://huggingface.co/datasets/owner/ds", "artifact", str(tmp_path)) is True
    assert ad.download_huggingface_artifact("https://huggingface.co/datasets/owner/group/ds", "artifact", str(tmp_path)) is True


def test_download_github_clone_success_removes_git(tmp_path, monkeypatch):
    # Simulate successful git clone and presence of .git directory
    def fake_run(cmd, check=False, capture_output=True, timeout=None):
        class R: pass
        r = R(); r.returncode = 0
        return r

    monkeypatch.setattr("subprocess.run", fake_run)
    # Pre-create .git folder to be removed by function
    download_dir = tmp_path / "artifact"
    download_dir.mkdir()
    (download_dir / ".git").mkdir()

    res = ad.download_github_repo("https://github.com/owner/repo", "artifact", str(tmp_path))
    assert res is True
    assert not (download_dir / ".git").exists()


def test_upload_artifact_to_s3_cleanup_on_failure(tmp_path):
    # Create a zip file
    zip_path = tmp_path / "x.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("a.txt", "hello")

    class BadS3:
        def upload(self, fobj, key, metadata=None):
            raise Exception("upload fail")

    res = ad.upload_artifact_to_s3(BadS3(), "artifact", "model", str(zip_path))
    assert res is None
    # file should be removed even on failure
    assert not zip_path.exists()


def test_get_artifact_download_url_failure(monkeypatch):
    class BadS3:
        def download_url(self, key, expiration=3600):
            raise Exception("fail")

    assert ad.get_artifact_download_url(BadS3(), "artifact", "model") is None


def test_process_artifact_download_failure(monkeypatch):
    class FakeS3:
        def file_exists(self, key):
            return False

    monkeypatch.setattr(ad, "download_and_zip_artifact", lambda url, name, t: None)
    key, url = ad.process_artifact_for_s3(FakeS3(), "https://huggingface.co/x/y", "artifact", "dataset")
    assert key is None and url is None


def test_hf_snapshot_raises_exception(tmp_path, monkeypatch):
    def bad_sd(**k):
        raise Exception("snap fail")

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", Mock(snapshot_download=bad_sd))
    assert ad.download_huggingface_artifact("https://huggingface.co/owner/model", "artifact", str(tmp_path)) is False


def test_download_github_zip_download_exception(tmp_path, monkeypatch):
    def fake_run_fail(*a, **k):
        raise Exception("git fail")

    monkeypatch.setattr("subprocess.run", fake_run_fail)
    import requests as _req

    def raise_req(*a, **k):
        raise _req.RequestException("boom")

    monkeypatch.setattr("requests.get", raise_req)
    assert ad.download_github_repo("https://github.com/owner/repo", "artifact", str(tmp_path)) is False


def test_download_and_zip_artifact_mkdtemp_failure(monkeypatch):
    monkeypatch.setattr("tempfile.mkdtemp", lambda *a, **k: (_ for _ in ()).throw(Exception("mkfail")))
    assert ad.download_and_zip_artifact("https://huggingface.co/owner/artifact", "artifact", "dataset") is None


def test_zip_write_failure(tmp_path, monkeypatch):
    base_temp = tmp_path / "base"
    base_temp.mkdir()
    artifact_dir = base_temp / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "f1.txt").write_text("hello")

    monkeypatch.setattr("tempfile.mkdtemp", lambda prefix, dir=None: str(base_temp))
    monkeypatch.setattr(ad, "download_huggingface_artifact", lambda url, name, tdir: True)

    import zipfile as _zip

    orig_write = _zip.ZipFile.write

    def bad_write(self, filename, arcname):
        raise Exception("write fail")

    monkeypatch.setattr(_zip.ZipFile, "write", bad_write)

    # Should handle write failure and still return a zip (may be empty)
    res = ad.download_and_zip_artifact("https://huggingface.co/owner/artifact", "artifact", "dataset")
    assert res is not None
    # restore is automatic due to monkeypatch teardown


def test_process_artifact_for_s3_upload_flow(monkeypatch, tmp_path):
    class FakeS3:
        def file_exists(self, key):
            return False

        def upload(self, f, key, metadata=None):
            return True

        def download_url(self, key, expiration=3600):
            return "https://s3.fake/url"

    s3 = FakeS3()
    z = tmp_path / "artifact.zip"
    with zipfile.ZipFile(z, "w") as zz:
        zz.writestr("a.txt", "hi")

    monkeypatch.setattr(ad, "download_and_zip_artifact", lambda url, name, t: str(z))
    monkeypatch.setattr(ad, "upload_artifact_to_s3", lambda s3, name, t, path: "artifacts/dataset/artifact.zip")
    monkeypatch.setattr(ad, "get_artifact_download_url", lambda s3, name, t: "https://s3.fake/url")

    key, url = ad.process_artifact_for_s3(s3, "https://huggingface.co/owner/artifact", "artifact", "dataset")
    assert key == "artifacts/dataset/artifact.zip"
    assert url == "https://s3.fake/url"

=======
    assert key.endswith("nameX.zip")
    assert url == "http://signed"
>>>>>>> parent of 763fad3 (90% line coverage, need to consolidate tests and improve error message production)
