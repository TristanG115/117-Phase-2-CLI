import os
import io
import tempfile
import zipfile
from unittest.mock import Mock, patch

import artifact_downloader as ad


def test_get_s3_key_for_artifact():
    assert ad.get_s3_key_for_artifact("name/with/slash", "model") == "artifacts/model/name_with_slash.zip"


def test_upload_and_check_and_download_url(tmp_path):
    # fake s3 storage
    class FakeS3:
        def __init__(self):
            self.data = {}

        def upload(self, fobj, key, metadata=None):
            self.data[key] = fobj.read()
            return True

        def file_exists(self, key):
            return key in self.data

        def download_url(self, key, expiration=3600):
            return f"https://s3.fake/{key}"

    s3 = FakeS3()

    # create small zip file on disk
    zip_path = tmp_path / "x.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("a.txt", "hello")

    s3_key = ad.upload_artifact_to_s3(s3, "artifact", "model", str(zip_path))
    assert s3_key == "artifacts/model/artifact.zip"
    assert ad.check_artifact_exists_in_s3(s3, "artifact", "model") is True
    assert ad.get_artifact_download_url(s3, "artifact", "model") == "https://s3.fake/artifacts/model/artifact.zip"


def test_upload_artifact_to_s3_failure(tmp_path):
    class BadS3:
        def upload(self, fobj, key, metadata=None):
            raise Exception("nope")

    zip_path = tmp_path / "x.zip"
    with open(zip_path, "wb") as f:
        f.write(b"x")

    assert ad.upload_artifact_to_s3(BadS3(), "artifact", "model", str(zip_path)) is None


def test_process_artifact_cache_hit(monkeypatch):
    class FakeS3:
        def file_exists(self, key):
            return True

        def download_url(self, key, expiration=3600):
            return "https://s3.fake/url"

    s3 = FakeS3()
    key, url = ad.process_artifact_for_s3(s3, "https://huggingface.co/datasets/x/y", "artifact", "dataset")
    assert key == "artifacts/dataset/artifact.zip"
    assert url == "https://s3.fake/url"


def test_download_and_zip_artifact_unsupported_and_errors(monkeypatch, tmp_path):
    # Unsupported URL
    assert ad.download_and_zip_artifact("ftp://example", "a", "other") is None

    # Simulate HTTP path but response non-200
    class R: pass
    r = R(); r.status_code = 404; r.text = ""
    monkeypatch.setattr("requests.get", lambda *a, **k: r)
    # Use tmp base dir to avoid using /home
    monkeypatch.setenv("TMP", str(tmp_path))
    assert ad.download_huggingface_artifact("https://huggingface.co/x/y", "artifact", str(tmp_path)) is False


def test_download_github_repo_zip_success(tmp_path, monkeypatch):
    # Simulate git clone failing and ZIP download working
    def fake_get(url, timeout=60, stream=False):
        class R:
            status_code = 200

            def iter_content(self, chunk_size=8192):
                # Create in-memory zip bytes
                bio = io.BytesIO()
                with zipfile.ZipFile(bio, "w") as z:
                    z.writestr("file.txt", "hi")
                bio.seek(0)
                yield bio.read()

        return R()

    monkeypatch.setattr("requests.get", fake_get)
    res = ad.download_github_repo("https://github.com/owner/repo", "artifact", str(tmp_path))
    assert res is True


def test_download_huggingface_snapshot_and_git_and_http(tmp_path, monkeypatch):
    # snapshot_download path
    monkeypatch.setitem(__import__('sys').modules, 'huggingface_hub', Mock(snapshot_download=lambda **k: None))
    assert ad.download_huggingface_artifact("https://huggingface.co/owner/model", "artifact", str(tmp_path)) is True

    # simulate huggingface_hub ImportError -> git clone success
    # Ensure import of huggingface_hub fails by replacing module with an empty mock
    monkeypatch.setitem(__import__('sys').modules, 'huggingface_hub', Mock())
    def fake_run(cmd, check=False, capture_output=True, timeout=None):
        class R: pass
        r = R(); r.returncode = 0
        return r

    monkeypatch.setattr("subprocess.run", fake_run)
    assert ad.download_huggingface_artifact("https://huggingface.co/owner/model", "artifact", str(tmp_path)) is True

    # simulate git clone failure and HTTP download success
    def fake_run_fail(*a, **k):
        raise Exception("git fail")

    class PageResp:
        status_code = 200
        text = "<html></html>"

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

