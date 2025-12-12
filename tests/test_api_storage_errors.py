import botocore
from botocore.exceptions import ClientError

from API.storage import S3Storage


def make_client(exc_on=None):
    class C:
        def __init__(self):
            pass

    c = C()

    def raise_client_error(*a, **k):
        raise ClientError({"Error": {"Code": exc_on or "Error", "Message": "boom"}}, "Op")

    c.upload_fileobj = raise_client_error
    c.generate_presigned_url = lambda *a, **k: (_ for _ in ()).throw(ClientError({"Error": {"Code": "Err", "Message": "boom"}}, "Op"))
    c.download_file = raise_client_error
    c.delete_object = raise_client_error
    c.list_objects_v2 = raise_client_error
    c.head_object = raise_client_error
    return c


def test_s3_upload_logs_error(monkeypatch, caplog):
    caplog.set_level("ERROR")
    monkeypatch.setattr(S3Storage, "load_config", lambda self: {"AWS_REGION": "r", "S3_BUCKET_NAME": "b"})
    monkeypatch.setattr("boto3.client", lambda *a, **k: make_client())

    s = S3Storage()
    try:
        s.upload(None, "k")
    except Exception:
        pass
    assert any("S3 upload failed" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_s3_generate_presigned_url_logs_error(monkeypatch, caplog):
    caplog.set_level("ERROR")
    monkeypatch.setattr(S3Storage, "load_config", lambda self: {"AWS_REGION": "r", "S3_BUCKET_NAME": "b"})
    monkeypatch.setattr("boto3.client", lambda *a, **k: make_client())

    s = S3Storage()
    try:
        s.download_url("k")
    except Exception:
        pass
    assert any("Could not generate download URL" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_s3_download_to_file_nosuchkey_logs_error(monkeypatch, caplog):
    caplog.set_level("ERROR")
    monkeypatch.setattr(S3Storage, "load_config", lambda self: {"AWS_REGION": "r", "S3_BUCKET_NAME": "b"})

    def make_client_nosk():
        class C:
            pass

        c = C()

        def dl(*a, **k):
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "not found"}}, "Op")

        c.download_file = dl
        return c

    monkeypatch.setattr("boto3.client", lambda *a, **k: make_client_nosk())
    s = S3Storage()
    from fastapi import HTTPException

    try:
        s.download_to_file("k", "/tmp/x")
    except HTTPException:
        pass
    assert any("Download failed for" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_get_file_metadata_nosuchkey_logs_error(monkeypatch, caplog):
    caplog.set_level("ERROR")
    monkeypatch.setattr(S3Storage, "load_config", lambda self: {"AWS_REGION": "r", "S3_BUCKET_NAME": "b"})

    def head(*a, **k):
        raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "not found"}}, "Op")

    monkeypatch.setattr("boto3.client", lambda *a, **k: type("C", (), {"head_object": head}))
    s = S3Storage()
    from fastapi import HTTPException

    try:
        s.get_file_metadata("k")
    except HTTPException:
        pass
    assert any("Could not get metadata" in r.message for r in caplog.records if r.levelname == "ERROR")
