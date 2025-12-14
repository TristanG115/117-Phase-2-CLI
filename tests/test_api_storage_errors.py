
# ==================================================
# BEGIN test_api_dynamo_errors.py
# ==================================================

from botocore.exceptions import ClientError

from API.dynamo import DynamoDB


def make_table_with_raise(method_name, exc):
    class T:
        pass

    t = T()

    def raiser(*a, **k):
        raise exc

    setattr(t, method_name, raiser)
    return t


def test_add_artifact_logs_error(monkeypatch, caplog):
    caplog.set_level("ERROR")
    monkeypatch.setattr(DynamoDB, "load_config", lambda self: {"AWS_REGION": "r", "S3_BUCKET_NAME": "b"})

    exc = ClientError({"Error": {"Code": "Err", "Message": "boom"}}, "PutItem")
    table = make_table_with_raise("put_item", exc)
    class FakeResource:
        def Table(self, name):
            return table
    monkeypatch.setattr("boto3.resource", lambda *a, **k: FakeResource())

    d = DynamoDB()
    try:
        d.add_artifact("x", "model")
    except Exception:
        pass

    assert any("Failed to add artifact" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_get_artifact_by_id_logs_error(monkeypatch, caplog):
    caplog.set_level("ERROR")
    monkeypatch.setattr(DynamoDB, "load_config", lambda self: {"AWS_REGION": "r", "S3_BUCKET_NAME": "b"})

    exc = ClientError({"Error": {"Code": "Err", "Message": "boom"}}, "GetItem")
    table = make_table_with_raise("get_item", exc)
    class FakeResource:
        def Table(self, name):
            return table
    monkeypatch.setattr("boto3.resource", lambda *a, **k: FakeResource())

    d = DynamoDB()
    res = d.get_artifact_by_id("x")
    assert res is None
    assert any("Failed to get artifact" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_list_artifacts_logs_error(monkeypatch, caplog):
    caplog.set_level("ERROR")
    monkeypatch.setattr(DynamoDB, "load_config", lambda self: {"AWS_REGION": "r", "S3_BUCKET_NAME": "b"})

    exc = ClientError({"Error": {"Code": "Err", "Message": "boom"}}, "Scan")
    table = make_table_with_raise("scan", exc)
    class FakeResource:
        def Table(self, name):
            return table
    monkeypatch.setattr("boto3.resource", lambda *a, **k: FakeResource())

    d = DynamoDB()
    res = d.list_artifacts()
    assert res == []
    assert any("Failed to list artifacts" in r.message for r in caplog.records if r.levelname == "ERROR")

# ==================================================
# BEGIN test_api_dynamo_storage_errors.py
# ==================================================

import json
import logging
import pytest

from API import dynamo as dynamo_mod
from API import storage as storage_mod
from fastapi import HTTPException


class _Raise:
    def __init__(self, exc):
        self._exc = exc

    def __getattr__(self, _):
        def f(*a, **k):
            raise self._exc

        return f


def _client_error(code="Error", message="boom"):
    return ClientError({"Error": {"Code": code, "Message": message}}, "op")


def make_dummy_dynamo(table_obj=None):
    # prevent __init__ from running
    orig = dynamo_mod.DynamoDB.__init__
    dynamo_mod.DynamoDB.__init__ = lambda self: None
    db = dynamo_mod.DynamoDB()
    dynamo_mod.DynamoDB.__init__ = orig
    db.table = table_obj
    return db


def make_dummy_s3(s3_obj=None, bucket_name="test-bucket"):
    orig = storage_mod.S3Storage.__init__
    storage_mod.S3Storage.__init__ = lambda self: None
    s = storage_mod.S3Storage()
    storage_mod.S3Storage.__init__ = orig
    s.s3 = s3_obj
    s.bucket_name = bucket_name
    return s


def test_dynamo_add_artifact_client_error():
    exc = _client_error()
    table = _Raise(exc)
    db = make_dummy_dynamo(table)
    with pytest.raises(RuntimeError):
        db.add_artifact("n", "model")


def test_dynamo_get_artifact_client_error_returns_none():
    exc = _client_error()
    table = _Raise(exc)
    db = make_dummy_dynamo(table)
    assert db.get_artifact_by_id("1") is None


def test_dynamo_update_artifact_client_error_returns_false():
    exc = _client_error()
    table = _Raise(exc)
    db = make_dummy_dynamo(table)
    assert db.update_artifact("1", score=1.0) is False


def test_dynamo_delete_artifact_client_error_returns_false():
    exc = _client_error()
    table = _Raise(exc)
    db = make_dummy_dynamo(table)
    assert db.delete_artifact("1") is False


def test_dynamo_list_artifacts_client_error_returns_empty():
    exc = _client_error()
    table = _Raise(exc)
    db = make_dummy_dynamo(table)
    assert db.list_artifacts() == []


def test_dynamo_search_artifacts_logs_on_exception(monkeypatch):
    # Force list_artifacts to raise a general Exception to hit search_artifacts except
    db = make_dummy_dynamo(table_obj=object())
    monkeypatch.setattr(db, "list_artifacts", lambda *a, **k: (_ for _ in ()).throw(Exception("boom")))
    assert db.search_artifacts("x") == []


def test_dynamo_reset_registry_client_error_raises():
    exc = _client_error()
    table = _Raise(exc)
    db = make_dummy_dynamo(table)
    with pytest.raises(RuntimeError):
        db.reset_registry()


def test_dynamo_get_registry_stats_client_error_returns_zero():
    exc = _client_error()
    table = _Raise(exc)
    db = make_dummy_dynamo(table)
    assert db.get_registry_stats() == {"total": 0}


def test_s3_upload_client_error_raises_storage_unavailable():
    exc = _client_error()
    s3 = _Raise(exc)
    s = make_dummy_s3(s3)
    with pytest.raises(storage_mod.StorageUnavailableError):
        s.upload(file_obj=object(), key="k")


def test_s3_generate_presigned_url_client_error_raises_http():
    exc = _client_error()
    s3 = _Raise(exc)
    s = make_dummy_s3(s3)
    with pytest.raises(HTTPException):
        s.download_url("k")


def test_s3_download_to_file_nosuchkey_raises_404():
    exc = _client_error(code="NoSuchKey")
    s3 = _Raise(exc)
    s = make_dummy_s3(s3)
    with pytest.raises(HTTPException) as ei:
        s.download_to_file("k", "tmp")
    assert ei.value.status_code == 404


def test_s3_delete_file_client_error_raises_http():
    exc = _client_error()
    s3 = _Raise(exc)
    s = make_dummy_s3(s3)
    with pytest.raises(HTTPException):
        s.delete_file("k")


def test_s3_list_files_client_error_raises_http():
    exc = _client_error()
    s3 = _Raise(exc)
    s = make_dummy_s3(s3)
    with pytest.raises(HTTPException):
        s.list_files()


def test_s3_get_file_metadata_nosuchkey_raises_404():
    exc = _client_error(code="NoSuchKey")
    s3 = _Raise(exc)
    s = make_dummy_s3(s3)
    with pytest.raises(HTTPException) as ei:
        s.get_file_metadata("k")
    assert ei.value.status_code == 404

# ==================================================
# BEGIN test_api_storage_errors.py
# ==================================================

import botocore

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

    try:
        s.get_file_metadata("k")
    except HTTPException:
        pass
    assert any("Could not get metadata" in r.message for r in caplog.records if r.levelname == "ERROR")