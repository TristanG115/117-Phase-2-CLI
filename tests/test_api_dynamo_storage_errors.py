import json
import logging
from botocore.exceptions import ClientError
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
