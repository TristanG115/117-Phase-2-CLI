import io
from unittest.mock import Mock, patch

import pytest

from botocore.exceptions import ClientError


def make_client_error(code="InternalError", msg="boom"):
    return ClientError({"Error": {"Code": code, "Message": msg}}, "Op")


def test_s3storage_upload_and_errors(monkeypatch):
    from API.storage import S3Storage, StorageUnavailableError

    # Provide config and avoid real boto3 client creation
    monkeypatch.setattr("API.storage.S3Storage.load_config", lambda self: {"AWS_REGION": "us-west-2", "S3_BUCKET_NAME": "bucket"})

    s = S3Storage()
    # upload_fileobj raises ClientError -> StorageUnavailableError
    s.s3 = Mock()
    s.s3.upload_fileobj.side_effect = make_client_error()
    with pytest.raises(StorageUnavailableError):
        s.upload(io.BytesIO(b"x"), "key")

    # generate_presigned_url raises -> HTTPException
    s.s3.generate_presigned_url.side_effect = make_client_error()
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        s.download_url("k")

    # download_file raises NoSuchKey -> HTTPException 404
    s.s3.download_file.side_effect = make_client_error(code="NoSuchKey")
    with pytest.raises(HTTPException) as ei:
        s.download_to_file("k", "./tmp")
    assert ei.value.status_code in (404, 500)

    # delete_object raises -> HTTPException
    s.s3.delete_object.side_effect = make_client_error()
    with pytest.raises(HTTPException):
        s.delete_file("k")

    # list_objects_v2 raises -> HTTPException
    s.s3.list_objects_v2.side_effect = make_client_error()
    with pytest.raises(HTTPException):
        s.list_files("p")

    # head_object NoSuchKey -> HTTPException 404
    s.s3.head_object.side_effect = make_client_error(code="NoSuchKey")
    with pytest.raises(HTTPException):
        s.get_file_metadata("k")


def test_dynamodb_error_branches(monkeypatch):
    from API.dynamo import DynamoDB

    # Prevent load_config from reading files
    monkeypatch.setattr("API.dynamo.DynamoDB.load_config", lambda self: {"AWS_REGION": "us-west-2", "S3_BUCKET_NAME": "b"})

    # Prevent boto3.resource from creating real resources
    monkeypatch.setattr("API.dynamo.boto3.resource", lambda *a, **k: Mock(Table=lambda name: Mock()))

    d = DynamoDB(table_name="t")
    # Replace table with a controllable mock
    d.table = Mock()

    # add_artifact -> put_item raises -> RuntimeError (and logs error)
    d.table.put_item.side_effect = make_client_error()
    with pytest.raises(RuntimeError):
        d.add_artifact("n", "model")

    # get_artifact_by_id -> get_item raises -> returns None
    d.table.get_item.side_effect = make_client_error()
    assert d.get_artifact_by_id("1") is None

    # update_artifact -> update_item raises -> returns False
    d.table.update_item.side_effect = make_client_error()
    assert d.update_artifact("1", foo=1) is False

    # delete_artifact -> delete_item raises -> returns False
    d.table.delete_item.side_effect = make_client_error()
    assert d.delete_artifact("1") is False

    # list_artifacts -> scan raises -> returns []
    d.table.scan.side_effect = make_client_error()
    assert d.list_artifacts() == []

    # search_artifacts -> make list_artifacts raise Exception -> search returns []
    monkeypatch.setattr(d, "list_artifacts", lambda *a, **k: (_ for _ in ()).throw(Exception("boom")))
    assert d.search_artifacts("q") == []

    # reset_registry -> simulate scan returns items but batch_writer delete raises ClientError -> RuntimeError
    d.table.scan.side_effect = None
    d.table.scan.return_value = {"Items": [{"artifact_id": "1"}]}

    class BadBatch:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def delete_item(self, Key):
            raise make_client_error()

    d.table.batch_writer = lambda: BadBatch()
    with pytest.raises(RuntimeError):
        d.reset_registry()

    # get_registry_stats -> make scan raise ClientError -> returns {'total':0}
    d.table.scan.side_effect = make_client_error()
    assert d.get_registry_stats() == {"total": 0}

    # update_model -> set update_artifact to return False -> raises RuntimeError
    monkeypatch.setattr(d, "update_artifact", lambda *a, **k: False)
    with pytest.raises(RuntimeError):
        d.update_model("1", {"foo": 1})
