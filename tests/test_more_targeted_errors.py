from unittest.mock import Mock
import pytest
from botocore.exceptions import ClientError


def make_client_error(code="InternalError", msg="boom"):
    return ClientError({"Error": {"Code": code, "Message": msg}}, "Op")


def test_dynamo_all_error_branches(monkeypatch, caplog):
    import API.dynamo as dynamo_mod

    # Patch load_config on class
    monkeypatch.setattr(dynamo_mod.DynamoDB, "load_config", lambda self: {"AWS_REGION": "r", "S3_BUCKET_NAME": "b"})

    # Create mock table with methods that raise ClientError for each operation
    table = Mock()
    table.put_item.side_effect = make_client_error()
    table.get_item.side_effect = make_client_error()
    table.update_item.side_effect = make_client_error()
    table.delete_item.side_effect = make_client_error()
    table.scan.side_effect = make_client_error()
    table.query.side_effect = make_client_error()

    # Ensure module uses our mocked boto3.resource
    class FakeResource:
        def Table(self, name):
            return table

    monkeypatch.setattr(dynamo_mod, "boto3", Mock(resource=lambda *a, **k: FakeResource()))

    d = dynamo_mod.DynamoDB(table_name="t")
    d.table = table

    # add_artifact -> raises RuntimeError
    with pytest.raises(RuntimeError):
        d.add_artifact("n", "model")
    # get_artifact_by_id -> returns None and logs
    assert d.get_artifact_by_id("1") is None
    # update -> False
    assert d.update_artifact("1", foo=1) is False
    # delete -> False
    assert d.delete_artifact("1") is False
    # list_artifacts -> []
    assert d.list_artifacts() == []
    # search_artifacts -> [] (query uses list_artifacts which will raise)
    monkeypatch.setattr(d, "list_artifacts", lambda *a, **k: (_ for _ in ()).throw(Exception("boom")))
    assert d.search_artifacts("q") == []


def test_storage_all_error_branches(monkeypatch):
    import API.storage as storage_mod

    # Patch load_config
    monkeypatch.setattr(storage_mod.S3Storage, "load_config", lambda self: {"AWS_REGION": "r", "S3_BUCKET_NAME": "b"})

    s = storage_mod.S3Storage()
    # Mock s3 client methods to raise ClientError
    s.s3 = Mock()
    s.s3.upload_fileobj.side_effect = make_client_error()
    s.s3.generate_presigned_url.side_effect = make_client_error()
    s.s3.download_file.side_effect = make_client_error()
    s.s3.delete_object.side_effect = make_client_error()
    s.s3.list_objects_v2.side_effect = make_client_error()
    s.s3.head_object.side_effect = make_client_error()

    from fastapi import HTTPException

    with pytest.raises(storage_mod.StorageUnavailableError):
        s.upload(Mock(), "k")

    with pytest.raises(HTTPException):
        s.download_url("k")

    with pytest.raises(HTTPException):
        s.download_to_file("k", "./t")

    with pytest.raises(HTTPException):
        s.delete_file("k")

    with pytest.raises(HTTPException):
        s.list_files("p")

    with pytest.raises(HTTPException):
        s.get_file_metadata("k")
