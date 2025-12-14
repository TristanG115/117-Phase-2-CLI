
# ==================================================
# BEGIN test_cover_remaining_errors.py
# ==================================================

import pytest


def test_cover_remaining_disabled():
    pytest.skip("Merged into tests/test_error_coverage_all.py")

# ==================================================
# BEGIN test_error_coverage_all.py
# ==================================================

import io
import json
import asyncio
import os
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

import server
import API.storage as storage_mod
import API.dynamo as dynamo_mod
import handlers.registry_handler as rh
import artifact_downloader as ad

client = TestClient(server.app)


def test_startup_event_logs_error(monkeypatch, caplog):
    caplog.set_level("ERROR")

    def bad_init():
        raise Exception("boom")

    monkeypatch.setattr(rh, "init_registry", bad_init)

    with pytest.raises(Exception):
        asyncio.run(server.startup_event())

    assert any("Failed to initialize registry" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_rate_request_not_found_logs_all(monkeypatch, caplog):
    caplog.set_level("ERROR")

    monkeypatch.setattr(rh, "list_artifacts", lambda: [])

    r = client.get("/artifact/model/9999999999/rate")
    assert r.status_code == 404
    assert any("Artifact '" in r.message for r in caplog.records if r.levelname == "ERROR")
    assert any("TEST FAILURE" in r.message for r in caplog.records if r.levelname == "ERROR")
    assert any("Total artifacts in registry" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_get_artifact_not_found_logs(monkeypatch, caplog):
    caplog.set_level("ERROR")
    monkeypatch.setattr(rh, "list_artifacts", lambda: [])

    r = client.get("/artifacts/model/12345")
    assert r.status_code == 404
    assert any("NOT FOUND: ID" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_lineage_json_decode_error(monkeypatch, caplog):
    caplog.set_level("ERROR")
    name = "artifact-for-lineage"
    aid = server.gen_id(name)
    bad_art = {"name": name, "metadata_json": "not-json"}
    monkeypatch.setattr(rh, "list_artifacts", lambda: [bad_art])

    r = client.get(f"/artifact/model/{aid}/lineage")
    assert r.status_code == 400
    assert any("JSON decode error" in r.message or "LINEAGE" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_s3_processing_error_on_register(monkeypatch, caplog):
    caplog.set_level("ERROR")

    monkeypatch.setattr(server, "S3_AVAILABLE", True)
    monkeypatch.setattr(server, "s3_storage", object())

    def bad_process(*a, **k):
        raise Exception("s3boom")

    monkeypatch.setattr(server, "process_artifact_for_s3", bad_process)

    client.post("/artifact/model", json={"name": "n", "url": "https://example.com/x"})
    assert any("[S3] Error processing" in rec.message for rec in caplog.records if rec.levelname == "ERROR")


def test_get_artifact_invalid_id_format_logs(monkeypatch, caplog):
    caplog.set_level("ERROR")
    monkeypatch.setattr(rh, "list_artifacts", lambda: [])

    r = client.get("/artifacts/model/notanumber")
    assert r.status_code == 404
    assert any("Invalid artifact ID format" in rec.message for rec in caplog.records if rec.levelname == "ERROR")


def test_packages_failure_logs(monkeypatch, caplog):
    caplog.set_level("ERROR")
    monkeypatch.setattr(rh, "list_artifacts", lambda: (_ for _ in ()).throw(Exception("boom")))

    r = client.get("/packages")
    assert r.status_code == 500
    assert any("/packages failed" in rec.message for rec in caplog.records if rec.levelname == "ERROR")


def test_index_load_error_logs(monkeypatch, caplog):
    caplog.set_level("ERROR")
    monkeypatch.setattr(rh, "list_artifacts", lambda: (_ for _ in ()).throw(Exception("boom")))

    r = client.get("/")
    assert r.status_code == 200
    assert any("Error loading dashboard" in rec.message for rec in caplog.records if rec.levelname == "ERROR")


def test_artifacts_json_parse_error_logs(caplog):
    caplog.set_level("ERROR")
    r = client.post("/artifacts", data="not-json")
    assert r.status_code == 400
    assert any("JSON parse error" in rec.message for rec in caplog.records if rec.levelname == "ERROR")


def test_rate_request_missing_artifact_logs_multiple_messages(monkeypatch, caplog):
    caplog.set_level("ERROR")
    monkeypatch.setattr(rh, "list_artifacts", lambda: [])

    r = client.get("/artifact/model/9999/rate")
    assert r.status_code == 404
    assert any("TEST FAILURE" in rec.message for rec in caplog.records if rec.levelname == "ERROR")
    assert any("Total artifacts in registry" in rec.message for rec in caplog.records if rec.levelname == "ERROR")


def make_client_error(code="Err", msg="boom"):
    return ClientError({"Error": {"Code": code, "Message": msg}}, "op")


def test_storage_upload_and_metadata_errors(monkeypatch, caplog, tmp_path):
    caplog.set_level("ERROR")
    s = storage_mod.S3Storage()

    def raise_upload(*a, **k):
        raise make_client_error()

    monkeypatch.setattr(s.s3, "upload_fileobj", raise_upload)

    with pytest.raises(storage_mod.StorageUnavailableError):
        s.upload(io.BytesIO(b"x"), "k")
    assert any("S3 upload failed for key" in r.message for r in caplog.records if r.levelname == "ERROR")

    caplog.clear()

    monkeypatch.setattr(s.s3, "list_objects_v2", lambda *a, **k: (_ for _ in ()).throw(make_client_error()))
    with pytest.raises(Exception):
        s.list_files()
    assert any("List failed for prefix" in r.message for r in caplog.records if r.levelname == "ERROR")

    caplog.clear()
    err = make_client_error(code="NoSuchKey")
    monkeypatch.setattr(s.s3, "head_object", lambda *a, **k: (_ for _ in ()).throw(err))
    with pytest.raises(Exception):
        s.get_file_metadata("nokey")
    assert any("Could not get metadata for" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_dynamo_client_errors(monkeypatch, caplog):
    caplog.set_level("ERROR")
    d = dynamo_mod.DynamoDB()

    def bad_put(*a, **k):
        raise make_client_error()

    monkeypatch.setattr(d.table, "put_item", bad_put)
    with pytest.raises(RuntimeError):
        d.add_artifact("name", "model")
    assert any("Failed to add artifact" in r.message for r in caplog.records if r.levelname == "ERROR")

    caplog.clear()
    monkeypatch.setattr(d.table, "get_item", lambda *a, **k: (_ for _ in ()).throw(make_client_error()))
    assert d.get_artifact_by_id("1") is None
    assert any("Failed to get artifact" in r.message for r in caplog.records if r.levelname == "ERROR")

    caplog.clear()
    monkeypatch.setattr(d.table, "update_item", lambda *a, **k: (_ for _ in ()).throw(make_client_error()))
    assert d.update_artifact("1", name="x") is False
    assert any("Failed to update artifact" in r.message for r in caplog.records if r.levelname == "ERROR")

    caplog.clear()
    monkeypatch.setattr(d.table, "delete_item", lambda *a, **k: (_ for _ in ()).throw(make_client_error()))
    assert d.delete_artifact("1") is False
    assert any("Failed to delete artifact" in r.message for r in caplog.records if r.levelname == "ERROR")

    caplog.clear()
    monkeypatch.setattr(d.table, "scan", lambda *a, **k: (_ for _ in ()).throw(make_client_error()))
    assert d.list_artifacts(limit=10) == []
    assert any("Failed to list artifacts" in r.message for r in caplog.records if r.levelname == "ERROR")

    caplog.clear()
    assert d.get_registry_stats() == {"total": 0}
    assert any("Failed to get stats" in r.message for r in caplog.records if r.levelname == "ERROR")

    caplog.clear()
    with pytest.raises(RuntimeError):
        d.reset_registry()
    assert any("Failed to reset registry" in r.message for r in caplog.records if r.levelname == "ERROR")

# ==================================================
# BEGIN test_error_log_generation.py
# ==================================================

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

# ==================================================
# BEGIN test_errors_consolidated.py
# ==================================================

import zipfile
import subprocess
import logging


from API.storage import S3Storage

client = TestClient(server.app)


def test_rate_model_error_listing(monkeypatch, caplog):
    caplog.set_level("ERROR")

    def boom():
        raise Exception("boom")

    monkeypatch.setattr(server.registry_handler, "list_artifacts", boom)

    r = client.get("/artifact/model/1/rate")
    assert r.status_code == 500
    assert any("Error listing artifacts" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_rate_model_invalid_id_format(monkeypatch, caplog):
    caplog.set_level("ERROR")

    monkeypatch.setattr(server.registry_handler, "list_artifacts", lambda: [])

    r = client.get("/artifact/model/notanumber/rate")
    assert r.status_code == 404
    assert any("Invalid ID format" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_rate_model_not_found_logs_errors(monkeypatch, caplog):
    caplog.set_level("ERROR")

    monkeypatch.setattr(server.registry_handler, "list_artifacts", lambda: [])

    r = client.get("/artifact/model/9999/rate")
    assert r.status_code == 404
    assert any("Artifact '" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_license_check_errors(monkeypatch, caplog):
    caplog.set_level("ERROR")

    def bad_get(*a, **k):
        raise server.requests.RequestException("boom")

    monkeypatch.setattr(server, "_verify_artifact_exists", lambda *_a, **_k: True)
    monkeypatch.setattr(server, "requests", server.requests)
    monkeypatch.setattr(server.requests, "get", bad_get)

    r = client.post("/artifact/model/1/license-check", json={"github_url": "https://github.com/owner/repo"})
    assert r.status_code == 502
    assert any("Error fetching GitHub page" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_calculate_size_unexpected_error(monkeypatch, caplog):
    caplog.set_level("ERROR")

    def bad_get(*a, **k):
        raise Exception("boom")

    monkeypatch.setattr(server, "requests", server.requests)
    monkeypatch.setattr(server.requests, "get", bad_get)

    res = server._calculate_artifact_size_api("https://github.com/owner/repo", "model")
    assert res == 0.0
    assert any("GitHub API error" in r.message or "Unexpected error for" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_log_audit_event_failure(monkeypatch, caplog):
    caplog.set_level("ERROR")

    monkeypatch.setattr(server.registry_handler, "get_artifact_by_id", lambda *_a, **_k: (_ for _ in ()).throw(Exception("boom")))
    server._log_audit_event(1, "x", "model", "AUDIT")
    assert any("Failed to log audit event" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_lineage_artifact_not_found_and_invalid(monkeypatch, caplog):
    caplog.set_level("ERROR")

    monkeypatch.setattr(server.registry_handler, "list_artifacts", lambda: [])
    r = client.get("/artifact/model/123/lineage")
    assert r.status_code == 404
    assert any("Lineage request - Artifact not found" in r.message for r in caplog.records if r.levelname == "ERROR")

    caplog.clear()
    r = client.get("/artifact/model/notanumber/lineage")
    assert r.status_code == 404
    assert any("Lineage request - Invalid ID format" in r.message for r in caplog.records if r.levelname == "ERROR")


# S3 storage error-paths
def make_client_error(code="Err", msg="boom"):

    return ClientError({"Error": {"Code": code, "Message": msg}}, "op")


def test_storage_errors(monkeypatch, caplog, tmp_path):
    caplog.set_level("ERROR")
    s = S3Storage()

    def raise_it(*a, **k):
        raise make_client_error()

    monkeypatch.setattr(s.s3, "generate_presigned_url", raise_it)
    with pytest.raises(Exception):
        s.download_url("key")
    assert any("Could not generate download URL" in r.message for r in caplog.records if r.levelname == "ERROR")

    monkeypatch.setattr(s.s3, "download_file", lambda *a, **k: (_ for _ in ()).throw(make_client_error(code="NoSuchKey")))
    with pytest.raises(Exception):
        s.download_to_file("key", str(tmp_path / "out"))
    assert any("Download failed for" in r.message for r in caplog.records if r.levelname == "ERROR")

    monkeypatch.setattr(s.s3, "delete_object", raise_it)
    with pytest.raises(Exception):
        s.delete_file("k")
    assert any("Delete failed for" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_dynamo_errors(monkeypatch, caplog):
    caplog.set_level("ERROR")
    d = dynamo_mod.DynamoDB()

    def raise_it(*a, **k):

        raise ClientError({"Error": {"Code": "Err", "Message": "boom"}}, "op")

    monkeypatch.setattr(d.table, "get_item", raise_it)
    assert d.get_artifact_by_id("1") is None
    assert any("Failed to get artifact" in r.message for r in caplog.records if r.levelname == "ERROR")

    caplog.clear()
    monkeypatch.setattr(d.table, "update_item", raise_it)
    assert d.update_artifact("1", name="x") is False
    assert any("Failed to update artifact" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_artifact_downloader_error_paths(monkeypatch, caplog, tmp_path):
    caplog.set_level("ERROR")

    # git clone fails -> http 404 -> logs
    def bad_clone(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(ad.subprocess, "run", bad_clone)

    class Resp:
        status_code = 404

    monkeypatch.setattr(ad.requests, "get", lambda *a, **k: Resp())

    res = ad.download_github_repo("https://github.com/owner/repo", "a", str(tmp_path))
    assert res is False
    assert any("HTTP download failed" in r.message for r in caplog.records if r.levelname == "ERROR")


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
