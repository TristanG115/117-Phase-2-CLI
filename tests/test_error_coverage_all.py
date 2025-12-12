import io
import json
import asyncio
import os
import pytest
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
