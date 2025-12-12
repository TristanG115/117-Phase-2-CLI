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
