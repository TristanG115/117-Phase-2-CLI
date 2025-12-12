from API.dynamo import DynamoDB
from unittest.mock import Mock
from botocore.exceptions import ClientError


def make_client_error():
    return ClientError({"Error": {"Code": "InternalError", "Message": "boom"}}, "Op")


def main():
    # Monkeypatch load_config to avoid file access
    DynamoDB.load_config = lambda self: {"AWS_REGION": "us-west-2", "S3_BUCKET_NAME": "b"}
    # Prevent boto3 resource creation
    import API.dynamo as mod
    mod.boto3.resource = lambda *a, **k: Mock(Table=lambda name: Mock())

    d = DynamoDB(table_name="t")
    d.table = Mock()
    d.table.put_item.side_effect = make_client_error()

    try:
        d.add_artifact("name", "model")
    except Exception as e:
        print("caught", type(e), e)


if __name__ == '__main__':
    main()
