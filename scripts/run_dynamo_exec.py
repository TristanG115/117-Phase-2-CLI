from API import dynamo as dynamo_mod
from botocore.exceptions import ClientError

class _Raise:
    def __init__(self, exc):
        self._exc = exc
    def __getattr__(self, _):
        def f(*a, **k):
            raise self._exc
        return f

def _client_error(msg='boom'):
    return ClientError({'Error':{'Code':'Error','Message':msg}}, 'op')

orig = dynamo_mod.DynamoDB.__init__
dynamo_mod.DynamoDB.__init__ = lambda self: None
db = dynamo_mod.DynamoDB()
dynamo_mod.DynamoDB.__init__ = orig
db.table = _Raise(_client_error())

try:
    db.add_artifact('n','model')
except Exception as e:
    print('caught', type(e), e)
