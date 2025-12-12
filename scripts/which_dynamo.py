import importlib
mod = importlib.import_module('API.dynamo')
print('module file=', getattr(mod,'__file__',None))
