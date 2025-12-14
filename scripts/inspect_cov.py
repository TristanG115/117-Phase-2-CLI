from coverage import CoverageData
import os

cov = CoverageData()
cov.read()

path = os.path.abspath('API/dynamo.py')
print('abs path:', path)
files = list(cov.measured_files())
print('measured files count:', len(files))
print('measured files sample:')
for f in files:
    print(f)
    if f.endswith('dynamo.py'):
        lines = sorted(cov.lines(f) or [])
        print('->', f, 'lines count', len(lines))
        print(lines[:200])
