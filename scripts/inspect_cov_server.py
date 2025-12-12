from coverage import CoverageData
import os

cov = CoverageData()
cov.read('.coverage')
path = os.path.abspath('server.py')
lines = cov.lines(path) or []
print('server.py covered lines count', len(lines))
print('sample lines:', sorted(lines)[:100])

for check in [1259, 1263, 1266, 1766, 914, 911]:
    print(check, 'present' if check in lines else 'missing')
