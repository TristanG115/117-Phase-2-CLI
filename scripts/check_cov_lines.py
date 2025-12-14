from coverage import CoverageData
import os

cov = CoverageData()
cov.read()

targets = ["API/dynamo.py","API/storage.py","server.py"]
for p in targets:
    ap = os.path.abspath(p)
    print('checking', ap)
    try:
        lines = cov.lines(ap)
    except Exception as e:
        lines = None
    print('->', lines)

print('\nMeasured files count:', len(list(cov.measured_files())))
print('Measured files:')
for f in sorted(cov.measured_files()):
    print(f)
