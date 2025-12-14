from coverage import CoverageData
cov = CoverageData()
cov.read()
names = ['dynamo','storage','server']
for f in sorted(cov.measured_files()):
    for n in names:
        if n in f.lower():
            print(n, '->', f)
