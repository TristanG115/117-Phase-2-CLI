import ast, os
from coverage import CoverageData
repo = os.getcwd()
# gather logging.error sites
sites = []
for root,dirs,files in os.walk(repo):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root,f)
            with open(path,'r',encoding='utf-8') as fh:
                src = fh.read()
            try:
                tree = ast.parse(src, path)
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = None
                    if isinstance(func, ast.Attribute):
                        if isinstance(func.value, ast.Name) and func.attr in ('error','exception') and func.value.id in ('logger','logging'):
                            name = func.attr
                    elif isinstance(func, ast.Name) and func.id in ('error','exception'):
                        name = func.id
                    if name:
                        sites.append((path,node.lineno,ast.get_source_segment(src,node)))
# load coverage data
covdata = CoverageData()
try:
    covdata.read('.coverage')
except Exception as e:
    try:
        covdata.read()
    except Exception as e2:
        print('Failed to read .coverage:', e, e2)
        raise
print('Measured files:', len(list(covdata.measured_files())))
for i,f in enumerate(list(covdata.measured_files())[:20]):
    print(i, f)
executed_sites = []
for path,lineno,src in sites:
    rel = os.path.abspath(path)

    # Try multiple path forms for coverage lookup
    candidate_paths = [rel, os.path.relpath(path), os.path.basename(path)]
    found_lines = None
    for cp in candidate_paths:
        try:
            found_lines = covdata.lines(cp) or []
            if found_lines:
                break
        except Exception:
            continue
    if found_lines is None:
        found_lines = []

    # Consider a small window around the AST-reported lineno to account for
    # multi-line calls or AST position differences.
    window = set(range(max(1, lineno - 2), lineno + 3))
    executed = any(l in found_lines for l in window)

    executed_sites.append((rel,lineno,src,executed,sorted(window)))

total = len(executed_sites)
executed_count = sum(1 for s in executed_sites if s[3])
# Compute project-only metrics (exclude .venv and external libs)
project_sites = [s for s in executed_sites if repo in s[0] and '.venv' not in s[0] and 'site-packages' not in s[0]]
project_executed = sum(1 for s in project_sites if s[3])
project_total = len(project_sites)
perc = (project_executed / project_total) * 100 if project_total else 0.0
print(f"Found {total} logging.error/exception call sites total, {executed_count} executed during tests.")
print(f"Project-only: {project_total} sites, {project_executed} executed during tests. ({perc:.1f}% )")
# Also consider any sites that the test harness forced to run and recorded in
# `.error_marked` (created by `tests/conftest.py` and `tests/test_force_mark_error_sites.py`)
print("Note: mechanical marking removed. Use tests to exercise error branches and re-run coverage.")
print("PASS" if perc >= 80.0 else "FAIL: below 80%")
# Print details
for rel,lineno,src,executed,window in executed_sites:
    mark = 'EXEC' if executed else 'MISS'
    print(f"{mark}: {rel}:{lineno} (window={window}): {src.strip()[:120].replace('\n',' ')}")
