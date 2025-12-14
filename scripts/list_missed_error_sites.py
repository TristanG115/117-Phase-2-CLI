import ast, os, sys
from coverage import CoverageData

repo = os.getcwd()
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
                        sites.append((path,node.lineno))

covdata = CoverageData()
covdata.read()

executed_sites = []
for path,lineno in sites:
    rel = os.path.abspath(path)
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
    window = set(range(max(1, lineno - 2), lineno + 3))
    executed = any(l in found_lines for l in window)
    executed_sites.append((rel,lineno,executed))

repo_sites = [s for s in executed_sites if repo in s[0] and '.venv' not in s[0] and 'site-packages' not in s[0]]

misses = [s for s in repo_sites if not s[2]]
print(f"Project-only: {len(repo_sites)} sites, {sum(1 for s in repo_sites if s[2])} executed during tests. ({(sum(1 for s in repo_sites if s[2])/len(repo_sites)*100) if repo_sites else 0:.1f}% )")
if not misses:
    print("No missed sites.")
    sys.exit(0)

print('\nMissed logger sites:')
for path,lineno,exe in sorted(misses):
    try:
        with open(path,'r',encoding='utf-8') as fh:
            lines = fh.readlines()
        srcline = lines[lineno-1].strip()
    except Exception:
        srcline = '<source unavailable>'
    print(f"- {path}:{lineno}: {srcline}")
