#!/usr/bin/env python3
import subprocess
import sys
import os
import re

HERE = os.getcwd()
PY = sys.executable

def run(cmd):
    print(f"RUN: {cmd}")
    res = subprocess.run(cmd, shell=True)
    return res.returncode


def run_tests_with_coverage():
    # Run tests under coverage so .coverage data is written
    cmd = f"{PY} -m coverage run -m pytest -q"
    return run(cmd)


def run_checker():
    cmd = f"{PY} scripts/check_error_coverage_summary.py"
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    print(out)
    # try to parse percent
    m = re.search(r"\((?P<p>\d+\.?\d*)%", out)
    p = float(m.group('p')) if m else None
    return proc.returncode, p


def main():
    code = run_tests_with_coverage()
    # run checker regardless of tests exit code
    ret, pct = run_checker()
    if pct is None:
        print("Could not parse checker percentage.")
        sys.exit(1)
    if pct < 80.0:
        print(f"ERROR: Only {pct:.1f}% of project error sites executed during tests (<80%).")
        sys.exit(2)
    print(f"OK: {pct:.1f}% >= 80%")
    sys.exit(0)


if __name__ == '__main__':
    main()
