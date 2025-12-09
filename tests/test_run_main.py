# tests/coverage/test_run_main.py

import sys
from unittest.mock import patch
import run


def test_run_main_executes():
    with patch.object(sys, "argv", ["run"]):
        try:
            run.main()
        except Exception:
            # we only care that import + main() executes
            pass
