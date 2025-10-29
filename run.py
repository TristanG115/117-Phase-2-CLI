#!/usr/bin/env python3
import json
import logging
import os
import subprocess
import sys

import requests

from handlers import ingest_handler, registry_handler
from model_evaluator import ModelEvaluator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(SCRIPT_DIR, "requirements.txt")


def validate_github_token() -> None:
    token = os.getenv("GITHUB_TOKEN")
    if not token or not token.strip():
        sys.stderr.write("Error: Invalid GITHUB_TOKEN\n")
        sys.exit(1)
    headers = {"Authorization": f"token {token}"}
    try:
        resp = requests.get(
            "https://api.github.com/rate_limit", headers=headers, timeout=5
        )
        if resp.status_code != 200:
            sys.stderr.write("Error: Invalid GITHUB_TOKEN\n")
            sys.exit(1)
    except Exception:
        sys.stderr.write("Error: Invalid GITHUB_TOKEN\n")
        sys.exit(1)


def validate_log_file() -> None:
    log_path = os.getenv("LOG_FILE")
    if not log_path:
        sys.stderr.write("Error: LOG_FILE not set\n")
        sys.exit(1)
    parent = os.path.dirname(log_path) or "."
    if not os.path.isdir(parent):
        sys.stderr.write(f"Error: parent directory {parent} does not exist\n")
        sys.exit(1)
    if os.path.exists(log_path):
        if not os.access(log_path, os.W_OK):
            sys.stderr.write(f"Error: cannot write to log file {log_path}\n")
            sys.exit(1)
    else:
        sys.stderr.write(f"Error: log file {log_path} does not exist\n")
        sys.exit(1)

    level_str = os.getenv("LOG_LEVEL", "1")
    if level_str == "0":
        logging.disable(logging.CRITICAL)
        return
    elif level_str == "2":
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(
        filename=log_path,
        level=log_level,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    logging.info("Logging initialized successfully.")


def install_dependencies():
    try:
        logging.info("Installing dependencies...")
        if not os.path.exists(REQUIREMENTS):
            with open(REQUIREMENTS, "w") as f:
                f.write(
                    """requests>=2.25.0
beautifulsoup4>=4.9.0
lxml>=4.6.0
python-dateutil>=2.8.0
urllib3>=1.26.0
GitPython>=3.1.0
PyGithub>=1.55.0
huggingface-hub>=0.10.0
flake8==7.0.0
black==24.8.0
pre-commit==3.6.2
pytest==8.3.2
coverage==7.3.2
"""
                )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS]
        )
        logging.info("Dependencies installed successfully.")
        print("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] pip failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected install error: {e}")
        sys.exit(1)


def process_url_file(url_file_path):
    try:
        if not os.path.exists(url_file_path):
            print(f"Error: URL file '{url_file_path}' not found.")
            sys.exit(1)
        evaluator = ModelEvaluator()
        evaluator.setup_logging()
        results = evaluator.evaluate_from_file(url_file_path)
        if not results:
            print("No model URLs found or processed successfully")
            sys.exit(1)
        evaluator.print_results_ndjson(results)
    except Exception as e:
        print(f"Error processing URL file: {e}")
        sys.exit(1)


def run_tests():
    import io
    import unittest

    import coverage

    tests_dir = os.path.join(SCRIPT_DIR, "tests")
    if not os.path.isdir(tests_dir):
        print("Error: No tests directory found")
        sys.exit(1)

    cov = coverage.Coverage()
    cov.start()
    loader = unittest.TestLoader()
    suite = loader.discover(tests_dir, pattern="test_*.py")
    buffer = io.StringIO()
    runner = unittest.TextTestRunner(stream=buffer, verbosity=1)
    result = runner.run(suite)
    cov.stop()
    cov.save()

    total_lines = 0
    covered_lines = 0
    for filename in cov.get_data().measured_files():
        if any(
            filename.endswith(f)
            for f in [
                "model_evaluator.py",
                "url_classifier.py",
                "resource_handlers.py",
                "metrics.py",
            ]
        ):
            analysis = cov.analysis2(filename)
            total_lines += len(analysis[1]) + len(analysis[2])
            covered_lines += len(analysis[1])

    coverage_percent = (
        int((covered_lines / total_lines * 100)) if total_lines > 0 else 0
    )
    total_tests = result.testsRun
    passed_tests = total_tests - len(result.failures) - len(result.errors)

    # exact syntax autograder expects
    sys.stdout.write(
        f"{passed_tests}/{total_tests} test cases passed. {coverage_percent}% line coverage achieved.\n"
    )
    sys.stdout.flush()
    sys.exit(0)


def run_tests_debug():
    try:
        import unittest

        loader = unittest.TestLoader()
        suite = loader.discover(os.path.join(SCRIPT_DIR, "tests"), pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        print("\n" + "=" * 70)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print("=" * 70)
        sys.exit(0 if result.wasSuccessful() else 1)
    except Exception as e:
        print(f"Tests failed: {e}")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: ./run <command>")
        print("\nCommands:")
        print("  install       Install project dependencies")
        print("  test          Run test suite with coverage")
        print("  debug         Run tests with verbose output")
        print("  ingest        Ingest a Hugging Face model into the registry")
        print("  list          List all locally ingested models")
        print("  search        Search local registry by model name or tags")
        print("  reset         Clear all stored models and downloaded files")
        print("  <URL_FILE>    Process and evaluate URLs from file")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd not in ("install", "debug"):
        validate_github_token()
        validate_log_file()

    if cmd == "install":
        install_dependencies()
    elif cmd == "test":
        run_tests()
    elif cmd == "debug":
        run_tests_debug()
    elif cmd == "ingest":
        if len(sys.argv) < 3:
            print("Usage: ./run ingest <huggingface_model_url>")
            sys.exit(1)
        hf_url = sys.argv[2]
        registry_handler.init_registry()
        result = ingest_handler.ingest_model(hf_url)
        print(json.dumps(result, indent=2))
        sys.exit(0)

    elif cmd == "list":
        registry_handler.init_registry()
        models = registry_handler.list_models()
        print(json.dumps(models, indent=2))
        sys.exit(0)

    elif cmd == "search":
        if len(sys.argv) < 3:
            print("Usage: ./run search <query>")
            sys.exit(1)
        query = sys.argv[2]
        registry_handler.init_registry()
        results = registry_handler.search_models(query)
        print(json.dumps(results, indent=2))
        sys.exit(0)
    elif cmd == "reset":
        registry_handler.init_registry()
        registry_handler.reset_registry()
        print("System registry reset successfully.")
        sys.exit(0)
    else:
        process_url_file(cmd)


if __name__ == "__main__":
    main()
