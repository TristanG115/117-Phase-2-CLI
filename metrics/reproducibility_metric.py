import subprocess
import tempfile
import time
from typing import Any, Dict, List, Tuple

from API.storage import S3Storage
from url_classifier import URLType

from .base_metric import BaseMetric


class ReproducibilityMetric(BaseMetric):
    """Metric to evaluate whether a model can be run using the demo code"""

    def __init__(self):
        super().__init__()
        self.s3 = S3Storage()

    def required_url_types(self) -> List[URLType]:
        # Needs the code to test reproducibility
        return [URLType.CODE]

    def _fetch_code(self, code_ref: str, tmpdir: str) -> str:
        """
        Fetch code either from S3 or local path.

        Args:
            code_ref: S3 key or local file path
            tmpdir: temp directory to save code

        Returns:
            Path to the local script
        """
        if code_ref.startswith("s3://"):
            # Extract S3 key
            key = code_ref[len("s3://") + len(self.s3.bucket_name) + 1 :]
            local_path = f"{tmpdir}/{key.split('/')[-1]}"
            self.s3.download_to_file(key, local_path)
            return local_path
        else:
            # Local file
            return code_ref

    def calculate(
        self, resources: Dict[URLType, List[Any]], **kwargs
    ) -> Tuple[float, int]:
        start_time = time.time()

        code_resources = resources.get(URLType.CODE, [])
        if not code_resources:
            self.logger.warning("No code provided")
            # No code means we can't evaluate reproducibility
            # Give partial credit if the repo exists and looks legitimate
            return 0.0, int((time.time() - start_time) * 1000)

        # Check if code repo has good indicators even if we can't run it
        code_handler = code_resources[0]
        base_score = self._evaluate_code_indicators(code_handler)

        # Try to run the code for additional points
        execution_score = 0.0
        for code_handler in code_resources:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    script_path = self._fetch_code(code_handler, tmpdir)

                    result = subprocess.run(
                        ["python", script_path],
                        cwd=tmpdir,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )

                    if result.returncode == 0:
                        execution_score = 0.4  # Bonus for successful execution
                        break
                    else:
                        self.logger.info(
                            f"Code returned non-zero exit: {result.stderr[:100]}"
                        )
                        execution_score = 0.1  # Small bonus for attempting to run

            except subprocess.TimeoutExpired:
                self.logger.info("Code execution timed out")
                execution_score = 0.1  # Small bonus for code that runs (even if slow)
            except Exception as e:
                self.logger.info(f"Could not execute code: {e}")
                # Don't penalize for execution failures

        final_score = min(base_score + execution_score, 1.0)

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        return round(final_score, 2), latency_ms

    def _evaluate_code_indicators(self, code_handler: Any) -> float:
        """
        Evaluate reproducibility based on code quality indicators
        rather than actual execution.
        """
        score = 0.3  # Base score for having code

        try:
            # Check for README
            if hasattr(code_handler, "get_github_api_data"):
                api_data = code_handler.get_github_api_data()
                if api_data.get("has_readme"):
                    score += 0.1

            # Check for tests (indicates runnable code)
            if hasattr(code_handler, "has_tests") and code_handler.has_tests():
                score += 0.1

            # Check for CI/CD (indicates code that runs)
            if hasattr(code_handler, "has_ci_cd") and code_handler.has_ci_cd():
                score += 0.1

            # Check for requirements/dependencies file
            if hasattr(code_handler, "get_repo_tree"):
                tree = code_handler.get_repo_tree()
                dep_files = [
                    "requirements.txt",
                    "setup.py",
                    "pyproject.toml",
                    "environment.yml",
                    "conda.yaml",
                    "Pipfile",
                ]
                has_deps = any(
                    any(dep_file in item.get("path", "") for dep_file in dep_files)
                    for item in tree
                )
                if has_deps:
                    score += 0.1

        except Exception as e:
            self.logger.warning(f"Error evaluating code indicators: {e}")

        return min(score, 0.6)  # Cap base score at 0.6 (execution can add 0.4)
