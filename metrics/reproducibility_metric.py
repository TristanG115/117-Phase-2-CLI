import time
import subprocess
import tempfile
import logging
from typing import Dict, List, Any, Tuple
from base_metric import BaseMetric
from url_classifier import URLType
from API.storage import S3Storage


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
        
    def calculate(self, resources: Dict[URLType, List[Any]]) -> Tuple[float, int]:
        start_time = time.time()

        code_resources = resources.get(URLType.CODE, [])
        if not code_resources:
            self.logger.warning("No code provided")
            return 0.0, int((time.time() - start_time) * 1000)
        
        score = 0.0
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
                        score = 1.0  # Runs with no debugging
                        break
                    else:
                        self.logger.warning(
                            f"Code returned non-zero exit: {result.stderr}"
                        )
                        score = 0.5  # Runs with debugging

            except subprocess.TimeoutExpired:
                # Code hung, runs with debugging potential
                self.logger.warning("Code execution timed out")
                score = 0.5
            except Exception as e:
                # Code failed to run, cannot reproduce
                self.logger.warning(f"Exception during reproducibility test: {e}")
                score = 0.0

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        return score, latency_ms