import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from metrics import METRIC_CLASSES
from metrics.base_metric import BaseMetric

# Phase 2 Metrics (only reproducibility exists)
from metrics.reproducibility_metric import ReproducibilityMetric
from resource_handlers import (
    BaseResourceHandler,
    CodeHandler,
    DatasetHandler,
    ModelHandler,
)
from url_classifier import URLClassifier, URLType

# Phase 2  metrics


class ReviewednessMetric(BaseMetric):
    """Stub: GitHub PR reviewedness metric (not implemented yet)."""

    def required_url_types(self) -> List[URLType]:
        return []

    def calculate(self, resources: Dict[URLType, List[Any]]) -> Tuple[float, int]:
        return -1.0, 1


class TreeScoreMetric(BaseMetric):
    """Stub: Parent average score metric (not implemented yet)."""

    def required_url_types(self) -> List[URLType]:
        return []

    def calculate(self, resources: Dict[URLType, List[Any]]) -> Tuple[float, int]:
        return -1.0, 1


class LineageMetric(BaseMetric):
    """Stub: Lineage calculation metric (not implemented yet)."""

    def required_url_types(self) -> List[URLType]:
        return []

    def calculate(self, resources: Dict[URLType, List[Any]]) -> Tuple[float, int]:
        return -1.0, 1


class ModelEvaluator:
    """Main orchestrator for evaluating models with their associated datasets and code"""

    def __init__(self, max_workers: int = 4):
        self.url_classifier = URLClassifier()
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)

        # --- Phase 1 metrics ---
        self.metrics = {
            name: metric_class() for name, metric_class in METRIC_CLASSES.items()
        }

        # --- Add Phase 2 metrics (one real metric, three stubs) ---
        self.metrics["reproducibility"] = ReproducibilityMetric()
        self.metrics["reviewedness"] = ReviewednessMetric()  # always returns -1
        self.metrics["treescore"] = TreeScoreMetric()  # always returns -1
        self.metrics["lineage"] = LineageMetric()  # always returns -1

    def evaluate_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        grouped_urls = self.url_classifier.group_urls_by_type(urls)
        resources = self._create_resource_handlers(grouped_urls)
        model_urls = grouped_urls[URLType.MODEL]

        results = []
        for model_url in model_urls:
            result = self._evaluate_single_model(model_url, resources)
            if result:
                results.append(result)

        return results

    def _create_resource_handlers(
        self, grouped_urls: Dict[URLType, List[str]]
    ) -> Dict[URLType, List[BaseResourceHandler]]:
        resources: Dict[URLType, List[BaseResourceHandler]] = {}

        if grouped_urls.get(URLType.MODEL):
            resources[URLType.MODEL] = [
                ModelHandler(u) for u in grouped_urls[URLType.MODEL]
            ]

        if grouped_urls.get(URLType.DATASET):
            resources[URLType.DATASET] = [
                DatasetHandler(u) for u in grouped_urls[URLType.DATASET]
            ]

        if grouped_urls.get(URLType.CODE):
            resources[URLType.CODE] = [
                CodeHandler(u) for u in grouped_urls[URLType.CODE]
            ]

        return resources

    def _evaluate_single_model(
        self, model_url: str, resources: Dict[URLType, List[BaseResourceHandler]]
    ) -> Optional[Dict[str, Any]]:
        try:
            model_handler = ModelHandler(model_url)

            model_id = model_handler.model_id or "unknown"
            model_name = model_id.split("/")[-1] if "/" in model_id else model_id

            metric_results = self._calculate_metrics_parallel(resources)
            net_score, net_score_latency = self._calculate_net_score(metric_results)

            size_score_dict = metric_results.get("size_score", {}).get("score", {})
            if isinstance(size_score_dict, dict):
                size_score_dict = {k: round(v, 2) for k, v in size_score_dict.items()}

            result = {
                "name": model_name,
                "category": "MODEL",
                "net_score": net_score,
                "net_score_latency": net_score_latency,
                # -------- Phase 1 metrics --------
                "ramp_up_time": round(metric_results["ramp_up_time"]["score"], 2),
                "ramp_up_time_latency": metric_results["ramp_up_time"]["latency"],
                "bus_factor": round(metric_results["bus_factor"]["score"], 2),
                "bus_factor_latency": metric_results["bus_factor"]["latency"],
                "performance_claims": round(
                    metric_results["performance_claims"]["score"], 2
                ),
                "performance_claims_latency": metric_results["performance_claims"][
                    "latency"
                ],
                "license": round(metric_results["license"]["score"], 2),
                "license_latency": metric_results["license"]["latency"],
                "size_score": size_score_dict,
                "size_score_latency": metric_results["size_score"]["latency"],
                "dataset_and_code_score": round(
                    metric_results["dataset_and_code_score"]["score"], 2
                ),
                "dataset_and_code_score_latency": metric_results[
                    "dataset_and_code_score"
                ]["latency"],
                "dataset_quality": round(metric_results["dataset_quality"]["score"], 2),
                "dataset_quality_latency": metric_results["dataset_quality"]["latency"],
                "code_quality": round(metric_results["code_quality"]["score"], 2),
                "code_quality_latency": metric_results["code_quality"]["latency"],
                # -------- Phase 2 metrics (one real, three stubs) --------
                "reproducibility": round(metric_results["reproducibility"]["score"], 2),
                "reproducibility_latency": metric_results["reproducibility"]["latency"],
                "reviewedness": round(metric_results["reviewedness"]["score"], 2),
                "reviewedness_latency": metric_results["reviewedness"]["latency"],
                "treescore": round(metric_results["treescore"]["score"], 2),
                "treescore_latency": metric_results["treescore"]["latency"],
                "lineage": round(metric_results["lineage"]["score"], 2),
                "lineage_latency": metric_results["lineage"]["latency"],
            }

            return result

        except Exception as e:
            self.logger.error(f"Error evaluating model {model_url}: {e}")
            return None

    def _calculate_metrics_parallel(
        self, resources: Dict[URLType, List[BaseResourceHandler]]
    ) -> Dict[str, Dict[str, Any]]:
        metric_results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_metric = {}

            for metric_name, metric in self.metrics.items():
                required = metric.required_url_types()

                available = {
                    url_type: resources.get(url_type, []) for url_type in required
                }

                future = executor.submit(self._safe_calculate_metric, metric, available)
                future_to_metric[future] = metric_name

            for future in as_completed(future_to_metric):
                mname = future_to_metric[future]
                try:
                    score, latency = future.result()
                    metric_results[mname] = {"score": score, "latency": latency}
                except Exception as e:
                    self.logger.error(f"Error calculating {mname}: {e}")
                    metric_results[mname] = {"score": 0.0, "latency": 0}

        return metric_results

    def _safe_calculate_metric(
        self, metric: BaseMetric, resources: Dict[URLType, List[BaseResourceHandler]]
    ):
        try:
            return metric.calculate(resources)
        except Exception as e:
            self.logger.error(f"Error in metric calculation: {e}")
            return 0.0, 0

    def _calculate_net_score(
        self, metric_results: Dict[str, Dict[str, Any]]
    ) -> Tuple[float, int]:
        """Equal-weight net score."""
        numeric_scores = []
        total_latency = 0

        for name, mr in metric_results.items():
            score = mr["score"]
            total_latency += mr["latency"]

            if isinstance(score, dict):
                continue

            numeric_scores.append(float(score))

        if numeric_scores:
            net_score = sum(numeric_scores) / len(numeric_scores)
        else:
            net_score = 0.0

        return round(net_score, 2), total_latency

    def evaluate_from_file(self, url_file_path: str) -> List[Dict[str, Any]]:
        self.logger.info(f"Starting evaluation of URL file: {url_file_path}")
        try:
            results = []
            with open(url_file_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        urls = [u.strip() for u in line.split(",") if u.strip()]
                        if urls:
                            line_results = self.evaluate_urls(urls)
                            results.extend(line_results)
            return results

        except Exception as e:
            self.logger.error(f"Error reading URL file: {e}")
            return []

    def print_results_ndjson(self, results: List[Dict[str, Any]]) -> None:
        for result in results:
            print(json.dumps(result))

    def setup_logging(self) -> None:
        log_file = os.environ.get("LOG_FILE")
        log_level = int(os.environ.get("LOG_LEVEL", "0"))

        if log_level == 0:
            logging.disable(logging.CRITICAL)
            return

        level = logging.INFO if log_level == 1 else logging.DEBUG
        logging.disable(logging.NOTSET)
        logging.getLogger().handlers.clear()

        if log_file:
            try:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                open(log_file, "a").close()

                logging.basicConfig(
                    filename=log_file,
                    level=level,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    force=True,
                )
            except Exception as e:
                print(f"Warning: Cannot write to log file '{log_file}': {e}")
                logging.basicConfig(level=level, force=True)
        else:
            logging.basicConfig(level=level, force=True)


def main():
    if len(sys.argv) != 2:
        print("Usage: python model_evaluator.py <URL_FILE>", file=sys.stderr)
        sys.exit(1)

    url_file = sys.argv[1]

    evaluator = ModelEvaluator()
    evaluator.setup_logging()

    results = evaluator.evaluate_from_file(url_file)

    if not results:
        print("No results generated", file=sys.stderr)
        sys.exit(1)

    evaluator.print_results_ndjson(results)


if __name__ == "__main__":
    main()
