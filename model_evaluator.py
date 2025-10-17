
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from url_classifier import URLClassifier, URLType
from resource_handlers import ModelHandler, DatasetHandler, CodeHandler, BaseResourceHandler
from metrics import METRIC_CLASSES
from metrics.base_metric import BaseMetric


class ModelEvaluator:
    """Main orchestrator for evaluating models with their associated datasets and code"""

    def __init__(self, max_workers: int = 4):
        self.url_classifier = URLClassifier()
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)

        # Initialize metrics
        self.metrics = {name: metric_class() for name, metric_class in METRIC_CLASSES.items()}

    def evaluate_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate a list of URLs and return results for MODEL URLs only

        Args:
            urls: List of URLs to evaluate

        Returns:
            List of evaluation results for model URLs
        """
        # Group URLs by type
        grouped_urls = self.url_classifier.group_urls_by_type(urls)

        # Create resource handlers
        resources = self._create_resource_handlers(grouped_urls)

        # Find model URLs to evaluate
        model_urls = grouped_urls[URLType.MODEL]

        results = []
        for model_url in model_urls:
            result = self._evaluate_single_model(model_url, resources)
            if result:
                results.append(result)

        return results

    def _create_resource_handlers(self, grouped_urls: Dict[URLType, List[str]]) -> Dict[URLType, List[BaseResourceHandler]]:
        """Create resource handlers for each URL type with error handling"""
        resources = {}

        # Create MODEL handlers
        if grouped_urls.get(URLType.MODEL):
            model_handlers = []
            for url in grouped_urls[URLType.MODEL]:
                try:
                    handler = ModelHandler(url)
                    model_handlers.append(handler)
                except Exception as e:
                    self.logger.error(f"Failed to create ModelHandler for {url}: {e}")
            if model_handlers:
                resources[URLType.MODEL] = model_handlers

        # Create DATASET handlers
        if grouped_urls.get(URLType.DATASET):
            dataset_handlers = []
            for url in grouped_urls[URLType.DATASET]:
                try:
                    handler = DatasetHandler(url)
                    dataset_handlers.append(handler)
                except Exception as e:
                    self.logger.error(f"Failed to create DatasetHandler for {url}: {e}")
            if dataset_handlers:
                resources[URLType.DATASET] = dataset_handlers

        # Create CODE handlers
        if grouped_urls.get(URLType.CODE):
            code_handlers = []
            for url in grouped_urls[URLType.CODE]:
                try:
                    handler = CodeHandler(url)
                    code_handlers.append(handler)
                except Exception as e:
                    self.logger.error(f"Failed to create CodeHandler for {url}: {e}")
            if code_handlers:
                resources[URLType.CODE] = code_handlers

        return resources

    def _evaluate_single_model(self, model_url: str, resources: Dict[URLType, List[BaseResourceHandler]]) -> Optional[Dict[str, Any]]:
        """Evaluate a single model with available resources"""
        try:
            model_handler = ModelHandler(model_url)
            # Extract just the model name part (e.g., "bert-base-uncased" from "google-bert/bert-base-uncased")
            model_id = model_handler.model_id or "unknown"
            if "/" in model_id:
                model_name = model_id.split("/")[-1]  # Get the last part after the slash
            else:
                model_name = model_id

            # Calculate metrics in parallel
            metric_results = self._calculate_metrics_parallel(resources)

            # Calculate net score
            net_score, net_score_latency = self._calculate_net_score(metric_results)

            # Build result according to specification
            result = {
                "name": model_name,
                "category": "MODEL",
                "net_score": net_score,
                "net_score_latency": net_score_latency,
                "ramp_up_time": metric_results.get("ramp_up_time", {}).get("score", 0.0),
                "ramp_up_time_latency": metric_results.get("ramp_up_time", {}).get("latency", 0),
                "bus_factor": metric_results.get("bus_factor", {}).get("score", 0.0),
                "bus_factor_latency": metric_results.get("bus_factor", {}).get("latency", 0),
                "performance_claims": metric_results.get("performance_claims", {}).get("score", 0.0),
                "performance_claims_latency": metric_results.get("performance_claims", {}).get("latency", 0),
                "license": metric_results.get("license", {}).get("score", 0.0),
                "license_latency": metric_results.get("license", {}).get("latency", 0),
                "size_score": metric_results.get("size_score", {}).get("score", {}),
                "size_score_latency": metric_results.get("size_score", {}).get("latency", 0),
                "dataset_and_code_score": metric_results.get("dataset_and_code_score", {}).get("score", 0.0),
                "dataset_and_code_score_latency": metric_results.get("dataset_and_code_score", {}).get("latency", 0),
                "dataset_quality": metric_results.get("dataset_quality", {}).get("score", 0.0),
                "dataset_quality_latency": metric_results.get("dataset_quality", {}).get("latency", 0),
                "code_quality": metric_results.get("code_quality", {}).get("score", 0.0),
                "code_quality_latency": metric_results.get("code_quality", {}).get("latency", 0)
            }

            return result

        except Exception as e:
            self.logger.error(f"Error evaluating model {model_url}: {e}")
            return None

    def _calculate_metrics_parallel(self, resources: Dict[URLType, List[BaseResourceHandler]]) -> Dict[str, Dict[str, Any]]:
        """Calculate all metrics in parallel with graceful handling of missing resources"""
        metric_results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all metric calculations
            future_to_metric = {}
            for metric_name, metric in self.metrics.items():
                # Get required resources for this metric
                required_types = metric.required_url_types()

                # Build available resources, including empty lists for missing types
                available_resources = {}
                for url_type in required_types:
                    available_resources[url_type] = resources.get(url_type, [])

                # Always try to calculate the metric, even with partial/missing resources
                # The metric implementations should handle missing resources gracefully
                future = executor.submit(self._safe_calculate_metric, metric, available_resources)
                future_to_metric[future] = metric_name

            # Collect results
            for future in as_completed(future_to_metric):
                metric_name = future_to_metric[future]
                try:
                    score, latency = future.result()
                    metric_results[metric_name] = {"score": score, "latency": latency}
                except Exception as e:
                    self.logger.error(f"Error calculating {metric_name}: {e}")
                    metric_results[metric_name] = {"score": 0.0, "latency": 0}

        return metric_results

    def _safe_calculate_metric(self, metric: BaseMetric, resources: Dict[URLType, List[BaseResourceHandler]]):
        """Safely calculate a metric with error handling"""
        try:
            return metric.calculate(resources)
        except Exception as e:
            self.logger.error(f"Error in metric calculation: {e}")
            return 0.0, 0

    def _calculate_net_score(self, metric_results: Dict[str, Dict[str, Any]]) -> Tuple[float, int]:
        """Calculate weighted net score"""
        # Define weights based on Sarah's priorities
        weights = {
            "license": 0.2,           # High priority - legal compliance
            "performance_claims": 0.15, # High priority - proven performance
            "ramp_up_time": 0.15,     # Important for adoption
            "bus_factor": 0.1,        # Risk management
            "size_score": 0.1,        # Deployment considerations
            "dataset_and_code_score": 0.1, # Reproducibility
            "dataset_quality": 0.1,   # Data quality matters
            "code_quality": 0.1       # Maintainability
        }

        weighted_sum = 0.0
        total_weight = 0.0
        total_latency = 0

        for metric_name, weight in weights.items():
            if metric_name in metric_results:
                score = metric_results[metric_name]["score"]
                latency = metric_results[metric_name]["latency"]

                # Handle size_score which is a dict
                if isinstance(score, dict):
                    # Average the hardware scores
                    if score:
                        score = sum(score.values()) / len(score.values())
                    else:
                        score = 0.0

                weighted_sum += score * weight
                total_weight += weight
                total_latency += latency

        net_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        return net_score, total_latency

    def evaluate_from_file(self, url_file_path: str) -> List[Dict[str, Any]]:
        """
        Evaluate URLs from a file where each line represents a group of related URLs

        Args:
            url_file_path: Path to file containing line-by-line comma-separated URLs

        Returns:
            List of evaluation results
        """
        self.logger.info(f"Starting evaluation of URL file: {url_file_path}")
        try:
            results = []
            with open(url_file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        # Split each line by commas and filter out empty URLs
                        line_urls = [url.strip() for url in line.split(',') if url.strip()]
                        if line_urls:
                            self.logger.info(f"Processing line {line_num} with {len(line_urls)} URLs")
                            # Evaluate each line's URLs as a group
                            line_results = self.evaluate_urls(line_urls)
                            results.extend(line_results)

            self.logger.info(f"Evaluation completed. Generated {len(results)} results")
            return results

        except FileNotFoundError:
            self.logger.error(f"URL file not found: {url_file_path}")
            return []
        except Exception as e:
            self.logger.error(f"Error reading URL file: {e}")
            return []

    def print_results_ndjson(self, results: List[Dict[str, Any]]) -> None:
        """Print results in NDJSON format to stdout"""
        for result in results:
            print(json.dumps(result))

    def setup_logging(self) -> None:
        """Setup logging based on environment variables"""
        log_file = os.environ.get('LOG_FILE')
        log_level = int(os.environ.get('LOG_LEVEL', '0'))

        if log_level == 0:
            logging.disable(logging.CRITICAL)
            return

        level = logging.INFO if log_level == 1 else logging.DEBUG

        # Re-enable logging first (in case it was disabled)
        logging.disable(logging.NOTSET)

        # Clear any existing handlers first
        logging.getLogger().handlers.clear()

        if log_file:
            try:
                # Validate log file path
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    # Try to create directory
                    os.makedirs(log_dir, exist_ok=True)

                # Test if we can write to the log file
                with open(log_file, 'a') as f:
                    pass

                logging.basicConfig(
                    filename=log_file,
                    level=level,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    force=True
                )
            except (OSError, PermissionError) as e:
                # Invalid log file path, fall back to console logging
                print(f"Warning: Cannot write to log file '{log_file}': {e}")
                logging.basicConfig(
                    level=level,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    force=True
                )
        else:
            logging.basicConfig(
                level=level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                force=True
            )


def main():
    """Main entry point for command line usage"""
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
