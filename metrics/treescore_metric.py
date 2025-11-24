import time
from typing import Any, Dict, List, Optional, Tuple

from url_classifier import URLType

from .base_metric import BaseMetric


class TreescoreMetric(BaseMetric):
    """Metric for calculating average score of parent models in lineage graph"""

    def __init__(self, registry_handler=None):
        super().__init__()
        self.registry_handler = registry_handler

    def required_url_types(self) -> List[URLType]:
        # Treescore doesn't depend on URLs, it depends on the lineage graph
        return []

    def calculate(
        self, resources: Dict[URLType, List[Any]], **kwargs
    ) -> Tuple[float, int]:
        start_time = time.time()

        # Extract artifact_id from kwargs
        artifact_id = kwargs.get("artifact_id")

        if not self.registry_handler or artifact_id is None:
            # No registry handler or artifact ID, cannot calculate treescore
            return 0.0, int((time.time() - start_time) * 1000)

        treescore = self._calculate_treescore(artifact_id)

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        return round(treescore, 2), latency_ms

    def _calculate_treescore(self, artifact_id: Optional[int]) -> float:  # noqa: C901
        """
        Calculate average of parent model scores from lineage graph

        Args:
            artifact_id: ID of the model to calculate treescore for

        Returns:
            Average net_score of all parent models, or 0.0 if no parents
        """
        try:
            # Explicit None check for type checker
            if self.registry_handler is None:
                return 0.0

            # Get lineage information for this artifact
            lineage = self.registry_handler.get_lineage(artifact_id)

            if not lineage or "parents" not in lineage:
                # No parents, treescore is 0
                return 0.0

            parent_ids = lineage["parents"]
            if not parent_ids:
                return 0.0

            # Get scores for all parents
            parent_scores = []
            for parent_id in parent_ids:
                parent_artifact = self.registry_handler.get_artifact_by_id(parent_id)
                if parent_artifact:
                    # Extract net_score from metadata
                    try:
                        import json

                        metadata = json.loads(
                            parent_artifact.get("metadata_json", "{}")
                        )
                        net_score = metadata.get("net_score", 0.0)
                        parent_scores.append(float(net_score))
                    except Exception as e:
                        self.logger.warning(
                            f"Could not extract score for parent {parent_id}: {e}"
                        )
                        continue

            if not parent_scores:
                return 0.0

            # Return average of parent scores
            return sum(parent_scores) / len(parent_scores)

        except Exception as e:
            self.logger.error(f"Error calculating treescore: {e}")
            return 0.0
