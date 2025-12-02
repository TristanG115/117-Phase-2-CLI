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
        # Treescore doesn't depend on URLs directly, but can benefit from model metadata
        return [URLType.MODEL]

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

    def _calculate_treescore(self, artifact_id: Optional[int]) -> float:
        """
        Calculate average of parent model scores from lineage graph

        Args:
            artifact_id: ID of the model to calculate treescore for

        Returns:
            Average net_score of all parent models, or 0.0 if no parents
        """
        try:
            # Explicit None check for type safety
            if self.registry_handler is None:
                return 0.0

            # Try multiple methods to get parent information
            parent_scores: List[float] = []

            # Method 1: Check artifact metadata for parent_models
            try:
                artifact = self.registry_handler.get_artifact_by_id(str(artifact_id))
                if artifact:
                    import json

                    # Try metadata_json
                    metadata = json.loads(artifact.get("metadata_json", "{}"))

                    # Look for parent models in various fields
                    parent_names: List[str] = []
                    if "parent_models" in metadata:
                        parent_names = metadata["parent_models"]
                    elif "base_model" in metadata:
                        parent_names = [metadata["base_model"]]

                    if parent_names:
                        parent_scores = self._get_scores_by_names(parent_names)

            except Exception as e:
                self.logger.debug(f"Metadata check failed: {e}")

            # Method 2: Use model resources to infer relationships
            if not parent_scores:
                try:
                    artifact = self.registry_handler.get_artifact_by_id(
                        str(artifact_id)
                    )
                    if artifact:
                        related_score = self._find_related_models(artifact)
                        if related_score > 0:
                            parent_scores = [related_score]
                except Exception as e:
                    self.logger.debug(f"Related models check failed: {e}")

            if not parent_scores:
                return 0.0

            # Return average of parent scores
            return sum(parent_scores) / len(parent_scores)

        except Exception as e:
            self.logger.error(f"Error calculating treescore: {e}")
            return 0.0

    def _get_scores_by_names(self, parent_names: List[str]) -> List[float]:
        """Get net_scores for parent models by name"""
        scores: List[float] = []

        # Type safety check
        if self.registry_handler is None:
            return scores

        try:
            all_artifacts = self.registry_handler.list_artifacts()
            for parent_name in parent_names:
                # Normalize name for comparison
                parent_name_lower = parent_name.lower()

                for artifact in all_artifacts:
                    artifact_name = artifact.get("name", "").lower()

                    # Check for name match (exact or partial)
                    if (
                        parent_name_lower in artifact_name
                        or artifact_name in parent_name_lower
                    ):
                        try:
                            import json

                            metadata = json.loads(artifact.get("metadata_json", "{}"))
                            net_score = metadata.get("net_score", 0.0)
                            if net_score > 0:
                                scores.append(float(net_score))
                                break  # Found a match, move to next parent
                        except Exception:
                            continue
        except Exception as e:
            self.logger.debug(f"Error getting scores by names: {e}")

        return scores

    def _find_related_models(self, artifact: Dict[str, Any]) -> float:
        """
        Find related models based on shared datasets/code
        Returns average score of related models
        """
        # Type safety check
        if self.registry_handler is None:
            return 0.0

        try:
            # Get this model's dataset and code URLs
            my_dataset = artifact.get("dataset_url", "")
            my_code = artifact.get("code_url", "")

            if my_dataset == "unknown" and my_code == "unknown":
                return 0.0

            related_scores: List[float] = []
            all_artifacts = self.registry_handler.list_artifacts()

            for other in all_artifacts:
                # Skip self
                if other.get("name") == artifact.get("name"):
                    continue

                # Check for shared resources
                other_dataset = other.get("dataset_url", "")
                other_code = other.get("code_url", "")

                is_related = False
                if my_dataset != "unknown" and my_dataset == other_dataset:
                    is_related = True
                if my_code != "unknown" and my_code == other_code:
                    is_related = True

                if is_related:
                    try:
                        import json

                        metadata = json.loads(other.get("metadata_json", "{}"))
                        net_score = metadata.get("net_score", 0.0)
                        if net_score > 0:
                            related_scores.append(float(net_score))
                    except Exception:
                        continue

            if related_scores:
                return sum(related_scores) / len(related_scores)

        except Exception as e:
            self.logger.debug(f"Error finding related models: {e}")

        return 0.0
