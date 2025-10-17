from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Any
import logging
from url_classifier import URLType


class BaseMetric(ABC):
    """Base class for all metrics"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def required_url_types(self) -> List[URLType]:
        """Returns list of URL types required to calculate this metric"""
        pass

    @abstractmethod
    def calculate(self, resources: Dict[URLType, List[Any]]) -> Tuple[float, int]:
        """
        Calculate the metric score

        Args:
            resources: Dictionary mapping URLType to list of resource handlers

        Returns:
            Tuple of (score, latency_ms)
        """
        pass