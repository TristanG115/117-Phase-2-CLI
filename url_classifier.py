from typing import List, Dict
from urllib.parse import urlparse
from enum import Enum


class URLType(Enum):
    MODEL = "MODEL"
    DATASET = "DATASET"
    CODE = "CODE"
    UNKNOWN = "UNKNOWN"


class URLClassifier:
    """Classifies URLs into MODEL, DATASET, or CODE categories"""

    def __init__(self):
        pass

    def classify_url(self, url: str) -> URLType:
        """
        Classify a URL into MODEL, DATASET, or CODE category

        Args:
            url: The URL to classify

        Returns:
            URLType enum value
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()

        if "huggingface.co" in domain:
            if "/datasets/" in path:
                return URLType.DATASET
            else:
                return URLType.MODEL
        elif "github.com" in domain:
            return URLType.CODE
        else:
            return URLType.UNKNOWN

    def group_urls_by_type(self, urls: List[str]) -> Dict[URLType, List[str]]:
        """
        Group a list of URLs by their type

        Args:
            urls: List of URLs to classify

        Returns:
            Dictionary mapping URLType to list of URLs of that type
        """
        grouped = {url_type: [] for url_type in URLType}

        for url in urls:
            url_type = self.classify_url(url)
            grouped[url_type].append(url)

        return grouped