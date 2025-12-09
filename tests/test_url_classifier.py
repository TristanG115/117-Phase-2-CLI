import unittest
import url_classifier as uc

from url_classifier import URLClassifier, URLType


def is_huggingface_model_url(url: str) -> bool:
    return "huggingface.co" in url and "/datasets/" not in url

def extract_model_name(url: str):
    if "huggingface.co" not in url:
        return None
    parts = url.split("huggingface.co/")[-1].split("/")
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return None

class TestURLClassifier(unittest.TestCase):
    """Test URL classification functionality"""

    def setUp(self):
        self.classifier = URLClassifier()

    def test_classify_huggingface_model(self):
        """Test 1: Classification of HuggingFace model URL"""
        url = "https://huggingface.co/google/gemma-3-270m"
        result = self.classifier.classify_url(url)
        self.assertEqual(result, URLType.MODEL)

    def test_classify_huggingface_dataset(self):
        """Test 2: Classification of HuggingFace dataset URL"""
        url = "https://huggingface.co/datasets/xlangai/AgentNet"
        result = self.classifier.classify_url(url)
        self.assertEqual(result, URLType.DATASET)

    def test_classify_github_code(self):
        """Test 3: Classification of GitHub repository URL"""
        url = "https://github.com/SkyworkAI/Matrix-Game"
        result = self.classifier.classify_url(url)
        self.assertEqual(result, URLType.CODE)

    def test_classify_unknown_url(self):
        """Test 4: Classification of unknown URL"""
        url = "https://example.com/some/path"
        result = self.classifier.classify_url(url)
        self.assertEqual(result, URLType.UNKNOWN)

    def test_group_urls_by_type(self):
        """Test 5: Grouping multiple URLs by type"""
        urls = [
            "https://huggingface.co/google/gemma-3-270m",
            "https://huggingface.co/datasets/xlangai/AgentNet",
            "https://github.com/SkyworkAI/Matrix-Game",
        ]
        result = self.classifier.group_urls_by_type(urls)

        self.assertEqual(len(result[URLType.MODEL]), 1)
        self.assertEqual(len(result[URLType.DATASET]), 1)
        self.assertEqual(len(result[URLType.CODE]), 1)
        self.assertEqual(len(result[URLType.UNKNOWN]), 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
