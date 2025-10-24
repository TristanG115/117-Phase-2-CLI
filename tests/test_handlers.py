import unittest
from unittest.mock import Mock, patch

from resource_handlers import CodeHandler, DatasetHandler, ModelHandler


class TestResourceHandlers(unittest.TestCase):
    """Test resource handler functionality"""

    def test_model_handler_initialization(self):
        """Test 6: ModelHandler initialization"""
        url = "https://huggingface.co/google/gemma-3-270m"
        handler = ModelHandler(url)
        self.assertEqual(handler.url, url)
        self.assertEqual(handler.model_id, "google/gemma-3-270m")

    def test_dataset_handler_initialization(self):
        """Test 7: DatasetHandler initialization"""
        url = "https://huggingface.co/datasets/xlangai/AgentNet"
        handler = DatasetHandler(url)
        self.assertEqual(handler.url, url)
        self.assertEqual(handler.dataset_id, "xlangai/AgentNet")

    def test_code_handler_initialization(self):
        """Test 8: CodeHandler initialization"""
        url = "https://github.com/SkyworkAI/Matrix-Game"
        handler = CodeHandler(url)
        self.assertEqual(handler.url, url)
        self.assertEqual(handler.repo_path, "SkyworkAI/Matrix-Game")

    @patch("requests.get")
    def test_model_handler_api_call(self, mock_get):
        """Test 9: ModelHandler API interaction"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"downloads": 1000, "likes": 50}
        mock_get.return_value = mock_response

        handler = ModelHandler("https://huggingface.co/google/gemma-3-270m")
        data = handler.get_huggingface_api_data()

        self.assertEqual(data["downloads"], 1000)
        self.assertEqual(data["likes"], 50)

    @patch("requests.get")
    def test_code_handler_api_call(self, mock_get):
        """Test 10: CodeHandler API interaction"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"stargazers_count": 100}
        mock_get.return_value = mock_response

        handler = CodeHandler("https://github.com/SkyworkAI/Matrix-Game")
        data = handler.get_github_api_data()

        self.assertEqual(data["stargazers_count"], 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
