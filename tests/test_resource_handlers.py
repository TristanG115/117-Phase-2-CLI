import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import subprocess

from handlers.base_resource_handler import BaseResourceHandler


class ConcreteHandler(BaseResourceHandler):
    """Concrete implementation for testing abstract base class"""
    
    def get_license_score(self) -> float:
        return 1.0
    
    def get_documentation_score(self) -> float:
        return 0.8
    
    def get_contributor_count(self) -> int:
        return 5


class TestBaseResourceHandler(unittest.TestCase):
    """Test base resource handler functionality"""

    def setUp(self):
        self.handler = ConcreteHandler("https://example.com/test")

    def test_initialization(self):
        """Test handler initialization"""
        self.assertEqual(self.handler.url, "https://example.com/test")
        self.assertEqual(self.handler._cached_data, {})

    def test_cache_operations(self):
        """Test cache get and set operations"""
        # Initially empty
        self.assertIsNone(self.handler._cache_get("test_key"))
        
        # Set and retrieve
        self.handler._cache_set("test_key", "test_value")
        self.assertEqual(self.handler._cache_get("test_key"), "test_value")
        
        # Set multiple values
        self.handler._cache_set("key2", 42)
        self.assertEqual(self.handler._cache_get("key2"), 42)
        self.assertEqual(self.handler._cache_get("test_key"), "test_value")

    def test_parse_license_identifier_mit(self):
        """Test MIT license detection"""
        self.assertEqual(self.handler._parse_license_identifier("MIT"), 1.0)
        self.assertEqual(self.handler._parse_license_identifier("mit"), 1.0)
        self.assertEqual(self.handler._parse_license_identifier("MIT License"), 1.0)

    def test_parse_license_identifier_apache(self):
        """Test Apache license detection"""
        self.assertEqual(self.handler._parse_license_identifier("apache-2.0"), 1.0)
        self.assertEqual(self.handler._parse_license_identifier("Apache 2.0"), 1.0)
        self.assertEqual(self.handler._parse_license_identifier("APACHE"), 1.0)

    def test_parse_license_identifier_bsd(self):
        """Test BSD license detection"""
        self.assertEqual(self.handler._parse_license_identifier("BSD"), 1.0)
        self.assertEqual(self.handler._parse_license_identifier("bsd-3-clause"), 1.0)
        self.assertEqual(self.handler._parse_license_identifier("BSD License"), 1.0)

    def test_parse_license_identifier_gpl(self):
        """Test GPL license detection"""
        self.assertEqual(self.handler._parse_license_identifier("GPL-3.0"), 1.0)
        self.assertEqual(self.handler._parse_license_identifier("gplv3"), 1.0)
        self.assertEqual(self.handler._parse_license_identifier("GPL-2.0"), 1.0)

    def test_parse_license_identifier_lgpl(self):
        """Test LGPL license detection"""
        self.assertEqual(self.handler._parse_license_identifier("LGPL-2.1"), 1.0)
        self.assertEqual(self.handler._parse_license_identifier("lgplv3"), 1.0)

    def test_parse_license_identifier_cc0(self):
        """Test CC0 license detection"""
        self.assertEqual(self.handler._parse_license_identifier("CC0"), 1.0)
        self.assertEqual(self.handler._parse_license_identifier("cc0-1.0"), 1.0)
        self.assertEqual(self.handler._parse_license_identifier("Creative Commons Zero"), 1.0)

    def test_parse_license_identifier_unlicense(self):
        """Test Unlicense detection"""
        self.assertEqual(self.handler._parse_license_identifier("unlicense"), 1.0)
        self.assertEqual(self.handler._parse_license_identifier("Public Domain"), 1.0)

    def test_parse_license_identifier_unknown(self):
        """Test unknown license returns 0"""
        self.assertEqual(self.handler._parse_license_identifier("proprietary"), 0.0)
        self.assertEqual(self.handler._parse_license_identifier("custom"), 0.0)
        self.assertEqual(self.handler._parse_license_identifier(""), 0.0)
        self.assertIsNone(self.handler._parse_license_identifier(None) or None)

    def test_parse_license_from_text_mit(self):
        """Test MIT license detection from text"""
        text = "This project is licensed under the MIT License"
        self.assertEqual(self.handler._parse_license_from_text(text), 1.0)

    def test_parse_license_from_text_apache(self):
        """Test Apache license detection from text"""
        text = "Licensed under Apache 2.0"
        self.assertEqual(self.handler._parse_license_from_text(text), 1.0)
        
        text2 = "This uses the apache license"
        self.assertEqual(self.handler._parse_license_from_text(text2), 1.0)

    def test_parse_license_from_text_bsd(self):
        """Test BSD license detection from text"""
        text = "Released under BSD License"
        self.assertEqual(self.handler._parse_license_from_text(text), 1.0)

    def test_parse_license_from_text_gpl(self):
        """Test GPL license detection from text"""
        text = "Licensed under GNU General Public License version 2"
        self.assertEqual(self.handler._parse_license_from_text(text), 1.0)
        
        text2 = "This is GPL-2.0 software"
        self.assertEqual(self.handler._parse_license_from_text(text2), 1.0)

    def test_parse_license_from_text_lgpl(self):
        """Test LGPL license detection from text"""
        text = "LGPL v2.1 license applies"
        self.assertEqual(self.handler._parse_license_from_text(text), 1.0)
        
        text2 = "lgpl-3.0 licensed"
        self.assertEqual(self.handler._parse_license_from_text(text2), 1.0)

    def test_parse_license_from_text_cc0(self):
        """Test CC0 license detection from text"""
        text = "Released to the public domain under CC0-1.0"
        self.assertEqual(self.handler._parse_license_from_text(text), 1.0)

    def test_parse_license_from_text_not_found(self):
        """Test no license found in text"""
        text = "This is proprietary software"
        self.assertEqual(self.handler._parse_license_from_text(text), 0.0)
        
        self.assertEqual(self.handler._parse_license_from_text(""), 0.0)
        self.assertEqual(self.handler._parse_license_from_text(None), 0.0)

    @patch("subprocess.run")
    def test_clone_repository_success(self, mock_run):
        """Test successful repository clone"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("tempfile.mkdtemp", return_value=temp_dir):
                result = self.handler._clone_repository("https://github.com/test/repo")
                self.assertEqual(result, temp_dir)

    @patch("subprocess.run")
    def test_clone_repository_failure(self, mock_run):
        """Test failed repository clone"""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "fatal: repository not found"
        mock_run.return_value = mock_result
        
        result = self.handler._clone_repository("https://github.com/test/nonexistent")
        self.assertIsNone(result)

    @patch("subprocess.run")
    def test_clone_repository_timeout(self, mock_run):
        """Test repository clone timeout"""
        mock_run.side_effect = subprocess.TimeoutExpired("git", 300)
        
        result = self.handler._clone_repository("https://github.com/test/repo")
        self.assertIsNone(result)

    @patch("subprocess.run")
    def test_clone_repository_exception(self, mock_run):
        """Test repository clone with exception"""
        mock_run.side_effect = Exception("Network error")
        
        result = self.handler._clone_repository("https://github.com/test/repo")
        self.assertIsNone(result)

    def test_abstract_methods_implemented(self):
        """Test that abstract methods are implemented"""
        self.assertEqual(self.handler.get_license_score(), 1.0)
        self.assertEqual(self.handler.get_documentation_score(), 0.8)
        self.assertEqual(self.handler.get_contributor_count(), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)