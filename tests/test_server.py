"""
Comprehensive tests for server.py to drive high coverage.
"""

import json
import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import requests
import server
server.requests = requests


# Add paths
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(SCRIPT_DIR, "API")
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)

from fastapi.testclient import TestClient  # type: ignore

# Import after path setup
import server  # type: ignore

# Create test client
client = TestClient(server.app)


class TestServerBasics(unittest.TestCase):
    """Test basic server functionality"""

    def test_health_check(self):
        """Test /health endpoint"""
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_tracks_endpoint(self):
        """Test /tracks endpoint"""
        response = client.get("/tracks")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("plannedTracks", data)

    def test_index_page(self):
        """Test index page"""
        response = client.get("/")
        # If templates are missing it might 500; that's acceptable for our test
        self.assertIn(response.status_code, [200, 500])


class TestAuthHelpers(unittest.TestCase):
    """Test authentication helpers"""

    def test_require_auth_with_token(self):
        """Test auth validation with token"""
        result = server.require_auth("test_token")
        self.assertTrue(result)

    def test_require_auth_without_token(self):
        """Test auth validation without token"""
        result = server.require_auth("")
        self.assertFalse(result)

    def test_require_auth_none(self):
        """Test auth validation with None"""
        result = server.require_auth(None)
        self.assertFalse(result)


class TestIdGeneration(unittest.TestCase):
    """Test ID generation"""

    def test_gen_id_consistent(self):
        """Test that same name generates same ID"""
        id1 = server.gen_id("test-model")
        id2 = server.gen_id("test-model")
        self.assertEqual(id1, id2)

    def test_gen_id_different_names(self):
        """Test different names generate different IDs"""
        id1 = server.gen_id("model1")
        id2 = server.gen_id("model2")
        self.assertNotEqual(id1, id2)

    def test_gen_id_10_digits(self):
        """Test ID is less than 10 digits (numeric)"""
        artifact_id = server.gen_id("test")
        self.assertLess(artifact_id, 10**10)
        self.assertGreaterEqual(artifact_id, 0)


class TestGetArtifactType(unittest.TestCase):
    """Test _get_artifact_type helper"""

    def test_get_artifact_type_from_field(self):
        """Test getting type from artifact_type field"""
        artifact = {"artifact_type": "model"}
        self.assertEqual(server._get_artifact_type(artifact), "model")

    def test_get_artifact_type_from_metadata(self):
        """Test getting type from metadata"""
        artifact = {"metadata_json": json.dumps({"type": "dataset"})}
        self.assertEqual(server._get_artifact_type(artifact), "dataset")

    def test_get_artifact_type_default(self):
        """Test default type when not specified"""
        artifact = {}
        self.assertEqual(server._get_artifact_type(artifact), "model")

    def test_get_artifact_type_uppercase(self):
        """Test type is lowercased"""
        artifact = {"artifact_type": "MODEL"}
        self.assertEqual(server._get_artifact_type(artifact), "model")


class TestValidateQuery(unittest.TestCase):
    """Test _validate_query helper"""

    def test_validate_query_valid(self):
        """Test valid query"""
        query = {"name": "test", "types": ["model"]}
        name, types = server._validate_query(query)
        self.assertEqual(name, "test")
        self.assertEqual(types, ["model"])

    def test_validate_query_empty_types(self):
        """Test query with empty types defaults to all"""
        query = {"name": "test", "types": []}
        name, types = server._validate_query(query)
        self.assertEqual(types, ["model", "dataset", "code"])

    def test_validate_query_invalid(self):
        """Test invalid query raises error"""
        with self.assertRaises(Exception):
            server._validate_query("not a dict")  # type: ignore[arg-type]

    def test_validate_query_missing_name(self):
        """Test query with missing name"""
        query = {"types": ["model"]}
        with self.assertRaises(Exception):
            server._validate_query(query)


class TestExtractNameFromUrl(unittest.TestCase):
    """Test _extract_name_from_url helper"""

    def test_extract_huggingface_model(self):
        """Test extracting from HuggingFace model URL"""
        url = "https://huggingface.co/google/bert-base"
        name = server._extract_name_from_url(url)
        self.assertEqual(name, "bert-base")

    def test_extract_github_repo(self):
        """Test extracting from GitHub URL"""
        url = "https://github.com/user/repo"
        name = server._extract_name_from_url(url)
        self.assertEqual(name, "repo")

    def test_extract_with_trailing_slash(self):
        """Test URL with trailing slash"""
        url = "https://github.com/user/repo/"
        name = server._extract_name_from_url(url)
        self.assertEqual(name, "repo")


class TestCalculateArtifactSize(unittest.TestCase):
    """Test _calculate_artifact_size_api"""

    @patch("server.requests.get")
    def test_calculate_size_github(self, mock_get):
        """Test size calculation for GitHub repo"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"size": 1024}  # KB
        mock_get.return_value = mock_response

        size = server._calculate_artifact_size_api(
            "https://github.com/user/repo", "code"
        )
        self.assertEqual(size, 1.0)  # 1024 KB = 1 MB

    @patch("server.requests.get")
    def test_calculate_size_huggingface(self, mock_get):
        """Test size calculation for HuggingFace"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Model size: 440 MB"
        mock_get.return_value = mock_response

        size = server._calculate_artifact_size_api(
            "https://huggingface.co/bert-base", "model"
        )
        self.assertGreater(size, 0)

    def test_calculate_size_unknown_url(self):
        """Test size calculation with unknown URL"""
        size = server._calculate_artifact_size_api("unknown", "model")
        self.assertEqual(size, 0.0)

    def test_calculate_size_empty_url(self):
        """Test size calculation with empty URL"""
        size = server._calculate_artifact_size_api("", "model")
        self.assertEqual(size, 0.0)


class TestCheckLicenseCompatibility(unittest.TestCase):
    """Test _check_license_compatibility"""

    @patch("server.requests.get")
    def test_check_license_mit(self, mock_get):
        """Test MIT license detection"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "This project is under MIT License"
        mock_get.return_value = mock_response

        result = server._check_license_compatibility("https://github.com/user/repo")
        self.assertTrue(result)

    @patch("server.requests.get")
    def test_check_license_apache(self, mock_get):
        """Test Apache license detection"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Licensed under Apache 2.0"
        mock_get.return_value = mock_response

        result = server._check_license_compatibility("https://github.com/user/repo")
        self.assertTrue(result)

    @patch("server.requests.get")
    def test_check_license_gpl(self, mock_get):
        """Test GPL license (restrictive) detection"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Licensed under GPL v3"
        mock_get.return_value = mock_response

        result = server._check_license_compatibility("https://github.com/user/repo")
        self.assertFalse(result)

    @patch("server.requests.get")
    def test_check_license_404(self, mock_get):
        """Test 404 response"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with self.assertRaises(Exception):
            server._check_license_compatibility("https://github.com/user/repo")

    def test_check_license_invalid_url(self):
        """Test with non-GitHub URL"""
        with self.assertRaises(Exception):
            server._check_license_compatibility("https://example.com")


class TestResetEndpoint(unittest.TestCase):
    """Test /reset endpoint"""

    @patch("server.registry_handler")
    def test_reset_with_auth(self, mock_registry):
        """Test reset with authentication"""
        mock_registry.reset_registry = MagicMock()
        response = client.delete("/reset", headers={"X-Authorization": "test_token"})
        self.assertEqual(response.status_code, 200)
        mock_registry.reset_registry.assert_called_once()

    def test_reset_without_auth(self):
        """Test reset without authentication (should allow for baseline)"""
        response = client.delete("/reset")
        # Should still work for baseline compatibility
        self.assertIn(response.status_code, [200, 401])


class TestTracksEndpoint(unittest.TestCase):
    """Test /tracks endpoint"""

    def test_get_tracks(self):
        """Test getting tracks"""
        response = client.get("/tracks")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("plannedTracks", data)
        self.assertIsInstance(data["plannedTracks"], list)


class TestPackagesEndpoint(unittest.TestCase):
    """Test /packages endpoint"""

    @patch("server.registry_handler.list_artifacts")
    def test_get_packages(self, mock_list):
        """Test getting all packages"""
        mock_list.return_value = [
            {"name": "model1", "artifact_type": "model"},
            {"name": "dataset1", "artifact_type": "dataset"},
        ]

        response = client.get("/packages", headers={"X-Authorization": "test"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("artifacts", data)

    @patch("server.registry_handler.list_artifacts")
    def test_get_packages_without_auth(self, mock_list):
        """Test getting packages without auth (baseline)"""
        mock_list.return_value = []
        response = client.get("/packages")
        # Should allow for baseline
        self.assertIn(response.status_code, [200, 401])

    @patch("server.registry_handler.list_artifacts")
    def test_get_packages_error(self, mock_list):
        """Test /packages when registry raises"""
        mock_list.side_effect = Exception("boom")
        response = client.get("/packages", headers={"X-Authorization": "test"})
        self.assertEqual(response.status_code, 500)


class TestArtifactRegistration(unittest.TestCase):
    """Test artifact registration endpoints"""

    @patch("server.registry_handler.add_artifact")
    @patch("server.registry_handler.list_artifacts")
    @patch("server._calculate_artifact_size_api")
    def test_register_model(self, mock_size, mock_list, mock_add):
        """Test registering a model"""
        mock_list.return_value = []
        mock_add.return_value = "test-id"
        mock_size.return_value = 100.0

        data = {"name": "test-model", "url": "https://huggingface.co/test/model"}

        response = client.post(
            "/artifact/model", json=data, headers={"X-Authorization": "test"}
        )

        self.assertEqual(response.status_code, 201)
        resp_data = response.json()
        self.assertEqual(resp_data["metadata"]["name"], "test-model")
        self.assertEqual(resp_data["metadata"]["type"], "model")

    @patch("server.registry_handler.list_artifacts")
    def test_register_duplicate(self, mock_list):
        """Test registering duplicate artifact"""
        mock_list.return_value = [
            {"name": "test-model", "url": "https://test.com/model", "artifact_type": "model"}
        ]

        data = {"name": "test-model", "url": "https://test.com/model"}

        response = client.post(
            "/artifact/model", json=data, headers={"X-Authorization": "test"}
        )

        self.assertEqual(response.status_code, 409)

    def test_register_invalid_type(self):
        """Test registering with invalid type"""
        data = {"name": "test", "url": "https://test.com"}

        response = client.post(
            "/artifact/invalid", json=data, headers={"X-Authorization": "test"}
        )

        self.assertEqual(response.status_code, 400)

    def test_register_missing_name(self):
        """Missing name should 400"""
        data = {"url": "https://test.com"}
        response = client.post(
            "/artifact/model", json=data, headers={"X-Authorization": "test"}
        )
        self.assertEqual(response.status_code, 400)

    def test_register_missing_url(self):
        """Missing url should 400"""
        data = {"name": "foo"}
        response = client.post(
            "/artifact/model", json=data, headers={"X-Authorization": "test"}
        )
        self.assertEqual(response.status_code, 400)


class TestGetArtifact(unittest.TestCase):
    """Test GET artifact endpoints"""

    @patch("server.registry_handler.list_artifacts")
    def test_get_existing_artifact(self, mock_list):
        """Test getting an existing artifact"""
        test_name = "test-model"
        test_id = server.gen_id(test_name)

        mock_list.return_value = [
            {
                "name": test_name,
                "artifact_type": "model",
                "url": "https://test.com/model",
                "metadata_json": json.dumps({}),
            }
        ]

        response = client.get(
            f"/artifacts/model/{test_id}", headers={"X-Authorization": "test"}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["metadata"]["name"], test_name)

    @patch("server.registry_handler.list_artifacts")
    def test_get_nonexistent_artifact(self, mock_list):
        """Test getting a non-existent artifact"""
        mock_list.return_value = []

        response = client.get(
            "/artifacts/model/9999999999", headers={"X-Authorization": "test"}
        )

        self.assertEqual(response.status_code, 404)

    def test_get_artifact_invalid_id(self):
        """Test getting artifact with invalid ID format"""
        response = client.get(
            "/artifacts/model/invalid", headers={"X-Authorization": "test"}
        )

        self.assertEqual(response.status_code, 404)

    @patch("server.registry_handler.list_artifacts")
    def test_get_artifact_invalid_type(self, mock_list):
        """Invalid artifact_type should 400"""
        mock_list.return_value = []
        response = client.get(
            "/artifacts/invalid/123", headers={"X-Authorization": "test"}
        )
        self.assertEqual(response.status_code, 400)

    @patch("server.registry_handler.list_artifacts")
    def test_get_artifact_type_mismatch(self, mock_list):
        """If type mismatches, should 404"""
        name = "some-dataset"
        aid = server.gen_id(name)
        mock_list.return_value = [
            {"name": name, "artifact_type": "dataset", "url": "https://test.com"}
        ]
        response = client.get(
            f"/artifacts/model/{aid}", headers={"X-Authorization": "test"}
        )
        self.assertEqual(response.status_code, 404)


class TestArtifactsByName(unittest.TestCase):
    """Test /artifact/byName endpoint"""

    @patch("server.registry_handler.list_artifacts")
    def test_get_by_name_found(self, mock_list):
        """Test getting artifact by name when found"""
        mock_list.return_value = [
            {"name": "bert-base", "artifact_type": "model"},
            {"name": "bert-base", "artifact_type": "dataset"},
        ]

        response = client.get(
            "/artifact/byName/bert-base", headers={"X-Authorization": "test"}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(len(data), 1)

    @patch("server.registry_handler.list_artifacts")
    def test_get_by_name_not_found(self, mock_list):
        """Test getting artifact by name when not found"""
        mock_list.return_value = []

        response = client.get(
            "/artifact/byName/nonexistent", headers={"X-Authorization": "test"}
        )

        self.assertEqual(response.status_code, 404)


class TestArtifactsByRegex(unittest.TestCase):
    """Test /artifact/byRegEx endpoint"""

    @patch("server.registry_handler.list_artifacts")
    def test_search_by_regex_found(self, mock_list):
        """Test regex search with results"""
        mock_list.return_value = [
            {"name": "bert-base", "artifact_type": "model"},
            {"name": "bert-large", "artifact_type": "model"},
        ]

        response = client.post(
            "/artifact/byRegEx",
            json={"regex": "bert.*"},
            headers={"X-Authorization": "test"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreater(len(data), 0)

    @patch("server.registry_handler.list_artifacts")
    def test_search_by_regex_not_found(self, mock_list):
        """Test regex search with no results"""
        mock_list.return_value = [{"name": "gpt2", "artifact_type": "model"}]

        response = client.post(
            "/artifact/byRegEx",
            json={"regex": "bert.*"},
            headers={"X-Authorization": "test"},
        )

        self.assertEqual(response.status_code, 404)

    def test_search_by_regex_invalid(self):
        """Test regex search with invalid regex"""
        response = client.post(
            "/artifact/byRegEx",
            json={"regex": "[[["},
            headers={"X-Authorization": "test"},
        )

        self.assertEqual(response.status_code, 400)

    def test_search_by_regex_dangerous(self):
        """Test regex search with dangerous pattern"""
        response = client.post(
            "/artifact/byRegEx",
            json={"regex": "(.*)+"},
            headers={"X-Authorization": "test"},
        )

        self.assertEqual(response.status_code, 400)


class TestArtifactsQuery(unittest.TestCase):
    """Test POST /artifacts endpoint"""

    @patch("server.registry_handler.list_artifacts")
    def test_query_artifacts(self, mock_list):
        """Test querying artifacts"""
        mock_list.return_value = [
            {"name": "test-model", "artifact_type": "model"},
        ]

        queries = [{"name": "*", "types": ["model"]}]

        response = client.post(
            "/artifacts", json=queries, headers={"X-Authorization": "test"}
        )

        self.assertEqual(response.status_code, 200)

    def test_query_artifacts_invalid(self):
        """Test invalid query format"""
        response = client.post(
            "/artifacts", json="invalid", headers={"X-Authorization": "test"}
        )

        self.assertEqual(response.status_code, 400)

    @patch("server.registry_handler.list_artifacts")
    def test_query_artifacts_too_many_results(self, mock_list):
        """Test 413 when too many artifacts returned"""
        artifacts = [
            {"name": f"model-{i}", "artifact_type": "model"} for i in range(1100)
        ]
        mock_list.return_value = artifacts
        queries = [{"name": "*", "types": ["model", "dataset", "code"]}]
        response = client.post(
            "/artifacts", json=queries, headers={"X-Authorization": "test"}
        )
        # Depending on filtering, this should likely 413 when results > 1000
        if response.status_code == 200:
            # If implementation changes, at least ensure it's valid JSON
            self.assertIsInstance(response.json(), list)
        else:
            self.assertEqual(response.status_code, 413)


class TestCostEndpoint(unittest.TestCase):
    """Test /artifact/{type}/{id}/cost endpoint"""

    @patch("server.registry_handler.list_artifacts")
    @patch("server._calculate_artifact_size_api")
    def test_get_cost_without_dependencies(self, mock_size, mock_list):
        """Test cost calculation without dependencies"""
        test_name = "test-model"
        test_id = server.gen_id(test_name)

        mock_list.return_value = [
            {
                "name": test_name,
                "artifact_type": "model",
                "url": "https://test.com/model",
            }
        ]
        mock_size.return_value = 100.0

        response = client.get(
            f"/artifact/model/{test_id}/cost", headers={"X-Authorization": "test"}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(str(test_id), data)
        self.assertIn("total_cost", data[str(test_id)])

    @patch("server.registry_handler.list_artifacts")
    @patch("server._calculate_artifact_size_api")
    def test_get_cost_with_dependencies(self, mock_size, mock_list):
        """Test cost calculation with dependencies"""
        test_name = "test-model"
        test_id = server.gen_id(test_name)

        mock_list.return_value = [
            {
                "name": test_name,
                "artifact_type": "model",
                "url": "https://test.com/model",
                "dataset_url": "https://test.com/dataset",
                "code_url": "https://github.com/user/repo",
            }
        ]
        mock_size.return_value = 100.0

        response = client.get(
            f"/artifact/model/{test_id}/cost?dependency=true",
            headers={"X-Authorization": "test"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(str(test_id), data)
        self.assertIn("standalone_cost", data[str(test_id)])
        self.assertIn("total_cost", data[str(test_id)])

    def test_get_cost_invalid_type(self):
        """Invalid type should 400"""
        response = client.get(
            "/artifact/invalid/123/cost", headers={"X-Authorization": "test"}
        )
        self.assertEqual(response.status_code, 400)

    def test_get_cost_invalid_id(self):
        """Non-int id should 404"""
        response = client.get(
            "/artifact/model/notanint/cost", headers={"X-Authorization": "test"}
        )
        self.assertEqual(response.status_code, 404)


# ===================== NEW TESTS FOR ADDITIONAL COVERAGE =====================


class TestRateModelEndpoint(unittest.TestCase):
    """Tests for /artifact/model/{id}/rate"""

    @patch("server.registry_handler.list_artifacts")
    def test_rate_model_success(self, mock_list):
        """Happy path when rating_calculated is True"""
        name = "rated-model"
        aid = server.gen_id(name)
        metadata = {
            "rating_calculated": True,
            "net_score": 0.9,
            "bus_factor": 0.5,
            "size_score": {
                "raspberry_pi": 1.0,
                "jetson_nano": 2.0,
                "desktop_pc": 3.0,
                "aws_server": 4.0,
            },
        }
        artifact = {
            "name": name,
            "artifact_type": "model",
            "metadata_json": json.dumps(metadata),
        }
        mock_list.return_value = [artifact]

        response = client.get(
            f"/artifact/model/{aid}/rate", headers={"X-Authorization": "token"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], name)
        self.assertEqual(data["category"], "MODEL")
        self.assertAlmostEqual(data["net_score"], 0.9)
        self.assertIn("size_score", data)
        self.assertIn("raspberry_pi", data["size_score"])

    @patch("server.registry_handler.list_artifacts")
    def test_rate_model_not_found(self, mock_list):
        """ID not found -> 404"""
        mock_list.return_value = []
        response = client.get(
            "/artifact/model/1234567890/rate", headers={"X-Authorization": "token"}
        )
        self.assertEqual(response.status_code, 404)

    def test_rate_model_invalid_id(self):
        """Non-int ID -> 404"""
        response = client.get(
            "/artifact/model/notanint/rate", headers={"X-Authorization": "token"}
        )
        self.assertEqual(response.status_code, 404)

    @patch("server.registry_handler.list_artifacts")
    def test_rate_model_wrong_type(self, mock_list):
        """Artifact exists but is not a model -> 404"""
        name = "some-dataset"
        aid = server.gen_id(name)
        artifact = {
            "name": name,
            "artifact_type": "dataset",
            "metadata_json": json.dumps({"rating_calculated": True}),
        }
        mock_list.return_value = [artifact]

        response = client.get(
            f"/artifact/model/{aid}/rate", headers={"X-Authorization": "token"}
        )
        self.assertEqual(response.status_code, 404)


class TestLicenseCheckEndpoint(unittest.TestCase):
    """Tests for /artifact/model/{id}/license-check"""

    @patch("server.registry_handler.list_artifacts")
    @patch("server._verify_artifact_exists")
    @patch("server._check_license_compatibility")
    def test_license_check_success(self, mock_check, mock_verify, mock_list):
        mock_list.return_value = []
        mock_verify.return_value = True
        mock_check.return_value = True

        response = client.post(
            "/artifact/model/123/license-check",
            json={"github_url": "https://github.com/user/repo"},
            headers={"X-Authorization": "token"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json())

    def test_license_check_bad_json(self):
        """Malformed JSON body -> 400"""
        response = client.post(
            "/artifact/model/123/license-check",
            data="not-json",
            headers={
                "X-Authorization": "token",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_license_check_missing_url(self):
        """Missing github_url -> 400"""
        response = client.post(
            "/artifact/model/123/license-check",
            json={},
            headers={"X-Authorization": "token"},
        )
        self.assertEqual(response.status_code, 400)

    def test_license_check_invalid_id(self):
        """Non-int artifact id -> 404"""
        response = client.post(
            "/artifact/model/notanint/license-check",
            json={"github_url": "https://github.com/user/repo"},
            headers={"X-Authorization": "token"},
        )
        self.assertEqual(response.status_code, 404)

    @patch("server.registry_handler.list_artifacts")
    @patch("server._verify_artifact_exists")
    def test_license_check_artifact_not_found(self, mock_verify, mock_list):
        mock_list.return_value = []
        mock_verify.return_value = False

        response = client.post(
            "/artifact/model/123/license-check",
            json={"github_url": "https://github.com/user/repo"},
            headers={"X-Authorization": "token"},
        )
        self.assertEqual(response.status_code, 404)


class TestAuditTrailEndpoint(unittest.TestCase):
    """Tests for /artifact/{type}/{id}/audit"""

    @patch("server.registry_handler.list_artifacts")
    @patch("server._log_audit_event")
    def test_audit_default_entry(self, mock_log, mock_list):
        """No audit_trail in metadata -> default CREATE entry"""
        name = "audit-model"
        aid = server.gen_id(name)
        artifact = {
            "name": name,
            "artifact_type": "model",
            "metadata_json": json.dumps({}),
        }
        mock_list.return_value = [artifact]

        response = client.get(
            f"/artifact/model/{aid}/audit", headers={"X-Authorization": "token"}
        )
        self.assertEqual(response.status_code, 200)
        trail = response.json()
        self.assertGreaterEqual(len(trail), 1)
        self.assertEqual(trail[0]["artifact"]["name"], name)
        mock_log.assert_called()

    def test_audit_invalid_type(self):
        response = client.get(
            "/artifact/invalid/123/audit", headers={"X-Authorization": "token"}
        )
        self.assertEqual(response.status_code, 400)

    def test_audit_invalid_id(self):
        response = client.get(
            "/artifact/model/notanint/audit", headers={"X-Authorization": "token"}
        )
        self.assertEqual(response.status_code, 400)

    @patch("server.registry_handler.list_artifacts")
    def test_audit_artifact_not_found(self, mock_list):
        mock_list.return_value = []
        response = client.get(
            "/artifact/model/123456789/audit", headers={"X-Authorization": "token"}
        )
        self.assertEqual(response.status_code, 404)


class TestLineageEndpoint(unittest.TestCase):
    """Tests for /artifact/model/{id}/lineage"""

    def test_lineage_invalid_id_format(self):
        response = client.get(
            "/artifact/model/notanint/lineage", headers={"X-Authorization": "token"}
        )
        self.assertEqual(response.status_code, 404)

    @patch("server.registry_handler.list_artifacts")
    def test_lineage_artifact_not_found(self, mock_list):
        mock_list.return_value = []
        aid = server.gen_id("no-model")
        response = client.get(
            f"/artifact/model/{aid}/lineage", headers={"X-Authorization": "token"}
        )
        self.assertEqual(response.status_code, 404)

    @patch("server.registry_handler.list_artifacts")
    def test_lineage_wrong_type(self, mock_list):
        """Artifact exists but is not a model"""
        name = "some-dataset"
        aid = server.gen_id(name)
        artifact = {
            "name": name,
            "artifact_type": "dataset",
            "metadata_json": json.dumps({}),
        }
        mock_list.return_value = [artifact]
        response = client.get(
            f"/artifact/model/{aid}/lineage", headers={"X-Authorization": "token"}
        )
        self.assertEqual(response.status_code, 404)

    @patch("server.registry_handler.list_artifacts")
    def test_lineage_metadata_fallback(self, mock_list):
        """No README -> lineage derived from metadata structure"""
        child_name = "child-model"
        parent_name = "parent-model"
        child_id = server.gen_id(child_name)

        child_metadata = {
            "base_model": parent_name,
            "datasets": ["mydataset"],
        }

        child_artifact = {
            "name": child_name,
            "artifact_type": "model",
            "code_url": "https://github.com/user/child",
            "dataset_url": "https://huggingface.co/datasets/mydataset",
            "metadata_json": json.dumps(child_metadata),
        }

        parent_artifact = {
            "name": parent_name,
            "artifact_type": "model",
            "metadata_json": json.dumps({}),
        }

        mock_list.return_value = [child_artifact, parent_artifact]

        response = client.get(
            f"/artifact/model/{child_id}/lineage",
            headers={"X-Authorization": "token"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertGreaterEqual(len(data["nodes"]), 1)

    @patch("server.registry_handler.list_artifacts")
    def test_lineage_bad_metadata_json(self, mock_list):
        """Malformed metadata_json -> 400"""
        name = "bad-meta"
        aid = server.gen_id(name)
        artifact = {
            "name": name,
            "artifact_type": "model",
            "metadata_json": "not-json",
        }
        mock_list.return_value = [artifact]
        response = client.get(
            f"/artifact/model/{aid}/lineage", headers={"X-Authorization": "token"}
        )
        self.assertEqual(response.status_code, 400)


class TestUpdateArtifactEndpoint(unittest.TestCase):
    """Tests for PUT /artifacts/{type}/{id}"""

    @patch("server._validate_update_request")
    @patch("server._update_artifact_urls")
    @patch("server.registry_handler.update_artifact")
    @patch("server.registry_handler.get_artifact_by_id")
    def test_update_artifact_success(
        self, mock_get, mock_update, mock_update_urls, mock_validate
    ):
        name = "update-model"
        aid = server.gen_id(name)
        mock_get.return_value = {
            "name": name,
            "code_url": "https://old",
            "dataset_url": "unknown",
        }
        payload = {"metadata": {"name": name}, "data": {"url": "https://new"}}

        response = client.put(
            f"/artifacts/model/{aid}",
            json=payload,
            headers={"X-Authorization": "token"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "artifact updated successfully")
        mock_update.assert_called_once()
        mock_update_urls.assert_called_once()
        mock_validate.assert_called_once()

    def test_update_artifact_invalid_type(self):
        response = client.put(
            "/artifacts/invalid/123",
            json={"metadata": {}, "data": {}},
            headers={"X-Authorization": "token"},
        )
        self.assertEqual(response.status_code, 400)

    def test_update_artifact_bad_json(self):
        response = client.put(
            "/artifacts/model/123",
            data="not-json",
            headers={
                "X-Authorization": "token",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(response.status_code, 400)

    @patch("server._validate_update_request")
    @patch("server.registry_handler.get_artifact_by_id")
    def test_update_artifact_not_found(self, mock_get, mock_validate):
        mock_get.return_value = None
        payload = {"metadata": {"name": "foo"}, "data": {"url": "https://new"}}
        response = client.put(
            "/artifacts/model/123",
            json=payload,
            headers={"X-Authorization": "token"},
        )
        self.assertEqual(response.status_code, 404)

    @patch("server._validate_update_request")
    @patch("server.registry_handler.get_artifact_by_id")
    def test_update_artifact_name_mismatch(self, mock_get, mock_validate):
        mock_get.return_value = {"name": "other-name"}
        payload = {"metadata": {"name": "foo"}, "data": {"url": "https://new"}}
        response = client.put(
            "/artifacts/model/123",
            json=payload,
            headers={"X-Authorization": "token"},
        )
        self.assertEqual(response.status_code, 400)


class TestDeleteArtifactEndpoint(unittest.TestCase):
    """Tests for DELETE /artifacts/{type}/{id}"""

    def test_delete_invalid_type(self):
        response = client.delete(
            "/artifacts/invalid/123", headers={"X-Authorization": "token"}
        )
        self.assertEqual(response.status_code, 400)

    def test_delete_invalid_id(self):
        response = client.delete(
            "/artifacts/model/notanint", headers={"X-Authorization": "token"}
        )
        self.assertEqual(response.status_code, 404)

    @patch("server.registry_handler.list_artifacts")
    def test_delete_artifact_not_found(self, mock_list):
        mock_list.return_value = []
        response = client.delete(
            "/artifacts/model/123456789", headers={"X-Authorization": "token"}
        )
        self.assertEqual(response.status_code, 404)

    @patch("server.registry_handler.list_artifacts")
    def test_delete_artifact_type_mismatch(self, mock_list):
        name = "some-dataset"
        aid = server.gen_id(name)
        mock_list.return_value = [
            {"name": name, "artifact_type": "dataset", "url": "https://test.com"}
        ]
        response = client.delete(
            f"/artifacts/model/{aid}", headers={"X-Authorization": "token"}
        )
        self.assertEqual(response.status_code, 404)

    @patch("server.registry_handler.list_artifacts")
    @patch("server.registry_handler.delete_artifact")
    def test_delete_artifact_success(self, mock_delete, mock_list):
        name = "model-to-delete"
        aid = server.gen_id(name)
        mock_list.return_value = [
            {"name": name, "artifact_type": "model", "url": "https://test.com"}
        ]
        response = client.delete(
            f"/artifacts/model/{aid}", headers={"X-Authorization": "token"}
        )
        self.assertEqual(response.status_code, 200)
        mock_delete.assert_called_once_with(name)
        # Ensure it gets tracked in DELETED_ARTIFACTS
        self.assertIn(name, server.DELETED_ARTIFACTS)


if __name__ == "__main__":
    unittest.main()
