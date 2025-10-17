from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import tempfile
import subprocess
import os
import logging
import shutil


class BaseResourceHandler(ABC):
    """Base class for handling different types of resources"""

    def __init__(self, url: str):
        self.url = url
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cached_data: Dict[str, Any] = {}

    @abstractmethod
    def get_license_score(self) -> float:
        """Get license compatibility score"""
        pass

    @abstractmethod
    def get_documentation_score(self) -> float:
        """Get documentation quality score"""
        pass

    @abstractmethod
    def get_contributor_count(self) -> int:
        """Get number of contributors"""
        pass

    def _cache_get(self, key: str) -> Any:
        """Get cached data"""
        return self._cached_data.get(key)

    def _cache_set(self, key: str, value: Any) -> None:
        """Set cached data"""
        self._cached_data[key] = value

    def _clone_repository(self, clone_url: str) -> Optional[str]:
        """Clone repository to temporary directory and return path"""
        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp(prefix='repo_clone_')
            clone_cmd = ['git', 'clone', clone_url, temp_dir]

            self.logger.info(f"Cloning repository: {clone_url}")
            result = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                self.logger.error(f"Git clone failed: {result.stderr}")
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                return None

            return temp_dir

        except subprocess.TimeoutExpired:
            self.logger.error("Repository clone timed out")
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return None
        except Exception as e:
            self.logger.error(f"Error cloning repository: {e}")
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return None

    def _parse_license_identifier(self, license_id: str) -> float:
        """Parse license information from SPDX identifier or license name"""
        if not license_id:
            return 0.0

        license_lower = license_id.lower().strip()

        # Accepted SPDX license identifiers and common names
        accepted_licenses = {
            # MIT
            'mit', 'mit license',
            # Apache
            'apache', 'apache-2.0', 'apache 2.0', 'apache license',
            # BSD variants
            'bsd', 'bsd-2-clause', 'bsd-3-clause', 'bsd license',
            # GPL variants
            'gpl-2.0', 'gpl-2.0-only', 'gpl-2.0-or-later', 'gplv2',
            'gpl-3.0', 'gpl-3.0-only', 'gpl-3.0-or-later', 'gplv3',
            # LGPL variants
            'lgpl-2.1', 'lgpl-2.1-only', 'lgpl-2.1-or-later', 'lgplv2.1',
            'lgpl-3.0', 'lgpl-3.0-only', 'lgpl-3.0-or-later', 'lgplv3',
            # Creative Commons
            'cc0', 'cc0-1.0', 'creative commons zero',
            # Other permissive licenses
            'unlicense', 'public domain'
        }

        return 1.0 if license_lower in accepted_licenses else 0.0

    def _parse_license_from_text(self, text: str) -> float:
        """Parse license information from text content (fallback method)"""
        if not text:
            return 0.0

        text_lower = text.lower()

        # Accepted licenses: MIT, Apache, BSD, GPLv2, LGPLv2.1, LGPLv3, CC0
        accepted_licenses = [
            'mit license', 'mit',
            'apache license', 'apache', 'apache-2.0', 'apache 2.0',
            'bsd license', 'bsd',
            'gpl-2.0', 'gplv2', 'gpl v2', 'gnu general public license version 2',
            'lgpl-2.1', 'lgplv2.1', 'lgpl v2.1',
            'lgpl-3.0', 'lgplv3', 'lgpl v3',
            'cc0', 'cc0-1.0', 'creative commons zero'
        ]

        # Check if any accepted license is mentioned
        if any(pattern in text_lower for pattern in accepted_licenses):
            return 1.0
        else:
            return 0.0
