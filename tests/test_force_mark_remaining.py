import pytest


def test_force_mark_remaining_disabled():
    pytest.skip("Mechanical force-marking disabled — using real tests only")
