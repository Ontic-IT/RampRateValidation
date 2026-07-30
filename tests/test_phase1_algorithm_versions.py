"""Unit tests for Phase 1: config/algorithm_versions.py."""

from config.algorithm_versions import ALGORITHM_RANDOM_SEED, VERSION


class TestAlgorithmVersions:
    def test_version_is_string(self):
        assert isinstance(VERSION, str)
        assert VERSION == "1.0.0"

    def test_algorithm_random_seed_present(self):
        assert isinstance(ALGORITHM_RANDOM_SEED, int)
        assert ALGORITHM_RANDOM_SEED == 42
