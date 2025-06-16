"""
Tests for source modules.
"""

import pytest


class TestSourcesModule:
    """Test cases for sources module imports."""

    def test_sources_init_import(self):
        """Test that sources module can be imported."""
        try:
            import nbedr.core.sources

            assert hasattr(nbedr.core.sources, "__name__")
        except ImportError:
            pytest.fail("Failed to import nbedr.core.sources")

    def test_factory_import(self):
        """Test that factory module can be imported."""
        try:
            import nbedr.core.sources.factory

            assert hasattr(nbedr.core.sources.factory, "__name__")
        except ImportError:
            pytest.fail("Failed to import nbedr.core.sources.factory")

    def test_base_import(self):
        """Test that base module can be imported."""
        try:
            import nbedr.core.sources.base

            assert hasattr(nbedr.core.sources.base, "__name__")
        except ImportError:
            pytest.fail("Failed to import nbedr.core.sources.base")

    def test_local_import(self):
        """Test that local module can be imported."""
        try:
            import nbedr.core.sources.local

            assert hasattr(nbedr.core.sources.local, "__name__")
        except ImportError:
            pytest.fail("Failed to import nbedr.core.sources.local")
