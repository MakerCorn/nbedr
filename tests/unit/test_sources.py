"""
Tests for source modules.
"""

import pytest


class TestSourcesModule:
    """Test cases for sources module imports."""

    def test_sources_init_import(self):
        """Test that sources module can be imported."""
        try:
            import core.sources
            assert hasattr(core.sources, '__name__')
        except ImportError:
            pytest.fail("Failed to import core.sources")

    def test_factory_import(self):
        """Test that factory module can be imported."""
        try:
            import core.sources.factory
            assert hasattr(core.sources.factory, '__name__')
        except ImportError:
            pytest.fail("Failed to import core.sources.factory")

    def test_base_import(self):
        """Test that base module can be imported."""
        try:
            import core.sources.base
            assert hasattr(core.sources.base, '__name__')
        except ImportError:
            pytest.fail("Failed to import core.sources.base")

    def test_local_import(self):
        """Test that local module can be imported."""
        try:
            import core.sources.local
            assert hasattr(core.sources.local, '__name__')
        except ImportError:
            pytest.fail("Failed to import core.sources.local")