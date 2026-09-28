"""Unit tests for rhoai.usecases.registry."""

import pytest

from rhoai.usecases import registry


class TestGet:
    def test_returns_fraud_detection_module(self) -> None:
        mod = registry.get("fraud-detection")
        assert hasattr(mod, "deploy")
        assert hasattr(mod, "verify")
        assert hasattr(mod, "cleanup")

    def test_raises_for_unknown_use_case(self) -> None:
        with pytest.raises(RuntimeError, match="Unknown use case"):
            registry.get("nonexistent-use-case")

    def test_error_message_lists_available(self) -> None:
        with pytest.raises(RuntimeError, match="fraud-detection"):
            registry.get("not-real")


class TestListAvailable:
    def test_returns_sorted_list(self) -> None:
        names = registry.list_available()
        assert isinstance(names, list)
        assert names == sorted(names)

    def test_contains_fraud_detection(self) -> None:
        assert "fraud-detection" in registry.list_available()
