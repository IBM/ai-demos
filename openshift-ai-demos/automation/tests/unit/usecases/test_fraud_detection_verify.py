"""Unit tests for rhoai.usecases.fraud_detection.verify."""

from unittest.mock import MagicMock, patch

import pytest

from rhoai.platform.verify import CheckResult

# Import the module via sys.modules to avoid the name clash with the `verify`
# function re-exported through fraud_detection/__init__.py.
import sys
import importlib
importlib.import_module("rhoai.usecases.fraud_detection.verify")
_verify_mod = sys.modules["rhoai.usecases.fraud_detection.verify"]


def _config() -> dict:
    """Minimal config that satisfies fraud_detection.verify()."""
    return {
        "repo_root": "/repo",
        "platform": {"namespace": "test-ns"},
        "deployment": {"models": [], "namespace": "test-ns"},
        "dsc":      {"name": "default-dsc", "dsci_name": "default-dsci"},
        "operator": {"name": "rhods-operator", "namespace": "redhat-ods-operator"},
        "timeouts": {"trustyai_ready": 300},
    }


# ---------------------------------------------------------------------------
# Platform check failure propagation
# ---------------------------------------------------------------------------

_MODULE = "rhoai.usecases.fraud_detection.verify"


class TestVerifyPlatformFailurePropagation:
    """verify() must raise when verify_platform() returns any failed CheckResult."""

    def test_raises_when_single_check_fails(self) -> None:
        mock_pv = MagicMock()
        mock_pv.verify_platform.return_value = [
            CheckResult(name="RHOAI Operator", passed=False, message="CSV not found"),
            CheckResult(name="DSCInitialization", passed=True),
            CheckResult(name="DataScienceCluster", passed=True),
        ]
        with patch(f"{_MODULE}.platform_verify", mock_pv):
            with pytest.raises(RuntimeError, match="Platform checks failed"):
                _verify_mod.verify(_config())

    def test_error_message_includes_failed_check_name(self) -> None:
        mock_pv = MagicMock()
        mock_pv.verify_platform.return_value = [
            CheckResult(name="RHOAI Operator", passed=False, message="CSV not found"),
            CheckResult(name="DSCInitialization", passed=True),
            CheckResult(name="DataScienceCluster", passed=True),
        ]
        with patch(f"{_MODULE}.platform_verify", mock_pv):
            with pytest.raises(RuntimeError, match="RHOAI Operator"):
                _verify_mod.verify(_config())

    def test_error_message_includes_failure_reason(self) -> None:
        mock_pv = MagicMock()
        mock_pv.verify_platform.return_value = [
            CheckResult(name="RHOAI Operator", passed=False, message="CSV not found"),
            CheckResult(name="DSCInitialization", passed=True),
            CheckResult(name="DataScienceCluster", passed=True),
        ]
        with patch(f"{_MODULE}.platform_verify", mock_pv):
            with pytest.raises(RuntimeError, match="CSV not found"):
                _verify_mod.verify(_config())

    def test_raises_when_multiple_checks_fail(self) -> None:
        mock_pv = MagicMock()
        mock_pv.verify_platform.return_value = [
            CheckResult(name="RHOAI Operator",     passed=False, message="not installed"),
            CheckResult(name="DSCInitialization",  passed=False, message="not ready"),
            CheckResult(name="DataScienceCluster", passed=True),
        ]
        with patch(f"{_MODULE}.platform_verify", mock_pv):
            with pytest.raises(RuntimeError) as exc_info:
                _verify_mod.verify(_config())
        msg = str(exc_info.value)
        assert "RHOAI Operator" in msg
        assert "DSCInitialization" in msg

    def test_does_not_raise_when_all_checks_pass(self) -> None:
        mock_pv = MagicMock()
        mock_pv.verify_platform.return_value = [
            CheckResult(name="RHOAI Operator",     passed=True),
            CheckResult(name="DSCInitialization",  passed=True),
            CheckResult(name="DataScienceCluster", passed=True),
        ]
        with patch(f"{_MODULE}.platform_verify", mock_pv), \
             patch(f"{_MODULE}.inference",        MagicMock()), \
             patch(f"{_MODULE}.trustyai",         MagicMock()), \
             patch(f"{_MODULE}.trustyai_client",  MagicMock()), \
             patch(f"{_MODULE}.print_summary"):
            _verify_mod.verify(_config())  # must not raise
