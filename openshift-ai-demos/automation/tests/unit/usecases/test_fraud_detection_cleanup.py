"""Unit tests for rhoai.usecases.fraud_detection.cleanup."""

import sys
from unittest.mock import MagicMock

import pytest

# Force the module to be imported, then retrieve it from sys.modules so we
# get the actual module object and not the `cleanup` function that __init__.py
# re-exports under the same dotted name.
import rhoai.usecases.fraud_detection.cleanup  # noqa: F401
_cleanup_mod = sys.modules["rhoai.usecases.fraud_detection.cleanup"]

from rhoai.usecases.fraud_detection.cleanup import cleanup  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_config(models: list) -> dict:
    return {
        "_use_case": "fraud-detection",
        "_config_file": "",
        "platform": {"namespace": "rhoai-ns"},
        "deployment": {"namespace": "test-ns", "models": models},
        "timeouts": {},
    }


def _patch_all(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Patch every external call cleanup makes and return the mocks.

    cleanup.py uses `from rhoai.platform import inference, storage, trustyai`
    so the names live as module-level attributes.  We retrieve the actual
    module from sys.modules to avoid the __init__.py re-export collision.
    """
    mocks: dict[str, MagicMock] = {}
    for attr in ("inference", "storage", "trustyai"):
        m = MagicMock()
        monkeypatch.setattr(_cleanup_mod, attr, m)
        mocks[attr] = m
    return mocks


# ---------------------------------------------------------------------------
# Tests: staged-model cleanup (Pod + PVC)
# ---------------------------------------------------------------------------

class TestStagedModelCleanup:
    """When a model has model_path set, cleanup must delete the staging Pod
    then the PVC in that order."""

    def test_calls_delete_staging_pod_for_staged_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mocks = _patch_all(monkeypatch)
        config = _base_config([{
            "name": "m",
            "model_path": "/some/model.onnx",
            "config_path": "/some/config.pbtxt",
        }])

        cleanup(config)

        mocks["storage"].delete_staging_pod.assert_called_once_with(
            "m-pvc", "test-ns"
        )

    def test_calls_delete_pvc_for_staged_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mocks = _patch_all(monkeypatch)
        config = _base_config([{
            "name": "m",
            "model_path": "/some/model.onnx",
            "config_path": "/some/config.pbtxt",
        }])

        cleanup(config)

        mocks["storage"].delete_pvc.assert_called_once_with("m-pvc", "test-ns")

    def test_pod_deleted_before_pvc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """delete_staging_pod must be called before delete_pvc."""
        mocks = _patch_all(monkeypatch)
        call_order: list[str] = []
        mocks["storage"].delete_staging_pod.side_effect = lambda *a, **kw: call_order.append("pod")
        mocks["storage"].delete_pvc.side_effect        = lambda *a, **kw: call_order.append("pvc")

        config = _base_config([{
            "name": "m",
            "model_path": "/some/model.onnx",
            "config_path": "/some/config.pbtxt",
        }])
        cleanup(config)

        assert call_order == ["pod", "pvc"]

    def test_uses_explicit_pvc_name_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mocks = _patch_all(monkeypatch)
        config = _base_config([{
            "name": "m",
            "model_path": "/some/model.onnx",
            "config_path": "/some/config.pbtxt",
            "pvc_name": "my-custom-pvc",
        }])

        cleanup(config)

        mocks["storage"].delete_staging_pod.assert_called_once_with(
            "my-custom-pvc", "test-ns"
        )
        mocks["storage"].delete_pvc.assert_called_once_with(
            "my-custom-pvc", "test-ns"
        )

    def test_default_pvc_name_is_model_name_pvc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mocks = _patch_all(monkeypatch)
        config = _base_config([{
            "name": "fraud-detection",
            "model_path": "/some/model.onnx",
            "config_path": "/some/config.pbtxt",
        }])

        cleanup(config)

        mocks["storage"].delete_pvc.assert_called_once_with(
            "fraud-detection-pvc", "test-ns"
        )

    def test_skips_pod_and_pvc_for_model_uri_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Models with model_uri (pre-staged) must not trigger staging cleanup."""
        mocks = _patch_all(monkeypatch)
        config = _base_config([{
            "name": "m",
            "model_uri": "pvc://existing-pvc/m",
        }])

        cleanup(config)

        mocks["storage"].delete_staging_pod.assert_not_called()
        mocks["storage"].delete_pvc.assert_not_called()

    def test_cleans_up_multiple_staged_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mocks = _patch_all(monkeypatch)
        config = _base_config([
            {"name": "alpha", "model_path": "/a/model.onnx", "config_path": "/a/c.pbtxt",
             "pvc_name": "alpha-pvc"},
            {"name": "beta",  "model_path": "/b/model.onnx", "config_path": "/b/c.pbtxt"},
        ])

        cleanup(config)

        assert mocks["storage"].delete_staging_pod.call_count == 2
        assert mocks["storage"].delete_pvc.call_count == 2
        mocks["storage"].delete_pvc.assert_any_call("alpha-pvc", "test-ns")
        mocks["storage"].delete_pvc.assert_any_call("beta-pvc", "test-ns")
