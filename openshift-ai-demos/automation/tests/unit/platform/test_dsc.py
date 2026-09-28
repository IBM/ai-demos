"""Unit tests for rhoai.platform.dsc."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rhoai.platform import dsc


@pytest.fixture(autouse=True)
def patch_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rhoai.platform.dsc.resources", MagicMock())
    monkeypatch.setattr("rhoai.platform.dsc.wait", MagicMock())


class TestApplyDsci:
    def test_calls_apply_manifest(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.dsc.resources", mock_resources)
        p = tmp_path / "dsci.yaml"
        p.write_text("apiVersion: v1\nkind: DSCInitialization\nmetadata:\n  name: x\n")
        dsc.apply_dsci(p)
        mock_resources.apply_manifest.assert_called_once_with(p)


class TestApplyDsc:
    def test_calls_apply_manifest(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.dsc.resources", mock_resources)
        p = tmp_path / "dsc.yaml"
        p.write_text("apiVersion: v1\nkind: DataScienceCluster\nmetadata:\n  name: x\n")
        dsc.apply_dsc(p)
        mock_resources.apply_manifest.assert_called_once_with(p)


class TestVerify:
    def test_passes_when_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("rhoai.platform.dsc._is_ready", lambda name: True)
        dsc.verify("default-dsc")  # should not raise

    def test_raises_when_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("rhoai.platform.dsc._is_ready", lambda name: False)
        mock_resources = MagicMock()
        mock_resources.status.return_value = {"phase": "Progressing"}
        monkeypatch.setattr("rhoai.platform.dsc.resources", mock_resources)
        with pytest.raises(RuntimeError, match="not ready"):
            dsc.verify("default-dsc")


class TestSetComponentStates:
    def test_sends_merge_patch_for_named_components_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.dsc.resources", mock_resources)

        dsc.set_component_states("default-dsc", {"kserve": "Managed", "trustyai": "Removed"})

        mock_resources.patch.assert_called_once_with(
            "DataScienceCluster",
            "default-dsc",
            {
                "spec": {
                    "components": {
                        "kserve":   {"managementState": "Managed"},
                        "trustyai": {"managementState": "Removed"},
                    }
                }
            },
            strategy="merge",
        )

    def test_single_component(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.dsc.resources", mock_resources)

        dsc.set_component_states("default-dsc", {"dashboard": "Managed"})

        patch_body = mock_resources.patch.call_args[0][2]
        assert patch_body["spec"]["components"] == {"dashboard": {"managementState": "Managed"}}


class TestWaitDsciReady:
    def test_delegates_to_generic_wait_with_dsci_kind(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.dsc.wait", mock_wait)

        dsc.wait_dsci_ready("default-dsci", 120)

        mock_wait.wait_until_ready.assert_called_once_with(
            "DSCInitialization", "default-dsci", timeout=120
        )


class TestIsReady:
    def test_true_when_phase_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.status.return_value = {"phase": "Ready"}
        monkeypatch.setattr("rhoai.platform.dsc.resources", mock_resources)
        assert dsc._is_ready("default-dsc") is True

    def test_false_when_phase_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.status.return_value = {"phase": "Progressing"}
        monkeypatch.setattr("rhoai.platform.dsc.resources", mock_resources)
        assert dsc._is_ready("default-dsc") is False


class TestIsReconciled:
    """_is_reconciled requires phase==Ready AND observedGeneration >= generation."""

    def _mock(self, monkeypatch: pytest.MonkeyPatch, phase: str, generation: int, observed: int) -> None:
        mock_resources = MagicMock()
        mock_resources.get.return_value = {
            "metadata": {"generation": generation},
            "status":   {"phase": phase, "observedGeneration": observed},
        }
        monkeypatch.setattr("rhoai.platform.dsc.resources", mock_resources)

    def test_true_when_ready_and_generation_matches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._mock(monkeypatch, "Ready", generation=3, observed=3)
        assert dsc._is_reconciled("default-dsc") is True

    def test_true_when_ready_and_observed_exceeds_generation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # observed can legitimately exceed generation in some operator implementations
        self._mock(monkeypatch, "Ready", generation=2, observed=3)
        assert dsc._is_reconciled("default-dsc") is True

    def test_false_when_ready_but_generation_not_yet_observed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # DSC is still Ready from the previous spec — operator hasn't picked up the patch yet
        self._mock(monkeypatch, "Ready", generation=4, observed=3)
        assert dsc._is_reconciled("default-dsc") is False

    def test_false_when_reconciling_and_generation_observed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Operator is working on the new generation but not done yet
        self._mock(monkeypatch, "Progressing", generation=4, observed=4)
        assert dsc._is_reconciled("default-dsc") is False

    def test_false_when_not_ready_and_generation_not_observed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._mock(monkeypatch, "Progressing", generation=4, observed=3)
        assert dsc._is_reconciled("default-dsc") is False


class TestWaitUntilReady:
    def test_uses_is_reconciled_not_is_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """wait_until_ready must poll _is_reconciled, not the bare phase check."""
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.dsc.wait", mock_wait)

        dsc.wait_until_ready("default-dsc", 600)

        mock_wait.wait_until.assert_called_once()
        condition_fn = mock_wait.wait_until.call_args[0][0]

        # Confirm the condition uses _is_reconciled by testing its behaviour:
        # Ready + generation not yet observed → False (stale Ready must not exit)
        mock_resources = MagicMock()
        mock_resources.get.return_value = {
            "metadata": {"generation": 5},
            "status":   {"phase": "Ready", "observedGeneration": 4},
        }
        monkeypatch.setattr("rhoai.platform.dsc.resources", mock_resources)
        assert condition_fn() is False

        # Ready + generation observed → True
        mock_resources.get.return_value = {
            "metadata": {"generation": 5},
            "status":   {"phase": "Ready", "observedGeneration": 5},
        }
        assert condition_fn() is True



class TestWaitUntilNoDeployments:
    def test_resolves_when_no_rhoai_deployments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.list_resources.return_value = []
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.dsc.resources", mock_resources)
        monkeypatch.setattr("rhoai.platform.dsc.wait", mock_wait)
        dsc.wait_until_no_deployments("redhat-ods-applications", timeout=60)
        mock_wait.wait_until.assert_called_once()

    def test_condition_true_when_no_deployments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.list_resources.return_value = []
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.dsc.resources", mock_resources)
        monkeypatch.setattr("rhoai.platform.dsc.wait", mock_wait)
        dsc.wait_until_no_deployments("redhat-ods-applications")
        # Extract the lambda and call it to confirm it queries with the label selector.
        condition_fn = mock_wait.wait_until.call_args[0][0]
        assert condition_fn() is True
        assert mock_resources.list_resources.call_args[0] == ("Deployment", "redhat-ods-applications")
        assert mock_resources.list_resources.call_args[1]["label_selector"] == (
            "app.kubernetes.io/part-of=rhods-operator"
        )

    def test_condition_false_when_deployments_remain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.list_resources.return_value = [{"metadata": {"name": "kserve-controller"}}]
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.dsc.resources", mock_resources)
        monkeypatch.setattr("rhoai.platform.dsc.wait", mock_wait)
        dsc.wait_until_no_deployments("redhat-ods-applications")
        condition_fn = mock_wait.wait_until.call_args[0][0]
        assert condition_fn() is False

    def test_passes_timeout_to_wait_until(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.dsc.resources", MagicMock())
        monkeypatch.setattr("rhoai.platform.dsc.wait", mock_wait)
        dsc.wait_until_no_deployments("redhat-ods-applications", timeout=120)
        _, kwargs = mock_wait.wait_until.call_args
        assert kwargs["timeout"] == 120

    def test_description_names_namespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.dsc.resources", MagicMock())
        monkeypatch.setattr("rhoai.platform.dsc.wait", mock_wait)
        dsc.wait_until_no_deployments("redhat-ods-applications")
        description = mock_wait.wait_until.call_args[0][1]
        assert "redhat-ods-applications" in description


class TestIsDscReady:
    def test_true_when_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.status.return_value = {"phase": "Ready"}
        monkeypatch.setattr("rhoai.platform.dsc.resources", mock_resources)
        assert dsc.is_dsc_ready("default-dsc") is True

    def test_false_when_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.status.return_value = {"phase": "Progressing"}
        monkeypatch.setattr("rhoai.platform.dsc.resources", mock_resources)
        assert dsc.is_dsc_ready("default-dsc") is False

    def test_false_when_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kubernetes.dynamic.exceptions import NotFoundError
        mock_resources = MagicMock()
        mock_resources.status.side_effect = NotFoundError(MagicMock())
        monkeypatch.setattr("rhoai.platform.dsc.resources", mock_resources)
        assert dsc.is_dsc_ready("default-dsc") is False


class TestIsDsciReady:
    def test_true_when_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.status.return_value = {"phase": "Ready"}
        monkeypatch.setattr("rhoai.platform.dsc.resources", mock_resources)
        assert dsc.is_dsci_ready("default-dsci") is True

    def test_false_when_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.status.return_value = {"phase": "Progressing"}
        monkeypatch.setattr("rhoai.platform.dsc.resources", mock_resources)
        assert dsc.is_dsci_ready("default-dsci") is False

    def test_false_when_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DSCI not yet created (fresh cluster) must return False, not raise 404."""
        from kubernetes.dynamic.exceptions import NotFoundError
        mock_resources = MagicMock()
        mock_resources.status.side_effect = NotFoundError(MagicMock())
        monkeypatch.setattr("rhoai.platform.dsc.resources", mock_resources)
        assert dsc.is_dsci_ready("default-dsci") is False
