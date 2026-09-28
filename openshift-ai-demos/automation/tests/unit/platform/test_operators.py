"""Unit tests for rhoai.platform.operators."""

from unittest.mock import MagicMock

import pytest

from rhoai.platform import operators


@pytest.fixture(autouse=True)
def patch_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rhoai.platform.operators.resources", MagicMock())
    monkeypatch.setattr("rhoai.platform.operators.wait", MagicMock())
    monkeypatch.setattr("rhoai.platform.operators.manifests", MagicMock())


class TestIsInstalled:
    def test_true_when_succeeded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "rhoai.platform.operators._csv_phase",
            lambda name, ns: "Succeeded",
        )
        assert operators.is_installed("rhods-operator", "redhat-ods-operator") is True

    def test_false_when_not_succeeded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "rhoai.platform.operators._csv_phase",
            lambda name, ns: "Installing",
        )
        assert operators.is_installed("rhods-operator", "redhat-ods-operator") is False


class TestInstall:
    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        channel: str = "stable",
        version: str = "",
    ) -> MagicMock:
        """Helper: run install() with a fake subscription dict and return the resources mock."""
        fake_sub = {"apiVersion": "operators.coreos.com/v1alpha1", "kind": "Subscription",
                    "metadata": {"name": "rhods-operator"}, "spec": {"channel": "stable",
                    "installPlanApproval": "Automatic"}}
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.operators.resources", mock_resources)
        monkeypatch.setattr("rhoai.platform.operators.load", lambda _: dict(fake_sub))
        monkeypatch.setattr("rhoai.platform.operators._approve_install_plan", MagicMock())
        operators.install("rhods-operator", "redhat-ods-operator", channel, "/repo", 300,
                          version=version)
        return mock_resources

    def test_applies_caller_channel_not_file_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The channel from the caller must override whatever is in the file."""
        mock_resources = self._run(monkeypatch, channel="stable-3.5")
        applied = mock_resources.apply_dict.call_args[0][0]
        assert applied["spec"]["channel"] == "stable-3.5"

    def test_no_starting_csv_when_version_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no version, startingCSV must not appear and approval stays Automatic."""
        mock_resources = self._run(monkeypatch, channel="stable")
        applied = mock_resources.apply_dict.call_args[0][0]
        assert "startingCSV" not in applied["spec"]
        assert applied["spec"]["installPlanApproval"] == "Automatic"

    def test_sets_starting_csv_and_manual_approval_when_version_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a version, startingCSV is set and installPlanApproval becomes Manual."""
        mock_resources = self._run(monkeypatch, channel="stable-3.5",
                                   version="rhods-operator.v3.5.0")
        applied = mock_resources.apply_dict.call_args[0][0]
        assert applied["spec"]["startingCSV"] == "rhods-operator.v3.5.0"
        assert applied["spec"]["installPlanApproval"] == "Manual"

    def test_uses_apply_dict_not_apply_manifest_for_subscription(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """install() must use apply_dict (not apply_manifest) for the Subscription
        so the in-memory mutations actually reach the cluster."""
        mock_resources = self._run(monkeypatch, channel="stable-3.5")
        # apply_dict must have been called (for the subscription)
        mock_resources.apply_dict.assert_called_once()
        # apply_manifest may be called for the OperatorGroup only
        og_call_kinds = [
            str(c) for c in mock_resources.apply_manifest.call_args_list
        ]
        assert not any("Subscription" in k for k in og_call_kinds)


class TestVerify:
    def test_passes_when_succeeded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "rhoai.platform.operators._csv_phase",
            lambda name, ns: "Succeeded",
        )
        operators.verify("rhods-operator", "redhat-ods-operator")  # should not raise

    def test_raises_when_not_succeeded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "rhoai.platform.operators._csv_phase",
            lambda name, ns: "Failed",
        )
        with pytest.raises(RuntimeError, match="not ready"):
            operators.verify("rhods-operator", "redhat-ods-operator")


class TestCsvPhase:
    def test_returns_phase_from_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.status.return_value = {"phase": "Succeeded"}
        monkeypatch.setattr("rhoai.platform.operators.resources", mock_resources)
        assert operators._csv_phase("my-csv", "ns") == "Succeeded"

    def test_returns_unknown_when_status_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.status.return_value = {}
        monkeypatch.setattr("rhoai.platform.operators.resources", mock_resources)
        assert operators._csv_phase("my-csv", "ns") == "Unknown"
