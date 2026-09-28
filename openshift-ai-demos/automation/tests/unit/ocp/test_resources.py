"""Unit tests for rhoai.ocp.resources.

All Kubernetes API calls are mocked — no live cluster required.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from rhoai.ocp import resources


def _make_resource_mock(items: list[dict] | None = None) -> MagicMock:
    """Return a mock DynamicResource whose .get() returns a list-like object."""
    res = MagicMock()
    if items is not None:
        list_result = MagicMock()
        list_result.to_dict.return_value = {"items": items}
        res.get.return_value = list_result
    return res


def _make_client_mock(resource_mock: MagicMock) -> MagicMock:
    client = MagicMock()
    client.resources.get.return_value = resource_mock
    return client


@pytest.fixture(autouse=True)
def no_kube_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent any real kubeconfig load during tests."""
    monkeypatch.setattr("rhoai.ocp.resources._client", lambda: MagicMock())


class TestApiHints:
    def test_guardrails_orchestrator_present(self) -> None:
        assert "GuardrailsOrchestrator" in resources._API_HINTS
        assert resources._API_HINTS["GuardrailsOrchestrator"] == "trustyai.opendatahub.io/v1alpha1"

    def test_inference_service_present(self) -> None:
        assert resources._API_HINTS["InferenceService"] == "serving.kserve.io/v1beta1"

    def test_dsc_present(self) -> None:
        assert resources._API_HINTS["DataScienceCluster"] == "datasciencecluster.opendatahub.io/v2"


class TestExists:
    def test_returns_true_when_resource_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        resource_mock = MagicMock()
        resource_mock.get.return_value.to_dict.return_value = {"metadata": {"name": "foo"}}
        client_mock = _make_client_mock(resource_mock)
        monkeypatch.setattr("rhoai.ocp.resources._client", lambda: client_mock)
        assert resources.exists("Pod", "foo", "default") is True

    def test_returns_false_when_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kubernetes.dynamic.exceptions import NotFoundError
        resource_mock = MagicMock()
        resource_mock.get.side_effect = NotFoundError(MagicMock())
        client_mock = _make_client_mock(resource_mock)
        monkeypatch.setattr("rhoai.ocp.resources._client", lambda: client_mock)
        assert resources.exists("Pod", "missing", "default") is False


class TestIsReady:
    def _patch_status(self, monkeypatch: pytest.MonkeyPatch, status_dict: dict[str, Any]) -> None:
        monkeypatch.setattr("rhoai.ocp.resources.status", lambda *_: status_dict)

    def test_deployment_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_status(monkeypatch, {"availableReplicas": 1, "replicas": 1})
        assert resources.is_ready("Deployment", "my-deploy", "ns") is True

    def test_deployment_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_status(monkeypatch, {"availableReplicas": 0, "replicas": 1})
        assert resources.is_ready("Deployment", "my-deploy", "ns") is False

    def test_pod_running_and_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_status(monkeypatch, {
            "phase": "Running",
            "containerStatuses": [{"ready": True}],
        })
        assert resources.is_ready("Pod", "my-pod", "ns") is True

    def test_pod_running_container_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_status(monkeypatch, {
            "phase": "Running",
            "containerStatuses": [{"ready": False}],
        })
        assert resources.is_ready("Pod", "my-pod", "ns") is False

    def test_job_succeeded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_status(monkeypatch, {"succeeded": 1})
        assert resources.is_ready("Job", "my-job", "ns") is True

    def test_job_not_succeeded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_status(monkeypatch, {"succeeded": 0})
        assert resources.is_ready("Job", "my-job", "ns") is False

    def test_dsc_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_status(monkeypatch, {"phase": "Ready"})
        assert resources.is_ready("DataScienceCluster", "default-dsc") is True

    def test_dsc_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_status(monkeypatch, {"phase": "Progressing"})
        assert resources.is_ready("DataScienceCluster", "default-dsc") is False

    def test_csv_succeeded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_status(monkeypatch, {"phase": "Succeeded"})
        assert resources.is_ready("ClusterServiceVersion", "csv", "ns") is True

    def test_generic_ready_condition(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_status(monkeypatch, {
            "conditions": [{"type": "Ready", "status": "True"}]
        })
        assert resources.is_ready("InferenceService", "isvc", "ns") is True

    def test_generic_not_ready_condition(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_status(monkeypatch, {
            "conditions": [{"type": "Ready", "status": "False"}]
        })
        assert resources.is_ready("InferenceService", "isvc", "ns") is False

    def test_not_found_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kubernetes.dynamic.exceptions import NotFoundError
        monkeypatch.setattr(
            "rhoai.ocp.resources.status",
            MagicMock(side_effect=NotFoundError(MagicMock()))
        )
        assert resources.is_ready("Pod", "gone", "ns") is False


class TestListResources:
    def test_returns_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        items = [{"metadata": {"name": "a"}}, {"metadata": {"name": "b"}}]
        resource_mock = _make_resource_mock(items)
        client_mock   = _make_client_mock(resource_mock)
        monkeypatch.setattr("rhoai.ocp.resources._client", lambda: client_mock)
        result = resources.list_resources("Pod", "default")
        assert len(result) == 2

    def test_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        resource_mock = _make_resource_mock([])
        client_mock   = _make_client_mock(resource_mock)
        monkeypatch.setattr("rhoai.ocp.resources._client", lambda: client_mock)
        result = resources.list_resources("Pod", "default")
        assert result == []
