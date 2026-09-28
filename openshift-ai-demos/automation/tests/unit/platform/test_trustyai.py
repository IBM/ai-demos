"""Unit tests for rhoai.platform.trustyai."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rhoai.platform import trustyai


@pytest.fixture(autouse=True)
def patch_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rhoai.platform.trustyai.resources", MagicMock())
    monkeypatch.setattr("rhoai.platform.trustyai.wait", MagicMock())


# ---------------------------------------------------------------------------
# enable_user_workload_monitoring
# ---------------------------------------------------------------------------

_MONITORING_YAML = """\
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-monitoring-config
  namespace: openshift-monitoring
data:
  config.yaml: |
    enableUserWorkload: true
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: user-workload-monitoring-config
  namespace: openshift-user-workload-monitoring
data:
  config.yaml: |
    prometheus:
      logLevel: debug
      retention: 15d
"""


class TestEnableUserWorkloadMonitoring:
    @pytest.fixture()
    def manifest(self, tmp_path: Path) -> Path:
        p = tmp_path / "monitoring-config.yaml"
        p.write_text(_MONITORING_YAML)
        return p

    def test_applies_two_configmaps(
        self, monkeypatch: pytest.MonkeyPatch, manifest: Path
    ) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        trustyai.enable_user_workload_monitoring(manifest)
        assert mock_resources.apply_dict.call_count == 2

    def test_applies_cluster_monitoring_config(
        self, monkeypatch: pytest.MonkeyPatch, manifest: Path
    ) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        trustyai.enable_user_workload_monitoring(manifest)
        first_call_body = mock_resources.apply_dict.call_args_list[0][0][0]
        assert first_call_body["metadata"]["name"] == "cluster-monitoring-config"
        assert first_call_body["metadata"]["namespace"] == "openshift-monitoring"

    def test_applies_user_workload_monitoring_config(
        self, monkeypatch: pytest.MonkeyPatch, manifest: Path
    ) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        trustyai.enable_user_workload_monitoring(manifest)
        second_call_body = mock_resources.apply_dict.call_args_list[1][0][0]
        assert second_call_body["metadata"]["name"] == "user-workload-monitoring-config"
        ns = second_call_body["metadata"]["namespace"]
        assert ns == "openshift-user-workload-monitoring"

    def test_passes_namespace_from_manifest(
        self, monkeypatch: pytest.MonkeyPatch, manifest: Path
    ) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        trustyai.enable_user_workload_monitoring(manifest)
        first_ns = mock_resources.apply_dict.call_args_list[0][0][1]
        assert first_ns == "openshift-monitoring"


# ---------------------------------------------------------------------------
# apply_rbac
# ---------------------------------------------------------------------------

class TestApplyRbac:
    def test_applies_each_document(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        path = tmp_path / "rbac.yaml"
        path.write_text(
            "kind: ServiceAccount\nmetadata:\n  name: sa\n---\n"
            "kind: RoleBinding\nmetadata:\n  name: rb\n"
        )
        trustyai.apply_rbac(path, "ns")
        assert mock_resources.apply_dict.call_count == 2

    def test_passes_namespace_to_apply_dict(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        path = tmp_path / "rbac.yaml"
        path.write_text("kind: ServiceAccount\nmetadata:\n  name: sa\n")
        trustyai.apply_rbac(path, "test-ns")
        assert mock_resources.apply_dict.call_args[0][1] == "test-ns"


# ---------------------------------------------------------------------------
# apply_logger_ca_bundle
# ---------------------------------------------------------------------------

class TestApplyLoggerCaBundle:
    def test_delegates_to_apply_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        path = tmp_path / "kserve-logger-ca-bundle.yaml"
        path.write_text("kind: ConfigMap\n")
        trustyai.apply_logger_ca_bundle(path, "test-ns")
        mock_resources.apply_manifest.assert_called_once_with(path, "test-ns")

    def test_passes_namespace_to_apply_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        path = tmp_path / "kserve-logger-ca-bundle.yaml"
        path.write_text("kind: ConfigMap\n")
        trustyai.apply_logger_ca_bundle(path, "my-ns")
        assert mock_resources.apply_manifest.call_args[0][1] == "my-ns"


# ---------------------------------------------------------------------------
# patch_inferenceservice_config
# ---------------------------------------------------------------------------

class TestPatchInferencserviceConfig:
    def test_calls_patch_with_correct_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        # get() must return a dict with a parseable logger JSON string so the
        # second patch (CA bundle merge) can deserialise and re-serialise it.
        mock_resources.get.return_value = {
            "data": {"logger": '{"cpuLimit": "1"}'}
        }
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        trustyai.patch_inferenceservice_config("test-ns")
        # First patch: annotation to stop RHOAI managing the ConfigMap.
        first_call = mock_resources.patch.call_args_list[0]
        assert first_call.args == (
            "ConfigMap",
            "inferenceservice-config",
            {"metadata": {"annotations": {"opendatahub.io/managed": "false"}}},
        )
        assert first_call.kwargs == {"namespace": "test-ns"}
        # Second patch: logger CA bundle keys merged in.
        second_call = mock_resources.patch.call_args_list[1]
        import json
        logger_cfg = json.loads(second_call.args[2]["data"]["logger"])
        assert logger_cfg["caBundle"]    == "kserve-logger-ca-bundle"
        assert logger_cfg["caCertFile"]  == "service-ca.crt"
        assert logger_cfg["tlsSkipVerify"] is False
        # Original keys preserved.
        assert logger_cfg["cpuLimit"] == "1"


# ---------------------------------------------------------------------------
# apply_trustyai_service
# ---------------------------------------------------------------------------

class TestApplyTrustyAIService:
    def test_delegates_to_apply_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        path = tmp_path / "trustyai-service.yaml"
        path.write_text("kind: TrustyAIService\n")
        trustyai.apply_trustyai_service(path, "ns")
        mock_resources.apply_manifest.assert_called_once_with(path, "ns")


# ---------------------------------------------------------------------------
# wait_until_ready
# ---------------------------------------------------------------------------

class TestWaitUntilReady:
    def test_delegates_to_wait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai.wait", mock_wait)
        trustyai.wait_until_ready("trustyai-service", "ns", timeout=120)
        mock_wait.wait_until_ready.assert_called_once_with(
            "Deployment", "trustyai-service", "ns", timeout=120, on_tick=None
        )


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

class TestVerify:
    def test_passes_when_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.is_ready.return_value = True
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        trustyai.verify("trustyai-service", "ns")  # should not raise

    def test_raises_when_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.is_ready.return_value = False
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        with pytest.raises(RuntimeError, match="not ready"):
            trustyai.verify("trustyai-service", "ns")

    def test_error_message_contains_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.is_ready.return_value = False
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        with pytest.raises(RuntimeError, match="my-trustyai"):
            trustyai.verify("my-trustyai", "ns")


# ---------------------------------------------------------------------------
# delete_trustyai_service
# ---------------------------------------------------------------------------

class TestDeleteTrustyAIService:
    def test_deletes_cr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        monkeypatch.setattr("rhoai.platform.trustyai.wait", mock_wait)
        trustyai.delete_trustyai_service("trustyai-service", "ns")
        mock_resources.delete_manifest.assert_called_once_with(
            "TrustyAIService", "trustyai-service", "ns"
        )

    def test_waits_for_cr_deletion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai.resources", MagicMock())
        monkeypatch.setattr("rhoai.platform.trustyai.wait", mock_wait)
        trustyai.delete_trustyai_service("trustyai-service", "ns")
        mock_wait.wait_until_deleted.assert_any_call(
            "TrustyAIService", "trustyai-service", "ns"
        )

    def test_waits_for_deployment_deletion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai.resources", MagicMock())
        monkeypatch.setattr("rhoai.platform.trustyai.wait", mock_wait)
        trustyai.delete_trustyai_service("trustyai-service", "ns")
        mock_wait.wait_until_deleted.assert_any_call(
            "Deployment", "trustyai-service", "ns"
        )

    def test_waits_for_cr_before_deployment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai.resources", MagicMock())
        monkeypatch.setattr("rhoai.platform.trustyai.wait", mock_wait)
        trustyai.delete_trustyai_service("trustyai-service", "ns")
        calls = [c.args[0] for c in mock_wait.wait_until_deleted.call_args_list]
        assert calls == ["TrustyAIService", "Deployment"]


# ---------------------------------------------------------------------------
# get_url
# ---------------------------------------------------------------------------

class TestGetUrl:
    def test_returns_status_url_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.status.return_value = {"url": "https://trustyai.apps.example.com"}
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        assert trustyai.get_url("trustyai-service", "ns") == "https://trustyai.apps.example.com"

    def test_falls_back_to_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.status.return_value = {}
        mock_resources.exists.return_value = True
        mock_resources.get.return_value = {
            "spec": {"host": "trustyai.apps.example.com", "tls": {"termination": "edge"}}
        }
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        assert trustyai.get_url("trustyai-service", "ns") == "https://trustyai.apps.example.com"

    def test_raises_when_no_url_and_no_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.status.return_value = {}
        mock_resources.exists.return_value = False
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        with pytest.raises(RuntimeError, match="no URL"):
            trustyai.get_url("trustyai-service", "ns")


# ---------------------------------------------------------------------------
# get_bearer_token
# ---------------------------------------------------------------------------

class TestGetBearerToken:
    def _patch_subprocess(self, monkeypatch: pytest.MonkeyPatch, stdout: str, returncode: int = 0):
        result = MagicMock()
        result.returncode = returncode
        result.stdout = stdout
        result.stderr = ""
        mock_sp = MagicMock(run=lambda *a, **kw: result)
        monkeypatch.setattr("rhoai.platform.trustyai.subprocess", mock_sp)

    def test_returns_token_from_json_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import json
        payload = json.dumps({"status": {"token": "my-token-abc"}})
        self._patch_subprocess(monkeypatch, payload)
        assert trustyai.get_bearer_token("trustyai-user", "ns") == "my-token-abc"

    def test_falls_back_to_plain_text_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_subprocess(monkeypatch, "plain-text-token\n")
        assert trustyai.get_bearer_token("trustyai-user", "ns") == "plain-text-token"

    def test_raises_on_nonzero_returncode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = MagicMock()
        result.returncode = 1
        result.stderr = "ServiceAccount not found"
        monkeypatch.setattr(
            "rhoai.platform.trustyai.subprocess",
            MagicMock(run=lambda *a, **kw: result),
        )
        with pytest.raises(RuntimeError, match="ServiceAccount not found"):
            trustyai.get_bearer_token("trustyai-user", "ns")


# ---------------------------------------------------------------------------
# delete_service_account / delete_role_binding
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_delete_logger_ca_bundle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        trustyai.delete_logger_ca_bundle("ns")
        mock_resources.delete_manifest.assert_called_once_with(
            "ConfigMap", "kserve-logger-ca-bundle", "ns"
        )

    def test_delete_service_account(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        trustyai.delete_service_account("trustyai-user", "ns")
        mock_resources.delete_manifest.assert_called_once_with(
            "ServiceAccount", "trustyai-user", "ns"
        )

    def test_delete_role_binding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai.resources", mock_resources)
        trustyai.delete_role_binding("trustyai-user-view", "ns")
        mock_resources.delete_manifest.assert_called_once_with(
            "RoleBinding", "trustyai-user-view", "ns"
        )


# ---------------------------------------------------------------------------
# resolve_and_apply_name_mapping
# ---------------------------------------------------------------------------

class TestResolveAndApplyNameMapping:
    def test_explicit_mappings_used_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit inputs and outputs are applied verbatim; /info is never fetched."""
        get_info = MagicMock()
        apply    = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai_client.get_model_info", get_info)
        monkeypatch.setattr("rhoai.platform.trustyai_client.apply_name_mapping", apply)

        result = trustyai.resolve_and_apply_name_mapping(
            "http://ta", "tok", "m",
            explicit_inputs={"customer_data_input-3": "Is Male?"},
            explicit_outputs={"predict": "Will Default?"},
            csv_headers=["a", "b"],          # present, but must be ignored
            num_input_tensors=1,
            pbtxt_output_names=["predict"],  # present, but explicit outputs win
        )
        get_info.assert_not_called()
        apply.assert_called_once_with(
            "http://ta", "tok", "m",
            {"customer_data_input-3": "Is Male?"},
            {"predict": "Will Default?"},
        )
        assert result == (
            {"customer_data_input-3": "Is Male?"},
            {"predict": "Will Default?"},
        )

    def test_nothing_available_applies_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No CSV, no pbtxt, empty response schema → nothing applied."""
        get_info = MagicMock(return_value={"data": {}})
        apply    = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai_client.get_model_info", get_info)
        monkeypatch.setattr("rhoai.platform.trustyai_client.apply_name_mapping", apply)

        result = trustyai.resolve_and_apply_name_mapping(
            "http://ta", "tok", "m",
            explicit_inputs={}, explicit_outputs={},
            csv_headers=None, num_input_tensors=3,
            pbtxt_output_names=[],
        )
        apply.assert_not_called()
        assert result == ({}, {})

    def test_auto_derives_flat_tensor_from_headers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CSV headers + single flat tensor → derived input mapping fetched and applied."""
        get_info = MagicMock(return_value={
            "data": {"inputSchema": {"items": {
                "t-0": {"name": "t-0", "columnIndex": 0},
                "t-1": {"name": "t-1", "columnIndex": 1},
            }}}
        })
        apply = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai_client.get_model_info", get_info)
        monkeypatch.setattr("rhoai.platform.trustyai_client.apply_name_mapping", apply)

        result = trustyai.resolve_and_apply_name_mapping(
            "http://ta", "tok", "m",
            explicit_inputs={}, explicit_outputs={},
            csv_headers=["age", "income"], num_input_tensors=1,
            pbtxt_output_names=["predict"],
        )
        get_info.assert_called_once()      # fetched once, reused for both schemas
        apply.assert_called_once_with(
            "http://ta", "tok", "m",
            {"t-0": "Age", "t-1": "Income"}, {"predict": "predict"},
        )
        assert result == ({"t-0": "Age", "t-1": "Income"}, {"predict": "predict"})

    def test_ambiguous_inputs_still_default_outputs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A >1-tensor suffixed layout derives no inputs, but outputs still default."""
        get_info = MagicMock(return_value={
            "data": {"inputSchema": {"items": {
                "a-0": {"name": "a-0", "columnIndex": 0},
                "b-0": {"name": "b-0", "columnIndex": 1},
            }}}
        })
        apply = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai_client.get_model_info", get_info)
        monkeypatch.setattr("rhoai.platform.trustyai_client.apply_name_mapping", apply)

        result = trustyai.resolve_and_apply_name_mapping(
            "http://ta", "tok", "m",
            explicit_inputs={}, explicit_outputs={},
            csv_headers=["a", "b"], num_input_tensors=2,
            pbtxt_output_names=["label"],
        )
        apply.assert_called_once_with("http://ta", "tok", "m", {}, {"label": "label"})
        assert result == ({}, {"label": "label"})

    def test_defaults_outputs_from_pbtxt_identity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No explicit outputs → identity mapping from pbtxt names; no /info fetch."""
        get_info = MagicMock()
        apply    = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai_client.get_model_info", get_info)
        monkeypatch.setattr("rhoai.platform.trustyai_client.apply_name_mapping", apply)

        result = trustyai.resolve_and_apply_name_mapping(
            "http://ta", "tok", "m",
            explicit_inputs={}, explicit_outputs={},
            csv_headers=None, num_input_tensors=1,
            pbtxt_output_names=["probabilities", "label"],
        )
        get_info.assert_not_called()
        apply.assert_called_once_with(
            "http://ta", "tok", "m",
            {}, {"probabilities": "probabilities", "label": "label"},
        )
        assert result == ({}, {"probabilities": "probabilities", "label": "label"})

    def test_defaults_outputs_from_response_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No pbtxt → identity mapping from the response (outputSchema) names."""
        get_info = MagicMock(return_value={
            "data": {"outputSchema": {"items": {
                "predict": {"name": "predict", "columnIndex": 0},
            }}}
        })
        apply = MagicMock()
        monkeypatch.setattr("rhoai.platform.trustyai_client.get_model_info", get_info)
        monkeypatch.setattr("rhoai.platform.trustyai_client.apply_name_mapping", apply)

        result = trustyai.resolve_and_apply_name_mapping(
            "http://ta", "tok", "m",
            explicit_inputs={}, explicit_outputs={},
            csv_headers=None, num_input_tensors=1,
            pbtxt_output_names=[],
        )
        get_info.assert_called_once()
        apply.assert_called_once_with("http://ta", "tok", "m", {}, {"predict": "predict"})
        assert result == ({}, {"predict": "predict"})
