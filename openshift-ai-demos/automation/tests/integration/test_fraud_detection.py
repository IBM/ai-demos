"""Integration smoke tests — no live cluster required.

Verifies that the full fraud-detection use-case module graph imports
cleanly and that config defaults wire up correctly.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call

import pytest
import yaml

from rhoai.config.loader import load_config
from rhoai.platform import manifests
from rhoai.usecases import registry
from rhoai.usecases.fraud_detection import assets

# Minimal InferenceService manifest that matches the structure deploy.py mutates.
_ISVC_MANIFEST: dict[str, Any] = {
    "apiVersion": "serving.kserve.io/v1beta1",
    "kind": "InferenceService",
    "metadata": {
        "name": "fraud-detection",
        "annotations": {
            "opendatahub.io/connection-type-ref": "s3",
        },
    },
    "spec": {
        "predictor": {
            "model": {
                "runtime": "triton-ppc64le-runtime",
                "storage": {"key": "s3-credentials", "path": "models/fraud"},
            }
        }
    },
}


def _write_manifest(tmp_path: Path) -> None:
    """Write the stub InferenceService manifest into the expected asset path."""
    dest = (
        tmp_path
        / "model-serving"
        / "predictive-models"
        / "triton"
        / "fraud-detection.yaml"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.dump(_ISVC_MANIFEST))


class TestModuleGraph:
    def test_registry_imports_fraud_detection(self) -> None:
        mod = registry.get("fraud-detection")
        assert mod is not None

    def test_deploy_verify_cleanup_callable(self) -> None:
        mod = registry.get("fraud-detection")
        assert callable(mod.deploy)
        assert callable(mod.verify)
        assert callable(mod.cleanup)


class TestConfigDefaults:
    def test_deployment_defaults_present(self) -> None:
        config = load_config()
        dep = config.get("deployment", {})
        # models list replaces the old scalar keys.
        models = dep.get("models", [])
        assert len(models) == 1
        assert models[0]["name"] == "fraud-detection"
        assert models[0]["model_uri"] == "pvc://fraud-model-pvc/models"
        assert dep.get("trustyai_service_name") == "trustyai-service"
        # runtime name is derived, not a config constant.
        assert assets.serving_runtime_name("fraud-detection") == "triton-fraud-detection"

    def test_timeouts_present(self) -> None:
        config = load_config()
        assert "operator_ready"  in config["timeouts"]
        assert "dsc_ready"       in config["timeouts"]
        assert "inference_ready" in config["timeouts"]
        assert "trustyai_ready"  in config["timeouts"]


class TestAssetPaths:
    def test_serving_runtime_template_path(self, tmp_path: Path) -> None:
        p = assets.get_serving_runtime_template(tmp_path)
        assert p == (
            tmp_path
            / "model-serving"
            / "predictive-models"
            / "triton"
            / "triton-ppc64le-runtime-template.yaml"
        )

    def test_model_manifest_path(self, tmp_path: Path) -> None:
        p = assets.get_model_manifest(tmp_path)
        assert p == (
            tmp_path / "model-serving" / "predictive-models" / "triton" / "fraud-detection.yaml"
        )

    def test_trustyai_service_path(self, tmp_path: Path) -> None:
        p = manifests.get_trustyai_service(tmp_path)
        assert p == tmp_path / "trustyai" / "service" / "trustyai-service.yaml"


def _write_request(tmp_path: Path, rel: str = "requests/model.json") -> str:
    """Write a stub KServe v2 inference request and return its absolute path."""
    import json as _json
    dest = tmp_path / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_json.dumps({"inputs": [{"name": "dense_input", "data": [0.1] * 10}]}))
    return str(dest)


class TestResolveInferenceRequest:
    def test_returns_configured_path(self, tmp_path: Path) -> None:
        from rhoai.usecases.fraud_detection.assets import resolve_inference_request
        abs_path = str(tmp_path / "requests" / "fraud.json")
        model = {"name": "fraud-detection", "inference_request": abs_path}
        result = resolve_inference_request(model)
        assert result == Path(abs_path)

    def test_returns_none_when_missing(self) -> None:
        from rhoai.usecases.fraud_detection.assets import resolve_inference_request
        assert resolve_inference_request({"name": "m"}) is None

    def test_returns_none_when_empty(self) -> None:
        from rhoai.usecases.fraud_detection.assets import resolve_inference_request
        assert resolve_inference_request({"name": "m", "inference_request": ""}) is None


class TestResolveInferenceDataset:
    def test_returns_configured_path(self, tmp_path: Path) -> None:
        from rhoai.usecases.fraud_detection.assets import resolve_inference_dataset
        abs_path = str(tmp_path / "inputs" / "data.csv")
        result = resolve_inference_dataset({"name": "m", "inference_dataset": abs_path})
        assert result == Path(abs_path)

    def test_raises_when_missing(self) -> None:
        from rhoai.usecases.fraud_detection.assets import resolve_inference_dataset
        with pytest.raises(ValueError, match="no inference_request or inference_dataset configured"):
            resolve_inference_dataset({"name": "m"})


class TestDeploySmoke:
    """Smoke-test deploy() wiring with all cluster I/O mocked.

    deploy() reads the InferenceService manifest from disk via yaml_io.load()
    and applies it via ocp_resources.apply_dict(). Both are patched here so the
    test never touches the real filesystem beyond the stub file we create in tmp_path.
    """

    def _make_config(
        self,
        tmp_path: Path,
        models: list[dict] | None = None,
    ) -> dict:
        config = load_config()
        config["repo_root"] = str(tmp_path)
        config["platform"]["namespace"] = "test-ns"
        if models is not None:
            config.setdefault("deployment", {})["models"] = models
        else:
            # Default: single model with a stub request file.
            rel = _write_request(tmp_path)
            config.setdefault("deployment", {})["models"] = [
                {"name": "fraud-detection", "model_uri": "pvc://fraud-model-pvc/models",
                 "inference_request": rel},
            ]
        return config

    # Stub payload returned by inference._load_request_payload and used as the
    # generated request when inference_dataset is set.  Must be a real dict so
    # render_curl_command can json.dumps it.
    _STUB_PAYLOAD = {"inputs": [{"name": "input", "shape": [1, 3], "datatype": "FP64", "data": [[0.1, 0.2, 0.3]]}]}

    def _patch_all(
        self, monkeypatch: pytest.MonkeyPatch, deploy_mod: Any, tmp_path: Path
    ) -> dict[str, MagicMock]:
        """Patch every external dependency of deploy_mod and return the mocks."""
        storage_mock          = MagicMock()
        inference_mock        = MagicMock()
        ocp_resources_mock    = MagicMock()
        prepare_mock          = MagicMock()
        trustyai_mock         = MagicMock()
        request_generator_mock = MagicMock()
        # platform_needs_reconciliation returns False → fast-path, no bootstrap call.
        prepare_mock.platform_needs_reconciliation.return_value = False
        # _read_tensor_schema must return a 2-tuple (input_name, datatype).
        inference_mock._read_tensor_schema.return_value = ("input", "FP64")
        # _load_request_payload must return a real dict (json.dumps is called on it).
        inference_mock._load_request_payload.return_value = self._STUB_PAYLOAD
        # verify_triton_inference must return a 3-tuple (payload, response, meta).
        inference_mock.verify_triton_inference.return_value = (
            self._STUB_PAYLOAD,
            {"outputs": [{"name": "output", "data": [0.9]}]},
            None,
        )
        # get_inference_url returns a plain string so rstrip and string concat work.
        inference_mock.get_inference_url.return_value = "http://fake-endpoint"
        # _TRITON_INFER_PATH must be a real string for .format() + concatenation.
        inference_mock._TRITON_INFER_PATH = "/v2/models/{model_name}/infer"
        # build_request_from_csv_file / iter_requests return (model_name, payload).
        request_generator_mock.build_request_from_csv_file.return_value = ("fraud-detection", self._STUB_PAYLOAD)
        request_generator_mock.iter_requests.return_value = iter([("fraud-detection", self._STUB_PAYLOAD)])
        # Auto name-mapping derivation (dataset mode) reads CSV headers + pbtxt specs.
        request_generator_mock.read_csv_headers.return_value = ["a", "b", "c"]
        request_generator_mock.parse_pbtxt.return_value = ("fraud-detection", [object()])

        monkeypatch.setattr(deploy_mod, "storage",           storage_mock)
        monkeypatch.setattr(deploy_mod, "inference",         inference_mock)
        monkeypatch.setattr(deploy_mod, "ocp_resources",     ocp_resources_mock)
        monkeypatch.setattr(deploy_mod, "prepare",           prepare_mock)
        monkeypatch.setattr(deploy_mod, "trustyai",          trustyai_mock)
        monkeypatch.setattr(deploy_mod, "request_generator", request_generator_mock)

        return {
            "storage":           storage_mock,
            "inference":         inference_mock,
            "ocp_resources":     ocp_resources_mock,
            "prepare":           prepare_mock,
            "trustyai":          trustyai_mock,
            "request_generator": request_generator_mock,
        }

    def _fresh_deploy_mod(self) -> Any:
        import importlib
        import sys

        sys.modules.pop("rhoai.usecases.fraud_detection.deploy", None)
        return importlib.import_module("rhoai.usecases.fraud_detection.deploy")

    # ------------------------------------------------------------------
    # Core wiring test (default config — pvc:// model, S3 skipped)
    # ------------------------------------------------------------------

    def test_deploy_calls_platform_modules(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _write_manifest(tmp_path)
        deploy_mod = self._fresh_deploy_mod()
        mocks = self._patch_all(monkeypatch, deploy_mod, tmp_path)

        config = self._make_config(tmp_path)
        deploy_mod.deploy(config)

        # Default models list has one pvc:// entry → S3 secret skipped.
        mocks["storage"].apply_s3_secret.assert_not_called()

        # ServingRuntime applied with model_name and runtime_name.
        mocks["inference"].apply_serving_runtime_from_template.assert_called_once()
        rt_call = mocks["inference"].apply_serving_runtime_from_template.call_args
        assert rt_call.kwargs["model_name"]   == "fraud-detection"
        assert rt_call.kwargs["runtime_name"] == "triton-fraud-detection"

        # InferenceService applied once with runtime field stamped.
        mocks["ocp_resources"].apply_dict.assert_called_once()
        applied = mocks["ocp_resources"].apply_dict.call_args[0][0]
        assert applied["spec"]["predictor"]["model"]["runtime"] == "triton-fraud-detection"

        call_args = mocks["inference"].wait_until_ready.call_args
        assert call_args.args == ("fraud-detection", "test-ns", config["timeouts"]["inference_ready"])

        mocks["inference"].verify_triton_inference.assert_called_once()
        call_args = mocks["inference"].verify_triton_inference.call_args[0]
        assert call_args[0] == "fraud-detection"   # isvc_name
        assert call_args[1] == "test-ns"           # namespace
        assert call_args[2] == "fraud-detection"   # model_name

    def test_deploy_generates_request_from_dataset_when_request_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When inference_dataset is set (no inference_request), a request JSON is
        generated from the dataset and passed to verify_triton_inference."""
        _write_manifest(tmp_path)
        deploy_mod = self._fresh_deploy_mod()
        mocks = self._patch_all(monkeypatch, deploy_mod, tmp_path)

        dataset = tmp_path / "inputs" / "demo-loan.csv"
        dataset.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_text("1.0,2.0,3.0\n")
        # inference_config_path is the pbtxt for request generation only — it is
        # separate from config_path (model staging) so model_uri + inference_dataset
        # models don't hit the mutual-exclusivity guard.  The actual parsing is
        # bypassed because request_generator is patched in _patch_all.
        pbtxt = tmp_path / "config.pbtxt"
        pbtxt.write_text('name: "fraud-detection"\nmax_batch_size: 0\n')

        config = self._make_config(tmp_path, models=[
            {"name": "fraud-detection", "model_uri": "pvc://fraud-model-pvc/models",
             "inference_dataset": str(dataset),
             "inference_config_path": str(pbtxt)},
        ])
        deploy_mod.deploy(config)

        # Smoke test and observations share one generator: request_generator.iter_requests
        # was called with the CSV dataset (batch_size=1 for the smoke request).
        mocks["request_generator"].iter_requests.assert_called_once()
        gen_call = mocks["request_generator"].iter_requests.call_args
        assert gen_call.args[1] == dataset
        assert gen_call.kwargs.get("batch_size") == 1

        # A generated request JSON was written and passed to verify_triton_inference.
        mocks["inference"].verify_triton_inference.assert_called_once()
        request_path = mocks["inference"].verify_triton_inference.call_args[0][3]
        assert request_path.suffix == ".json"

    def test_bias_monitoring_dataset_mode_sends_batched_payloads(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Dataset mode derives observations from the dataset (no observations.path),
        batched by observations.batch_size, and posts them via send_observation_payloads."""
        deploy_mod = self._fresh_deploy_mod()
        mocks = self._patch_all(monkeypatch, deploy_mod, tmp_path)
        monkeypatch.setattr(deploy_mod, "trustyai_client", MagicMock())

        dataset = tmp_path / "inputs" / "d.csv"
        dataset.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_text("1.0\n2.0\n3.0\n")
        pbtxt = tmp_path / "config.pbtxt"
        pbtxt.write_text('name: "m"\nmax_batch_size: 0\n')

        batch = self._STUB_PAYLOAD
        mocks["request_generator"].iter_requests.return_value = iter(
            [("m", batch), ("m", batch), ("m", batch)]
        )
        mocks["inference"].send_observation_payloads.return_value = 3

        model = {
            "name": "m",
            "inference_dataset": str(dataset),
            "inference_config_path": str(pbtxt),
            "bias_monitoring": {"observations": {"batch_size": 50}},
        }
        deploy_mod._configure_bias_monitoring(
            model, "http://route", "token", "test-ns", str(tmp_path), 300
        )

        # Observations came from the dataset generator, batched at batch_size=50.
        mocks["request_generator"].iter_requests.assert_called_once()
        assert mocks["request_generator"].iter_requests.call_args.kwargs["batch_size"] == 50

        # The generated payloads were posted; the file-based sender was not used.
        mocks["inference"].send_observation_payloads.assert_called_once()
        sent = mocks["inference"].send_observation_payloads.call_args[0][2]
        assert len(sent) == 3
        mocks["inference"].send_observations.assert_not_called()

    # ------------------------------------------------------------------
    # model_uri variant: PVC — S3 secret skipped, storageUri injected
    # ------------------------------------------------------------------

    def test_deploy_pvc_uri_skips_s3_and_sets_storage_uri(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _write_manifest(tmp_path)
        deploy_mod = self._fresh_deploy_mod()
        mocks = self._patch_all(monkeypatch, deploy_mod, tmp_path)

        config = self._make_config(tmp_path, models=[
            {"name": "fraud-detection", "model_uri": "pvc://fraud-model-pvc/models",
             "inference_request": _write_request(tmp_path)},
        ])
        deploy_mod.deploy(config)

        mocks["storage"].apply_s3_secret.assert_not_called()

        mocks["ocp_resources"].apply_dict.assert_called_once()
        applied_dict = mocks["ocp_resources"].apply_dict.call_args[0][0]
        model_spec = applied_dict["spec"]["predictor"]["model"]
        assert model_spec["storageUri"] == "pvc://fraud-model-pvc/models"
        assert model_spec["runtime"] == "triton-fraud-detection"
        assert "storage" not in model_spec
        assert "opendatahub.io/connection-type-ref" not in applied_dict["metadata"]["annotations"]

    # ------------------------------------------------------------------
    # model_uri variant: plain S3 path — path updated, S3 secret applied
    # ------------------------------------------------------------------

    def test_deploy_s3_path_updates_storage_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _write_manifest(tmp_path)
        deploy_mod = self._fresh_deploy_mod()
        mocks = self._patch_all(monkeypatch, deploy_mod, tmp_path)

        config = self._make_config(tmp_path, models=[
            {"name": "fraud-detection", "model_uri": "models/my-fraud-model",
             "inference_request": _write_request(tmp_path)},
        ])
        deploy_mod.deploy(config)

        mocks["storage"].apply_s3_secret.assert_called_once()

        mocks["ocp_resources"].apply_dict.assert_called_once()
        applied_dict = mocks["ocp_resources"].apply_dict.call_args[0][0]
        model_spec = applied_dict["spec"]["predictor"]["model"]
        assert model_spec["storage"]["path"] == "models/my-fraud-model"
        assert "storageUri" not in model_spec

    # ------------------------------------------------------------------
    # Multi-model: two entries → two ServingRuntimes, two ISVCs, two smoke tests
    # ------------------------------------------------------------------

    def test_deploy_two_models_calls_each_platform_api_twice(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _write_manifest(tmp_path)
        deploy_mod = self._fresh_deploy_mod()
        mocks = self._patch_all(monkeypatch, deploy_mod, tmp_path)

        req_baseline  = _write_request(tmp_path, "requests/baseline.json")
        req_candidate = _write_request(tmp_path, "requests/candidate.json")
        config = self._make_config(tmp_path, models=[
            {"name": "fraud-detection-baseline",  "model_uri": "pvc://pvc/models",
             "inference_request": req_baseline},
            {"name": "fraud-detection-candidate", "model_uri": "pvc://pvc/biased",
             "inference_request": req_candidate},
        ])
        deploy_mod.deploy(config)

        # Both models are pvc:// → S3 secret never applied.
        mocks["storage"].apply_s3_secret.assert_not_called()

        # ServingRuntime applied once per model.
        assert mocks["inference"].apply_serving_runtime_from_template.call_count == 2

        # InferenceService applied once per model.
        assert mocks["ocp_resources"].apply_dict.call_count == 2

        # wait_until_ready called once per model with the correct name each time.
        wait_calls = mocks["inference"].wait_until_ready.call_args_list
        assert len(wait_calls) == 2
        assert wait_calls[0].args[0] == "fraud-detection-baseline"
        assert wait_calls[1].args[0] == "fraud-detection-candidate"

        # Each ServingRuntime call uses a unique runtime_name derived from model name.
        rt_calls = mocks["inference"].apply_serving_runtime_from_template.call_args_list
        assert rt_calls[0].kwargs["runtime_name"] == "triton-fraud-detection-baseline"
        assert rt_calls[1].kwargs["runtime_name"] == "triton-fraud-detection-candidate"

        # Each ISVC dict has the correct runtime field stamped.
        applied_dicts = [c[0][0] for c in mocks["ocp_resources"].apply_dict.call_args_list]
        assert applied_dicts[0]["spec"]["predictor"]["model"]["runtime"] == "triton-fraud-detection-baseline"
        assert applied_dicts[1]["spec"]["predictor"]["model"]["runtime"] == "triton-fraud-detection-candidate"

        # verify_triton_inference receives each model's own request path.
        smoke_calls = mocks["inference"].verify_triton_inference.call_args_list
        assert len(smoke_calls) == 2
        assert smoke_calls[0][0][0] == "fraud-detection-baseline"
        assert smoke_calls[0][0][3] == tmp_path / req_baseline
        assert smoke_calls[1][0][0] == "fraud-detection-candidate"
        assert smoke_calls[1][0][3] == tmp_path / req_candidate

    # ------------------------------------------------------------------
    # Multi-model: S3 secret applied once even when only one model uses S3
    # ------------------------------------------------------------------

    def test_deploy_s3_secret_applied_once_when_any_model_uses_s3(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _write_manifest(tmp_path)
        deploy_mod = self._fresh_deploy_mod()
        mocks = self._patch_all(monkeypatch, deploy_mod, tmp_path)

        req = _write_request(tmp_path)
        config = self._make_config(tmp_path, models=[
            {"name": "baseline",  "model_uri": "pvc://pvc/models",  "inference_request": req},
            {"name": "candidate", "model_uri": "models/biased",     "inference_request": req},
        ])
        deploy_mod.deploy(config)

        # S3 secret applied exactly once regardless of how many models need it.
        mocks["storage"].apply_s3_secret.assert_called_once()

    # ------------------------------------------------------------------
    # Multi-model: each ISVC receives the correct name from its model entry
    # ------------------------------------------------------------------

    def test_deploy_each_isvc_gets_correct_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _write_manifest(tmp_path)
        deploy_mod = self._fresh_deploy_mod()
        mocks = self._patch_all(monkeypatch, deploy_mod, tmp_path)

        req = _write_request(tmp_path)
        config = self._make_config(tmp_path, models=[
            {"name": "fraud-detection-baseline",  "model_uri": "pvc://pvc/models", "inference_request": req},
            {"name": "fraud-detection-candidate", "model_uri": "pvc://pvc/biased", "inference_request": req},
        ])
        deploy_mod.deploy(config)

        applied_dicts = [c[0][0] for c in mocks["ocp_resources"].apply_dict.call_args_list]
        assert applied_dicts[0]["metadata"]["name"] == "fraud-detection-baseline"
        assert applied_dicts[1]["metadata"]["name"] == "fraud-detection-candidate"

    # ------------------------------------------------------------------
    # EndpointUnreachable: deploy marks step skipped, does not raise
    # ------------------------------------------------------------------

    def test_deploy_endpoint_unreachable_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When verify_triton_inference raises EndpointUnreachable, deploy continues."""
        from rhoai.platform.inference import EndpointUnreachable

        _write_manifest(tmp_path)
        deploy_mod = self._fresh_deploy_mod()
        mocks = self._patch_all(monkeypatch, deploy_mod, tmp_path)
        monkeypatch.setattr(deploy_mod, "print_summary", MagicMock())

        mocks["inference"].verify_triton_inference.side_effect = EndpointUnreachable(
            "https://model.example.com/v2/models/fraud-detection/infer",
            "curl -sk ...",
        )

        req = _write_request(tmp_path)
        config = self._make_config(tmp_path, models=[
            {"name": "fraud-detection", "model_uri": "pvc://fraud-model-pvc/models",
             "inference_request": req},
        ])
        # Must not raise — deployment succeeded; only inference validation was skipped.
        deploy_mod.deploy(config)
        mocks["inference"].verify_triton_inference.assert_called_once()

    def test_deploy_summary_shows_skipped_when_unreachable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """print_summary receives a result with validation_skipped=True."""
        from rhoai.platform.inference import EndpointUnreachable
        from rhoai.usecases.fraud_detection.assets import ModelResult

        _write_manifest(tmp_path)
        deploy_mod = self._fresh_deploy_mod()
        mocks = self._patch_all(monkeypatch, deploy_mod, tmp_path)

        captured: list[ModelResult] = []
        monkeypatch.setattr(deploy_mod, "print_summary", lambda r, **_: captured.extend(r))

        mocks["inference"].verify_triton_inference.side_effect = EndpointUnreachable(
            "https://model.example.com/v2/models/fraud-detection/infer",
            "curl -sk ...",
        )

        req = _write_request(tmp_path)
        config = self._make_config(tmp_path, models=[
            {"name": "fraud-detection", "model_uri": "pvc://fraud-model-pvc/models",
             "inference_request": req},
        ])
        deploy_mod.deploy(config)

        assert len(captured) == 1
        assert captured[0].name == "fraud-detection"
        assert captured[0].validation_skipped is True
        assert captured[0].unreachable is not None

    def test_deploy_summary_shows_passed_when_reachable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """print_summary receives a result with validation_skipped=False on success."""
        from rhoai.usecases.fraud_detection.assets import ModelResult

        _write_manifest(tmp_path)
        deploy_mod = self._fresh_deploy_mod()
        mocks = self._patch_all(monkeypatch, deploy_mod, tmp_path)

        captured: list[ModelResult] = []
        monkeypatch.setattr(deploy_mod, "print_summary", lambda r, **_: captured.extend(r))

        req = _write_request(tmp_path)
        config = self._make_config(tmp_path, models=[
            {"name": "fraud-detection", "model_uri": "pvc://fraud-model-pvc/models",
             "inference_request": req},
        ])
        deploy_mod.deploy(config)

        assert len(captured) == 1
        assert captured[0].name == "fraud-detection"
        assert captured[0].validation_skipped is False
        assert captured[0].unreachable is None

class TestVerifyCmd:
    """Unit tests for _verify_cmd — the copy-pasteable follow-up command builder."""

    def test_no_config_file(self) -> None:
        from rhoai.usecases.fraud_detection.deploy import _verify_cmd
        cmd = _verify_cmd("fraud-detection", "")
        assert cmd == "rhoai usecase verify fraud-detection"
        assert "-c" not in cmd

    def test_with_config_file(self) -> None:
        from rhoai.usecases.fraud_detection.deploy import _verify_cmd
        cmd = _verify_cmd("fraud-detection", "config-fraud-detection.yaml")
        assert "rhoai usecase verify fraud-detection" in cmd
        assert "-c config-fraud-detection.yaml" in cmd

    def test_config_file_appears_in_follow_up_and_next(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When config_file is set, every printed command includes -c <path>."""
        from unittest.mock import call, patch
        from rhoai.usecases.fraud_detection.assets import ModelResult
        from rhoai.usecases.fraud_detection.deploy import print_summary

        printed: list[str] = []
        with patch("rhoai.usecases.fraud_detection.deploy._console") as mock_console:
            mock_console.print.side_effect = lambda s, *a, **kw: printed.append(str(s))
            print_summary(
                [ModelResult(name="m", validation_skipped=False)],
                use_case="fraud-detection",
                namespace="test-ns",
                config_file="path/to/config.yaml",
            )

        full_output = "\n".join(printed)
        # The Next section must include -c
        assert "-c path/to/config.yaml" in full_output

