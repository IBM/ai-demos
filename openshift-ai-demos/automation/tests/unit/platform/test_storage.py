"""Unit tests for rhoai.platform.storage."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rhoai.platform import storage


@pytest.fixture(autouse=True)
def patch_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rhoai.platform.storage.resources", MagicMock())
    monkeypatch.setattr("rhoai.platform.storage.wait", MagicMock())


class TestCreateSecret:
    def test_base64_encodes_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import base64
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.storage.resources", mock_resources)

        storage.create_secret("my-secret", "ns", {"user": "admin", "pass": "secret"})

        call_args = mock_resources.apply_dict.call_args[0][0]
        assert call_args["kind"] == "Secret"
        assert call_args["data"]["user"] == base64.b64encode(b"admin").decode()
        assert call_args["data"]["pass"] == base64.b64encode(b"secret").decode()

    def test_default_type_is_opaque(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.storage.resources", mock_resources)
        storage.create_secret("s", "ns", {})
        call_args = mock_resources.apply_dict.call_args[0][0]
        assert call_args["type"] == "Opaque"


class TestCreatePvc:
    def test_skips_if_already_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.exists.return_value = True
        monkeypatch.setattr("rhoai.platform.storage.resources", mock_resources)
        result = storage.create_pvc("my-pvc", "ns", "10Gi")
        mock_resources.apply_dict.assert_not_called()
        assert result is False

    def test_creates_pvc_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.exists.return_value = False
        monkeypatch.setattr("rhoai.platform.storage.resources", mock_resources)
        result = storage.create_pvc("my-pvc", "ns", "10Gi")
        mock_resources.apply_dict.assert_called_once()
        assert result is True

    def test_raises_if_storage_class_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        # PVC does not exist (so we proceed), StorageClass does not exist (should raise)
        mock_resources.exists.side_effect = lambda kind, name, *_: False
        monkeypatch.setattr("rhoai.platform.storage.resources", mock_resources)
        with pytest.raises(RuntimeError, match="StorageClass"):
            storage.create_pvc("pvc", "ns", "5Gi", storage_class="missing-class")


class TestVerify:
    def test_passes_when_secret_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.exists.return_value = True
        monkeypatch.setattr("rhoai.platform.storage.resources", mock_resources)
        storage.verify("s3-credentials", "ns")  # should not raise

    def test_raises_when_secret_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.exists.return_value = False
        monkeypatch.setattr("rhoai.platform.storage.resources", mock_resources)
        with pytest.raises(RuntimeError, match="S3 credentials"):
            storage.verify("s3-credentials", "ns")


class TestValidateLocalArtifacts:
    def test_passes_when_all_files_exist(self, tmp_path: Path) -> None:
        f1 = tmp_path / "model.onnx"
        f2 = tmp_path / "config.pbtxt"
        f1.write_bytes(b"")
        f2.write_bytes(b"")
        storage.validate_local_artifacts(f1, f2)  # should not raise

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        existing = tmp_path / "model.onnx"
        existing.write_bytes(b"")
        missing = tmp_path / "config.pbtxt"
        with pytest.raises(FileNotFoundError, match="config.pbtxt"):
            storage.validate_local_artifacts(existing, missing)

    def test_raises_on_first_missing_when_both_absent(self, tmp_path: Path) -> None:
        a = tmp_path / "a.onnx"
        b = tmp_path / "b.pbtxt"
        with pytest.raises(FileNotFoundError, match="a.onnx"):
            storage.validate_local_artifacts(a, b)

    def test_passes_with_no_paths(self) -> None:
        storage.validate_local_artifacts()  # variadic — zero args is valid


class TestCopyFilesToPvc:
    """Tests for copy_files_to_pvc.

    The function creates a staging Pod, waits for Running, copies each file
    via oc cp at the caller-supplied destination paths, and deletes the Pod.
    It is format-agnostic: the caller decides the layout.
    """

    def _make_files(self, tmp_path: Path) -> tuple[Path, Path]:
        f1 = tmp_path / "model.onnx"
        f2 = tmp_path / "config.pbtxt"
        f1.write_bytes(b"fake-model")
        f2.write_bytes(b"fake-config")
        return f1, f2

    def _ok_proc(self) -> MagicMock:
        m = MagicMock()
        m.returncode = 0
        return m

    def _running_resources(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        mock = MagicMock()
        mock.status.return_value = {"phase": "Running"}
        monkeypatch.setattr("rhoai.platform.storage.resources", mock)
        return mock

    def test_copies_each_file_in_the_supplied_map(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f1, f2 = self._make_files(tmp_path)
        self._running_resources(monkeypatch)

        captured: list[list[str]] = []

        def fake_run(cmd: list[str], **kw: object) -> MagicMock:
            captured.append(cmd)
            return self._ok_proc()

        monkeypatch.setattr("subprocess.run", fake_run)

        storage.copy_files_to_pvc(
            "my-pvc", "ns",
            {"/dest/a/model.onnx": f1, "/dest/b/config.pbtxt": f2},
        )

        cp_dests = [c[-1].split(":")[-1] for c in captured if c[:2] == ["oc", "cp"]]
        assert "/dest/a/model.onnx" in cp_dests
        assert "/dest/b/config.pbtxt" in cp_dests

    def test_staging_pod_deleted_on_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f1, _ = self._make_files(tmp_path)
        mock_res = self._running_resources(monkeypatch)
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: self._ok_proc())

        storage.copy_files_to_pvc("my-pvc", "ns", {"/dest/f": f1})

        pod_deletes = [
            c for c in mock_res.delete_manifest.call_args_list if c.args[0] == "Pod"
        ]
        assert len(pod_deletes) >= 1

    def test_staging_pod_deleted_on_oc_cp_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pod must be cleaned up even when oc cp fails."""
        f1, _ = self._make_files(tmp_path)
        mock_res = self._running_resources(monkeypatch)

        fail_proc = MagicMock()
        fail_proc.returncode = 1
        fail_proc.stderr = "permission denied"
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: fail_proc)

        with pytest.raises(RuntimeError):
            storage.copy_files_to_pvc("my-pvc", "ns", {"/dest/f": f1})

        pod_deletes = [
            c for c in mock_res.delete_manifest.call_args_list if c.args[0] == "Pod"
        ]
        assert len(pod_deletes) >= 1

    def test_pod_name_truncated_to_63_chars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f1, _ = self._make_files(tmp_path)
        mock_res = self._running_resources(monkeypatch)
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: self._ok_proc())

        storage.copy_files_to_pvc("x" * 70, "ns", {"/d/f": f1})

        applied = mock_res.apply_dict.call_args[0][0]
        assert len(applied["metadata"]["name"]) <= 63

    def test_raises_when_pod_does_not_reach_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f1, _ = self._make_files(tmp_path)
        mock_res = MagicMock()
        mock_res.status.return_value = {"phase": "Pending"}
        monkeypatch.setattr("rhoai.platform.storage.resources", mock_res)
        monkeypatch.setattr("time.sleep", lambda _: None)

        with pytest.raises(RuntimeError, match="did not reach Running"):
            storage.copy_files_to_pvc("my-pvc", "ns", {"/d/f": f1}, timeout=0)


class TestDeployModelStagingBranch:
    """Tests for the staging branch inside _deploy_model.

    Exercises mutual-exclusivity validation without hitting the cluster.
    """

    def test_raises_when_model_uri_and_model_path_both_set(
        self, tmp_path: Path
    ) -> None:
        from rhoai.usecases.fraud_detection.deploy import _deploy_model

        model_file  = tmp_path / "model.onnx"
        model_file.write_bytes(b"")
        config_file = tmp_path / "config.pbtxt"
        config_file.write_bytes(b"")

        model = {
            "name": "m",
            "model_uri":   "pvc://existing/m",
            "model_path":  str(model_file),
            "config_path": str(config_file),
        }
        with pytest.raises(ValueError, match="mutually exclusive"):
            _deploy_model(model, "/repo", "plat-ns", "work-ns", 1)

    def test_model_path_without_config_path_is_allowed(self) -> None:
        """config_path is optional — model_path alone proceeds past validation
        (a missing model file then fails artifact validation, not the guard)."""
        from rhoai.usecases.fraud_detection.deploy import _deploy_model

        model = {"name": "m", "model_path": "/some/model.onnx"}
        with pytest.raises(FileNotFoundError):
            _deploy_model(model, "/repo", "plat-ns", "work-ns", 1)

    def test_raises_when_only_config_path_set(self) -> None:
        from rhoai.usecases.fraud_detection.deploy import _deploy_model

        model = {"name": "m", "config_path": "/some/config.pbtxt"}
        with pytest.raises(ValueError, match="config_path requires model_path"):
            _deploy_model(model, "/repo", "plat-ns", "work-ns", 1)

    def test_generate_staging_pbtxt_writes_and_records_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """_generate_staging_pbtxt renders a config.pbtxt from the ONNX model,
        names it after the ISVC, and records config_path back on the model."""
        from rhoai.platform import config_generator
        from rhoai.usecases.fraud_detection.deploy import _generate_staging_pbtxt

        captured: dict = {}

        def _fake_generate(model_path, *, name, max_batch_size, dynamic_batching):
            captured.update(
                model_path=model_path, name=name,
                max_batch_size=max_batch_size, dynamic_batching=dynamic_batching,
            )
            return f'name: "{name}"\nmax_batch_size: {max_batch_size}\n'

        monkeypatch.setattr(config_generator, "generate_pbtxt", _fake_generate)

        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"")
        model = {"name": "loan", "model_path": str(model_file), "max_batch_size": 4}

        result = _generate_staging_pbtxt(model, "loan", str(tmp_path))

        assert result.name == "config.pbtxt"                 # Triton requires this name
        assert result.parent.name == "loan"                  # per-model subdirectory
        assert "inputs" in result.parts                      # persistent, discoverable
        assert result.exists()
        assert model["config_path"] == str(result)          # recorded for downstream
        assert captured["name"] == "loan"                    # Triton name = ISVC name
        assert captured["max_batch_size"] == 4               # per-model override honored
        assert captured["dynamic_batching"] is True          # default


class TestDeleteStagingPod:
    """Tests for delete_staging_pod.

    Verifies that the function derives the pod name from the PVC name
    using the same formula as copy_files_to_pvc, then deletes and waits.
    """

    def test_derives_pod_name_from_pvc_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_resources = MagicMock()
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.storage.resources", mock_resources)
        monkeypatch.setattr("rhoai.platform.storage.wait", mock_wait)

        storage.delete_staging_pod("my-pvc", "test-ns")

        delete_call = mock_resources.delete_manifest.call_args
        assert delete_call.args[0] == "Pod"
        assert "my-pvc" in delete_call.args[1]

    def test_calls_wait_until_deleted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_resources = MagicMock()
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.storage.resources", mock_resources)
        monkeypatch.setattr("rhoai.platform.storage.wait", mock_wait)

        storage.delete_staging_pod("my-pvc", "test-ns")

        mock_wait.wait_until_deleted.assert_called_once()
        kind, name, ns = mock_wait.wait_until_deleted.call_args.args
        assert kind == "Pod"
        assert "my-pvc" in name
        assert ns == "test-ns"

    def test_pod_name_matches_copy_files_naming(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The name deleted must be the same name copy_files_to_pvc creates."""
        from rhoai.platform.storage import _STAGING_POD_PREFIX, _K8S_NAME_MAX

        mock_resources = MagicMock()
        mock_resources.status.return_value = {"phase": "Running"}
        monkeypatch.setattr("rhoai.platform.storage.resources", mock_resources)
        monkeypatch.setattr("rhoai.platform.storage.wait", MagicMock())
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: MagicMock(returncode=0))

        pvc_name = "fraud-model-pvc"
        expected_pod = f"{_STAGING_POD_PREFIX}-{pvc_name}"[:_K8S_NAME_MAX]

        # Capture which pod name copy_files_to_pvc creates
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "f"
            f.write_bytes(b"")
            storage.copy_files_to_pvc(pvc_name, "ns", {"/d/f": f})

        created_pod = mock_resources.apply_dict.call_args[0][0]["metadata"]["name"]

        # Now capture which pod name delete_staging_pod deletes
        storage.delete_staging_pod(pvc_name, "ns")
        deleted_pod = mock_resources.delete_manifest.call_args_list[-1].args[1]

        assert created_pod == deleted_pod == expected_pod

    def test_pod_name_truncated_to_63_chars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.storage.resources", mock_resources)
        monkeypatch.setattr("rhoai.platform.storage.wait", MagicMock())

        storage.delete_staging_pod("x" * 70, "ns")

        deleted_name = mock_resources.delete_manifest.call_args.args[1]
        assert len(deleted_name) <= 63
