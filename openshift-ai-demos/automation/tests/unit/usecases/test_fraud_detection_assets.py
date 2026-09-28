"""Unit tests for rhoai.usecases.fraud_detection.assets."""

from pathlib import Path

import pytest

from rhoai.usecases.fraud_detection.assets import (
    observation_batch_size,
    triton_file_map,
    triton_pvc_uri,
    validate_model_config,
)


class TestTritonFileMap:
    """triton_file_map encodes the Triton directory layout.

    The platform layer (storage.copy_files_to_pvc) is format-agnostic and
    accepts a plain {dest: src} dict.  These tests verify that assets.py
    produces the correct Triton-specific layout.
    """

    def test_model_file_placed_under_versioned_subdir(self, tmp_path: Path) -> None:
        model  = tmp_path / "model.onnx"
        config = tmp_path / "config.pbtxt"
        model.write_bytes(b"")
        config.write_bytes(b"")

        file_map = triton_file_map("my-model", model, config)

        model_dests = [k for k in file_map if "model.onnx" in k]
        assert len(model_dests) == 1
        # Must sit under the versioned sub-directory, e.g. /mnt/pvc/my-model/1/
        assert "/my-model/1/model.onnx" in model_dests[0]

    def test_config_file_placed_directly_under_model_dir(self, tmp_path: Path) -> None:
        model  = tmp_path / "model.onnx"
        config = tmp_path / "config.pbtxt"
        model.write_bytes(b"")
        config.write_bytes(b"")

        file_map = triton_file_map("my-model", model, config)

        config_dests = [k for k in file_map if "config.pbtxt" in k]
        assert len(config_dests) == 1
        # Must sit directly under the model directory, NOT under the version dir
        assert "/my-model/config.pbtxt" in config_dests[0]
        assert "/1/" not in config_dests[0]

    def test_model_name_used_as_top_level_directory(self, tmp_path: Path) -> None:
        model  = tmp_path / "model.onnx"
        config = tmp_path / "config.pbtxt"
        model.write_bytes(b"")
        config.write_bytes(b"")

        file_map = triton_file_map("fraud-detection", model, config)

        assert all("fraud-detection" in k for k in file_map)

    def test_source_paths_map_to_original_local_files(self, tmp_path: Path) -> None:
        model  = tmp_path / "model.onnx"
        config = tmp_path / "config.pbtxt"
        model.write_bytes(b"")
        config.write_bytes(b"")

        file_map = triton_file_map("m", model, config)

        assert set(file_map.values()) == {model, config}

    def test_model_filename_is_preserved(self, tmp_path: Path) -> None:
        """The model filename (not hardcoded 'model.onnx') is used as-is."""
        model  = tmp_path / "fraud_v2_quant.onnx"
        config = tmp_path / "config.pbtxt"
        model.write_bytes(b"")
        config.write_bytes(b"")

        file_map = triton_file_map("m", model, config)

        model_dests = [k for k in file_map if "fraud_v2_quant.onnx" in k]
        assert len(model_dests) == 1

    def test_returns_exactly_two_entries(self, tmp_path: Path) -> None:
        model  = tmp_path / "model.onnx"
        config = tmp_path / "config.pbtxt"
        model.write_bytes(b"")
        config.write_bytes(b"")

        assert len(triton_file_map("m", model, config)) == 2


class TestTritonPvcUri:
    def test_returns_pvc_root_uri(self) -> None:
        # Must point to the PVC root — no model sub-path.
        # KServe mounts the PVC root as /mnt/models; Triton then finds
        # <model_name>/ as a direct child, which is what --load-model needs.
        assert triton_pvc_uri("my-pvc", "fraud-model") == "pvc://my-pvc"

    def test_uri_contains_pvc_name(self) -> None:
        uri = triton_pvc_uri("fraud-model-pvc", "m")
        assert "fraud-model-pvc" in uri

    def test_uri_does_not_contain_model_subpath(self) -> None:
        # A sub-path would cause KServe to descend into the model directory
        # before mounting, leaving Triton unable to find the model repository.
        uri = triton_pvc_uri("my-pvc", "demo-loan-nn-onnx")
        assert uri == "pvc://my-pvc"
        assert "demo-loan-nn-onnx" not in uri


class TestObservationBatchSize:
    def test_defaults_to_one_when_unset(self) -> None:
        assert observation_batch_size({"name": "m"}) == 1

    def test_defaults_to_one_when_no_observations_block(self) -> None:
        model = {"name": "m", "bias_monitoring": {"spd_monitors": []}}
        assert observation_batch_size(model) == 1

    def test_reads_nested_value(self) -> None:
        model = {"name": "m", "bias_monitoring": {"observations": {"batch_size": 50}}}
        assert observation_batch_size(model) == 50


class TestValidateModelConfig:
    """validate_model_config enforces the JSON-mode / dataset-mode contract."""

    # --- accepted configurations ---

    def test_json_mode_request_only_is_valid(self) -> None:
        validate_model_config({"name": "m", "inference_request": "req.json"})

    def test_json_mode_with_explicit_observation_source_is_valid(self) -> None:
        validate_model_config({
            "name": "m",
            "inference_request": "req.json",
            "bias_monitoring": {"observations": {"path": "obs/"}},
        })

    def test_dataset_mode_with_pbtxt_is_valid(self) -> None:
        validate_model_config({
            "name": "m",
            "inference_dataset": "data.csv",
            "inference_config_path": "config.pbtxt",
        })

    def test_dataset_mode_with_batch_size_is_valid(self) -> None:
        validate_model_config({
            "name": "m",
            "inference_dataset": "data.csv",
            "config_path": "config.pbtxt",
            "bias_monitoring": {"observations": {"batch_size": 50}},
        })

    def test_dataset_mode_with_model_path_is_valid(self) -> None:
        # model_path (ONNX) satisfies the pbtxt requirement: the framework
        # generates a config.pbtxt from the model during deployment.
        validate_model_config({
            "name": "m",
            "inference_dataset": "data.csv",
            "model_path": "model.onnx",
        })

    # --- rejected combinations ---

    def test_rejects_both_request_and_dataset(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            validate_model_config({
                "name": "m",
                "inference_request": "req.json",
                "inference_dataset": "data.csv",
            })

    def test_rejects_neither_request_nor_dataset(self) -> None:
        with pytest.raises(ValueError, match="either 'inference_request'"):
            validate_model_config({"name": "m"})

    def test_rejects_dataset_with_explicit_observation_source(self) -> None:
        with pytest.raises(ValueError, match="cannot be combined with 'inference_dataset'"):
            validate_model_config({
                "name": "m",
                "inference_dataset": "data.csv",
                "config_path": "config.pbtxt",
                "bias_monitoring": {"observations": {"path": "obs/"}},
            })

    def test_rejects_request_with_batch_size(self) -> None:
        with pytest.raises(ValueError, match="not applicable"):
            validate_model_config({
                "name": "m",
                "inference_request": "req.json",
                "bias_monitoring": {"observations": {"batch_size": 50}},
            })

    def test_rejects_dataset_without_pbtxt(self) -> None:
        with pytest.raises(ValueError, match="requires a Triton 'config.pbtxt'"):
            validate_model_config({"name": "m", "inference_dataset": "data.csv"})

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "50", True])
    def test_rejects_invalid_batch_size(self, bad: object) -> None:
        with pytest.raises(ValueError, match="integer >= 1"):
            validate_model_config({
                "name": "m",
                "inference_dataset": "data.csv",
                "config_path": "config.pbtxt",
                "bias_monitoring": {"observations": {"batch_size": bad}},
            })
