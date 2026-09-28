"""Unit tests for rhoai.platform.config_generator — ONNX → Triton config.pbtxt."""

import sys
import types
from pathlib import Path

import pytest

from rhoai.platform import config_generator as cg
from rhoai.platform.config_generator import (
    ModelSchema,
    TensorSpec,
    generate_pbtxt,
    render_pbtxt,
    supports_batching,
    write_pbtxt,
)

# ---------------------------------------------------------------------------
# Helpers — a fake onnxruntime module for extract_schema
# ---------------------------------------------------------------------------

class _FakeNodeArg:
    def __init__(self, name: str, type_: str, shape: list):
        self.name = name
        self.type = type_
        self.shape = shape


def _install_fake_ort(monkeypatch, inputs, outputs):
    """Install a fake ``onnxruntime`` module whose session returns given I/O."""
    class _Session:
        def __init__(self, path, providers=None):
            pass

        def get_inputs(self):
            return inputs

        def get_outputs(self):
            return outputs

    fake = types.ModuleType("onnxruntime")
    fake.InferenceSession = _Session
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)


# ---------------------------------------------------------------------------
# render_pbtxt
# ---------------------------------------------------------------------------

def test_render_unbatched_preserves_full_shape():
    schema = ModelSchema(
        name="loan",
        inputs=[TensorSpec("customer_data_input", "TYPE_FP64", [-1, 11])],
        outputs=[TensorSpec("predict", "TYPE_INT64", [-1])],
    )
    out = render_pbtxt(schema, max_batch_size=0, dynamic_batching=True)

    assert 'name: "loan"' in out
    assert 'platform: "onnxruntime_onnx"' in out
    assert "max_batch_size: 0" in out
    assert "dims: [ -1, 11 ]" in out
    assert "dims: [ -1 ]" in out
    assert "reshape" not in out          # no batching → no reshape
    assert "dynamic_batching" not in out  # only emitted when batching


def test_render_batched_strips_leading_dim_and_reshapes_rank1():
    schema = ModelSchema(
        name="upi",
        inputs=[TensorSpec("f", "TYPE_FP32", [-1, 1])],
        outputs=[
            TensorSpec("label", "TYPE_INT64", [-1]),
            TensorSpec("probs", "TYPE_FP32", [-1, 2]),
        ],
    )
    out = render_pbtxt(schema, max_batch_size=4, dynamic_batching=True)

    assert "max_batch_size: 4" in out
    assert "dims: [ 1 ]" in out                 # input [-1,1] → [1]
    assert "reshape: { shape: [ ] }" in out     # rank-1 output [-1] → [1] + reshape
    assert "dims: [ 2 ]" in out                 # output [-1,2] → [2]
    assert "dynamic_batching {\n}" in out


def test_render_batched_without_dynamic_batching_block():
    schema = ModelSchema(
        name="m",
        inputs=[TensorSpec("in", "TYPE_FP32", [-1, 3])],
        outputs=[TensorSpec("out", "TYPE_FP32", [-1, 3])],
    )
    out = render_pbtxt(schema, max_batch_size=8, dynamic_batching=False)
    assert "dynamic_batching" not in out


# ---------------------------------------------------------------------------
# supports_batching
# ---------------------------------------------------------------------------

def test_supports_batching_true_when_all_leading_dims_dynamic():
    schema = ModelSchema(
        name="m",
        inputs=[TensorSpec("in", "TYPE_FP32", [-1, 4])],
        outputs=[TensorSpec("out", "TYPE_FP32", [-1])],  # rank-1 is fine
    )
    assert supports_batching(schema) is True


def test_supports_batching_false_when_leading_dim_fixed():
    schema = ModelSchema(
        name="m",
        inputs=[TensorSpec("in", "TYPE_FP32", [1, 4])],
        outputs=[TensorSpec("out", "TYPE_FP32", [-1])],
    )
    assert supports_batching(schema) is False


def test_supports_batching_false_on_scalar_tensor():
    schema = ModelSchema(
        name="m",
        inputs=[TensorSpec("in", "TYPE_FP32", [])],  # rank-0 scalar
        outputs=[TensorSpec("out", "TYPE_FP32", [-1])],
    )
    assert supports_batching(schema) is False


# ---------------------------------------------------------------------------
# extract_schema (via fake onnxruntime)
# ---------------------------------------------------------------------------

def test_extract_schema_maps_types_and_normalizes_dynamic_dims(monkeypatch):
    _install_fake_ort(
        monkeypatch,
        inputs=[_FakeNodeArg("x", "tensor(double)", ["batch", 11])],
        outputs=[_FakeNodeArg("y", "tensor(int64)", ["batch"])],
    )
    schema = cg.extract_schema(Path("model.onnx"), name="loan")

    assert schema.name == "loan"
    assert schema.inputs[0].datatype == "TYPE_FP64"
    assert schema.inputs[0].shape == [-1, 11]     # symbolic "batch" → -1
    assert schema.outputs[0].datatype == "TYPE_INT64"
    assert schema.outputs[0].shape == [-1]


def test_extract_schema_defaults_name_to_file_stem(monkeypatch):
    _install_fake_ort(
        monkeypatch,
        inputs=[_FakeNodeArg("x", "tensor(float)", [-1, 2])],
        outputs=[_FakeNodeArg("y", "tensor(float)", [-1, 2])],
    )
    schema = cg.extract_schema(Path("/tmp/mymodel.onnx"))
    assert schema.name == "mymodel"


def test_extract_schema_rejects_unsupported_type(monkeypatch):
    _install_fake_ort(
        monkeypatch,
        inputs=[_FakeNodeArg("x", "tensor(complex64)", [-1, 2])],
        outputs=[_FakeNodeArg("y", "tensor(float)", [-1, 2])],
    )
    with pytest.raises(ValueError, match="Unsupported ONNX tensor type"):
        cg.extract_schema(Path("model.onnx"))


def test_extract_schema_raises_without_onnxruntime(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # forces ImportError
    with pytest.raises(RuntimeError, match="onnxruntime is required"):
        cg.extract_schema(Path("model.onnx"))


# ---------------------------------------------------------------------------
# generate_pbtxt / write_pbtxt
# ---------------------------------------------------------------------------

def test_generate_pbtxt_defaults_to_no_batching(monkeypatch):
    _install_fake_ort(
        monkeypatch,
        inputs=[_FakeNodeArg("x", "tensor(double)", ["batch", 11])],
        outputs=[_FakeNodeArg("y", "tensor(int64)", ["batch"])],
    )
    out = generate_pbtxt(Path("model.onnx"), name="loan")
    assert "max_batch_size: 0" in out
    assert "dims: [ -1, 11 ]" in out


def test_generate_pbtxt_downgrades_when_batching_impossible(monkeypatch):
    # Fixed leading dim → batching requested but not possible → falls back to 0.
    _install_fake_ort(
        monkeypatch,
        inputs=[_FakeNodeArg("x", "tensor(float)", [1, 4])],
        outputs=[_FakeNodeArg("y", "tensor(float)", [1, 2])],
    )
    out = generate_pbtxt(Path("model.onnx"), name="m", max_batch_size=8)
    assert "max_batch_size: 0" in out


def test_write_pbtxt_writes_file(monkeypatch, tmp_path):
    _install_fake_ort(
        monkeypatch,
        inputs=[_FakeNodeArg("x", "tensor(float)", [-1, 4])],
        outputs=[_FakeNodeArg("y", "tensor(float)", [-1, 2])],
    )
    dest = tmp_path / "sub" / "config.pbtxt"
    returned = write_pbtxt(Path("model.onnx"), dest, name="m", max_batch_size=4)

    assert returned == dest
    assert dest.exists()
    text = dest.read_text()
    assert 'name: "m"' in text
    assert "max_batch_size: 4" in text
