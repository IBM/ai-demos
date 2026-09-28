"""Generate a Triton ``config.pbtxt`` from an ONNX (ONNX Runtime) model.

Used when a model is deployed from a local ONNX artifact without a
user-supplied ``config.pbtxt`` — the framework introspects the model's I/O
signature and renders a minimal, valid config for the ``onnxruntime_onnx``
backend.

Schema extraction (:func:`extract_schema`) and pbtxt rendering
(:func:`render_pbtxt`) are kept separate so a future backend can reuse the
renderer.  ``onnxruntime`` is an optional dependency (extra ``onnx``) imported
lazily — only models that actually need generation require it installed.
"""

from dataclasses import dataclass
from pathlib import Path

from rhoai.utils.logger import get_logger

log = get_logger(__name__)

PLATFORM = "onnxruntime_onnx"

# ONNX Runtime tensor element type strings -> Triton data types.
# https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto
_ONNX_TO_TRITON: dict[str, str] = {
    "tensor(bool)": "TYPE_BOOL",
    "tensor(uint8)": "TYPE_UINT8",
    "tensor(uint16)": "TYPE_UINT16",
    "tensor(uint32)": "TYPE_UINT32",
    "tensor(uint64)": "TYPE_UINT64",
    "tensor(int8)": "TYPE_INT8",
    "tensor(int16)": "TYPE_INT16",
    "tensor(int32)": "TYPE_INT32",
    "tensor(int64)": "TYPE_INT64",
    "tensor(float16)": "TYPE_FP16",
    "tensor(bfloat16)": "TYPE_BF16",
    "tensor(float)": "TYPE_FP32",
    "tensor(double)": "TYPE_FP64",
    "tensor(string)": "TYPE_STRING",
}


@dataclass
class TensorSpec:
    """One model input or output tensor.

    ``shape`` preserves the ONNX rank; every dynamic dimension (symbolic name
    or unknown) is normalized to ``-1``, matching Triton's variable-dim marker.
    """

    name: str
    datatype: str  # Triton TYPE_* string
    shape: list[int]


@dataclass
class ModelSchema:
    name: str
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]


# --------------------------------------------------------------------------- #
# Schema extraction (ONNX Runtime backend)
# --------------------------------------------------------------------------- #
def _normalize_dim(dim: object) -> int:
    """ONNX Runtime reports fixed dims as ``int`` and dynamic dims as a symbolic
    ``str`` (or ``None``). Triton represents any variable dimension as ``-1``."""
    if isinstance(dim, int) and dim > 0:
        return dim
    return -1


def _to_triton_dtype(onnx_type: str) -> str:
    try:
        return _ONNX_TO_TRITON[onnx_type]
    except KeyError:
        raise ValueError(f"Unsupported ONNX tensor type: {onnx_type!r}") from None


def extract_schema(model_path: Path, name: str | None = None) -> ModelSchema:
    """Load an ONNX model with ONNX Runtime and extract its I/O schema.

    ONNX Runtime does not expose the ONNX graph name, so the Triton model name
    defaults to the file stem. Pass ``name`` to override (deploy passes the
    InferenceService name so it matches the Triton model directory).

    Raises:
        RuntimeError: If ``onnxruntime`` is not installed.
        ValueError:   If a tensor uses an unsupported ONNX element type.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError(
            "onnxruntime is required to generate a config.pbtxt from an ONNX "
            "model. Install it with: pip install 'rhoai[onnx]' (or "
            "pip install onnxruntime), or supply 'config_path' explicitly."
        ) from None

    log.debug("Introspecting ONNX model: %s", model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def _specs(node_args) -> list[TensorSpec]:
        return [
            TensorSpec(
                name=arg.name,
                datatype=_to_triton_dtype(arg.type),
                shape=[_normalize_dim(d) for d in arg.shape],
            )
            for arg in node_args
        ]

    return ModelSchema(
        name=name or model_path.stem,
        inputs=_specs(session.get_inputs()),
        outputs=_specs(session.get_outputs()),
    )


def supports_batching(schema: ModelSchema) -> bool:
    """Whether every input and output has a dynamic leading dimension.

    A rank-1 tensor (``[batch]``) is fine — Triton represents it as ``dims: [1]``
    with ``reshape: { shape: [] }`` (see :func:`render_pbtxt`). Only a rank-0
    scalar, which has no leading dimension to carry the batch, is incompatible.

    Note: a dynamic leading dim is *necessary* but not *sufficient* for the
    caller to want batching — it may be a genuine variable-length dimension. The
    decision to batch is therefore an explicit caller choice (``max_batch_size``),
    and this predicate only guards against requesting it when it cannot work.
    """
    tensors = schema.inputs + schema.outputs
    if not tensors:
        return False
    return all(len(t.shape) >= 1 and t.shape[0] == -1 for t in tensors)


# --------------------------------------------------------------------------- #
# Rendering (backend-agnostic)
# --------------------------------------------------------------------------- #
def _render_tensor(spec: TensorSpec, dims: list[int], reshape_scalar: bool = False) -> str:
    dims_str = ", ".join(str(d) for d in dims)
    lines = [
        "  {",
        f'    name: "{spec.name}"',
        f"    data_type: {spec.datatype}",
        f"    dims: [ {dims_str} ]",
    ]
    if reshape_scalar:
        # A batch-only (rank-1) tensor has no per-sample dims once Triton strips
        # the batch dim; Triton requires dims: [1] reshaped back to a scalar.
        lines.append("    reshape: { shape: [ ] }")
    lines.append("  }")
    return "\n".join(lines)


def render_pbtxt(
    schema: ModelSchema,
    max_batch_size: int,
    dynamic_batching: bool,
) -> str:
    """Render a Triton ``config.pbtxt`` string for the ``onnxruntime_onnx`` backend.

    When ``max_batch_size > 0`` Triton owns the leading batch dimension, so each
    tensor's ``dims`` drops its first entry; a rank-1 (batch-only) tensor becomes
    ``dims: [1]`` with ``reshape: { shape: [] }``. When it is ``0`` the full ONNX
    shape (including any leading ``-1``) is emitted verbatim.
    """
    batched = max_batch_size > 0

    def render_one(spec: TensorSpec) -> str:
        if not batched:
            return _render_tensor(spec, spec.shape)
        dims = spec.shape[1:]
        if not dims:
            return _render_tensor(spec, [1], reshape_scalar=True)
        return _render_tensor(spec, dims)

    lines = [
        f'name: "{schema.name}"',
        f'platform: "{PLATFORM}"',
        f"max_batch_size: {max_batch_size}",
        "input [",
        ",\n".join(render_one(s) for s in schema.inputs),
        "]",
        "output [",
        ",\n".join(render_one(s) for s in schema.outputs),
        "]",
    ]
    if batched and dynamic_batching:
        lines.append("dynamic_batching {\n}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
def generate_pbtxt(
    model_path: Path,
    *,
    name: str | None = None,
    max_batch_size: int = 0,
    dynamic_batching: bool = True,
) -> str:
    """Extract the schema from an ONNX model and render its ``config.pbtxt``.

    ``max_batch_size`` defaults to ``0`` (no batching): a leading dynamic dim is
    ambiguous — it may be a batch dim or a real variable-length dim — so the
    model's signature is preserved verbatim unless the caller opts into batching.
    A value ``> 0`` requires a dynamic leading dim on every tensor; if that does
    not hold, it is silently reduced to ``0``.
    """
    schema = extract_schema(model_path, name=name)
    if max_batch_size > 0 and not supports_batching(schema):
        log.info(
            "Model '%s' has no dynamic leading dimension on every tensor — "
            "generating config with max_batch_size: 0.", schema.name,
        )
        max_batch_size = 0
    return render_pbtxt(schema, max_batch_size=max_batch_size, dynamic_batching=dynamic_batching)


def write_pbtxt(
    model_path: Path,
    output_path: Path,
    *,
    name: str | None = None,
    max_batch_size: int = 0,
    dynamic_batching: bool = True,
) -> Path:
    """Generate a ``config.pbtxt`` from an ONNX model and write it to ``output_path``.

    Returns the written path.
    """
    pbtxt = generate_pbtxt(
        model_path,
        name=name,
        max_batch_size=max_batch_size,
        dynamic_batching=dynamic_batching,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(pbtxt)
    log.info("Generated config.pbtxt for '%s' at %s", name or model_path.stem, output_path)
    return output_path
