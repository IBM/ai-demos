"""Fraud Detection use-case assets.

Three responsibilities:

    Path resolution  — the only file that knows where fraud-detection
                       manifests live on disk.  Shared platform manifests
                       are delegated to platform.manifests.

    Triton layout    — builds the file map that storage.copy_files_to_pvc
                       needs to construct the Triton repository structure on a PVC.
                       The ``<model>/1/<file>`` convention lives here, not in
                       platform/storage.py (which is format-agnostic).

    Use-case types   — data types and config helpers shared by deploy.py
                       and verify.py so neither module depends on the other.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rhoai.platform import manifests
from rhoai.platform.inference import EndpointUnreachable

# Triton requires model files under a versioned sub-directory.
# Version "1" is the conventional first version used throughout this framework.
_TRITON_MODEL_VERSION = "1"


def serving_runtime_name(model_name: str) -> str:
    """Return the ServingRuntime resource name for a given model.

    Convention: ``triton-<model_name>``.  Each model deployment creates its
    own ServingRuntime so that Triton's ``--load-model`` flag is correct for
    that model and runtimes never overwrite each other.

    This value is passed as RUNTIME_NAME to the Triton Template and must match
    the ``spec.predictor.model.runtime`` field in the InferenceService manifest.
    """
    return f"triton-{model_name}"


def get_serving_runtime_template(repo_root: str | Path) -> Path:
    """Return the path to the Triton ServingRuntime OpenShift Template.

    The Template is processed via ``oc process`` before the ServingRuntime
    CR is created — see platform.inference.apply_serving_runtime_from_template().

    Returns:
        Path to model-serving/predictive-models/triton/triton-ppc64le-runtime-template.yaml
    """
    return manifests.get_triton_serving_runtime_template(repo_root)


def get_model_manifest(repo_root: str | Path) -> Path:
    """Return the path to the Fraud Detection InferenceService manifest.

    Returns:
        Path to model-serving/predictive-models/triton/fraud-detection.yaml
    """
    return (
        Path(repo_root)
        / "model-serving"
        / "predictive-models"
        / "triton"
        / "fraud-detection.yaml"
    )


def triton_file_map(
    model_name: str,
    model_file: Path,
    config_file: Path,
) -> dict[str, Path]:
    """Return the file map for a Triton model repository on a PVC.

    Encodes the Triton directory layout convention so that no other module
    needs to know about it.  Pass the returned dict directly to
    ``storage.copy_files_to_pvc()``.

    The resulting structure on the PVC will be:

        <model_name>/
          config.pbtxt          ← config_file
          1/
            <model_file.name>   ← model_file  (filename preserved)

    Args:
        model_name:  Top-level directory name on the PVC (matches the ISVC name).
        model_file:  Local path to the model file (e.g. model.onnx).
        config_file: Local path to the Triton config file (config.pbtxt).

    Returns:
        Dict mapping absolute PVC destination paths to local source Paths,
        ready to pass to ``storage.copy_files_to_pvc(files=...)``.
    """
    base = f"/mnt/pvc/{model_name}"
    return {
        f"{base}/{_TRITON_MODEL_VERSION}/{model_file.name}": model_file,
        f"{base}/{config_file.name}": config_file,
    }


def triton_pvc_uri(pvc_name: str, model_name: str) -> str:  # noqa: ARG001
    """Return the KServe ``pvc://`` URI for a staged Triton model.

    KServe resolves ``pvc://<name>`` by mounting the PVC root as
    ``/mnt/models``.  Triton is started with
    ``--model-repository=/mnt/models``, so it expects to find the model
    directory (e.g. ``<model_name>/``) *directly inside* that mount point.

    The staging layout on the PVC is therefore::

        <model_name>/
          config.pbtxt
          1/
            model.onnx

    which Triton sees as ``/mnt/models/<model_name>/`` — exactly what
    ``--load-model=<model_name>`` needs.

    A sub-path in the URI (``pvc://<name>/<model_name>``) would cause KServe
    to descend into the model directory before mounting, presenting its
    *contents* as ``/mnt/models``.  Triton would then look for a sub-directory
    named ``<model_name>`` inside the model directory itself and fail with
    "model not found in any model repository".

    Args:
        pvc_name:   Name of the PVC the model was staged onto.
        model_name: Unused — kept for a consistent call-site signature so
                    callers do not need to special-case this function.

    Returns:
        ``pvc://<pvc_name>``  (no sub-path)
    """
    return f"pvc://{pvc_name}"


# ---------------------------------------------------------------------------
# Use-case types
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Outcome of a single model deploy or verify operation."""
    name:               str
    model_uri:          str                        = ""
    endpoint:           str                        = ""
    inference_input:    dict[str, Any] | None      = field(default=None, repr=False)
    inference_output:   dict[str, Any] | None      = field(default=None, repr=False)
    curl_cmd:           str                        = ""
    request_path:       Path | None                = field(default=None, repr=False)
    # Local artifacts the framework generated for this model (shown in the
    # summary so users can inspect them). None when the user supplied their own.
    generated_config_path:  Path | None            = None
    generated_request_path: Path | None            = None
    validation_skipped: bool                       = False
    unreachable:        EndpointUnreachable | None = field(default=None, repr=False)


def resolve_inference_request(model: dict[str, Any]) -> Path | None:
    """Return the Path to this model's inference request file, if set.

    The configured ``inference_request`` is used as given — supply an absolute
    path (matching how ``model_path`` / ``config_path`` are supplied).
    """
    path = model.get("inference_request", "")
    if not path:
        return None
    return Path(path)


def resolve_inference_dataset(model: dict[str, Any]) -> Path:
    """Return the Path to the dataset used to auto-generate a request.

    The configured ``inference_dataset`` is used as given — supply an absolute
    path (matching how ``model_path`` / ``config_path`` are supplied).
    """
    path = model.get("inference_dataset", "")
    if not path:
        raise ValueError(
            f"Model '{model.get('name', '?')}' has no inference_request or inference_dataset configured. "
            "Set inference_request: <absolute json/csv path> or inference_dataset: <absolute csv path>."
        )
    return Path(path)


def observation_batch_size(model: dict[str, Any]) -> int:
    """Return observations-per-request for dataset mode.

    Reads ``bias_monitoring.observations.batch_size``, defaulting to 1 when
    unset.  Only meaningful in dataset mode — JSON mode rejects the key during
    validation (see :func:`validate_model_config`).
    """
    obs = (model.get("bias_monitoring") or {}).get("observations") or {}
    return int(obs.get("batch_size", 1))


def validate_model_config(model: dict[str, Any]) -> None:
    """Validate a model's inference-input configuration, failing fast on ambiguity.

    Two mutually exclusive modes select how the smoke-test request and (when
    bias monitoring is enabled) the observation requests are produced:

    * **JSON mode** — ``inference_request`` points at a pre-built KServe v2 JSON
      (or a CSV converted with the model's schema).  Observations, if any, are
      declared explicitly under ``bias_monitoring.observations.path``/``files``.
    * **Dataset mode** — ``inference_dataset`` is the single source of truth:
      the smoke test is the first generated request and observations are the
      whole dataset, batched by ``bias_monitoring.observations.batch_size``.
      Requires a Triton ``config.pbtxt`` via ``inference_config_path`` (or
      ``config_path``).

    Rejected combinations:

    1. ``inference_request`` and ``inference_dataset`` both set (ambiguous mode).
    2. neither set (no input source).
    3. ``inference_dataset`` + ``observations.path``/``files`` (dataset already
       supplies observations).
    4. ``inference_request`` + ``observations.batch_size`` (batching only
       applies to dataset mode).
    5. ``inference_dataset`` without a ``config.pbtxt`` schema source.

    Args:
        model: A single entry from ``deployment.models``.

    Raises:
        ValueError: If the configuration is ambiguous or incomplete.
    """
    name        = model.get("name", "?")
    has_request = bool(model.get("inference_request"))
    has_dataset = bool(model.get("inference_dataset"))

    obs            = (model.get("bias_monitoring") or {}).get("observations") or {}
    has_obs_source = bool(obs.get("path")) or bool(obs.get("files"))
    has_obs_batch  = "batch_size" in obs

    if has_request and has_dataset:
        raise ValueError(
            f"Model '{name}': 'inference_request' and 'inference_dataset' are mutually "
            "exclusive — set exactly one."
        )
    if not has_request and not has_dataset:
        raise ValueError(
            f"Model '{name}': set either 'inference_request' (JSON/CSV) or "
            "'inference_dataset' (CSV/JSON)."
        )

    if has_obs_batch:
        batch = obs["batch_size"]
        if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
            raise ValueError(
                f"Model '{name}': 'bias_monitoring.observations.batch_size' must be an "
                f"integer >= 1, got {batch!r}."
            )

    if has_request and has_obs_batch:
        raise ValueError(
            f"Model '{name}': 'bias_monitoring.observations.batch_size' is not applicable "
            "with 'inference_request' — batching applies only to 'inference_dataset' mode."
        )

    if has_dataset:
        if has_obs_source:
            raise ValueError(
                f"Model '{name}': 'bias_monitoring.observations.path'/'files' cannot be "
                "combined with 'inference_dataset' — observations are derived from the dataset."
            )
        if not (
            model.get("inference_config_path")
            or model.get("config_path")
            or model.get("model_path")
        ):
            raise ValueError(
                f"Model '{name}': 'inference_dataset' requires a Triton 'config.pbtxt' via "
                "'inference_config_path' (or 'config_path'), or a 'model_path' ONNX model "
                "the framework can generate one from."
            )


def render_curl_command(infer_url: str, payload: dict[str, Any]) -> str:
    """Return a copy-pasteable curl command with the request payload inline."""
    body = json.dumps(payload, separators=(",", ":"))
    return (
        f"curl -sk -X POST {infer_url}"
        f" -H 'Content-Type: application/json'"
        f" -d '{body}'"
    )


def render_curl_command_file(infer_url: str, request_path: Path) -> str:
    """Return a copy-pasteable curl command that reads the payload from a file.

    Uses ``-d @<path>`` instead of embedding the JSON inline, producing a
    command that is readable and directly runnable at a terminal.
    """
    return (
        f"curl -sk -X POST \\\n"
        f"  {infer_url} \\\n"
        f"  -H 'Content-Type: application/json' \\\n"
        f"  -d @{request_path}"
    )
