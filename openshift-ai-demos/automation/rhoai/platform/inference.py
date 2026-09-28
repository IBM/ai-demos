"""KServe InferenceService and ServingRuntime capability."""

import json
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import urllib3

from rhoai.ocp import resources, wait
from rhoai.utils.logger import get_logger

log = get_logger(__name__)

# Triton HTTP REST API paths (KServe v2 inference protocol).
_TRITON_INFER_PATH = "/v2/models/{model_name}/infer"

# Smoke-test defaults — not user-facing config.
_INFER_TIMEOUT   = 30    # seconds for the inference HTTP call
_RETRY_DELAY     = 3     # seconds to wait before one connection-error retry
_POOL_CONNECTIONS = 1


def apply_serving_runtime_from_template(
    template_path: Path,
    platform_namespace: str,
    deploy_namespace: str,
    model_name: str,
    runtime_name: str,
) -> None:
    """Instantiate an OpenShift Template and apply the resulting ServingRuntime.

    Mirrors the original manual flow:
        oc process -n <platform_namespace> <template-name> -p MODEL_NAME=<m> -p RUNTIME_NAME=<r> \
            | oc apply -n <deploy_namespace> -f -

    The Template is first uploaded to <platform_namespace> (redhat-ods-applications)
    where RHOAI catalogs serving-runtime templates, then processed server-side,
    and the resulting ServingRuntime is applied into <deploy_namespace>.

    Args:
        model_name:   Passed as MODEL_NAME — the model directory Triton loads
                      (must match the InferenceService name).
        runtime_name: Passed as RUNTIME_NAME — the metadata.name of the created
                      ServingRuntime. Must be unique per deployment so that multiple
                      models can coexist without overwriting each other's runtime.
    """
    log.info("Deploying Triton ServingRuntime '%s'", runtime_name)
    log.debug("Processing ServingRuntime template %s", template_path.name)
    resources.process_template(
        template_path, platform_namespace, deploy_namespace,
        {"MODEL_NAME": model_name, "RUNTIME_NAME": runtime_name},
    )


def wait_until_ready(
    name: str,
    namespace: str,
    timeout: int = 60,
    on_tick: Callable[[float], Any] | None = None,
) -> None:
    """Block until the InferenceService is Ready. Raises TimeoutError."""
    log.info("Waiting for InferenceService '%s'", name)
    log.debug("Timeout: %ss", timeout)
    wait.wait_until_ready("InferenceService", name, namespace, timeout=timeout, on_tick=on_tick)


def wait_until_all_ready(
    namespace: str,
    timeout: int = 300,
    on_tick: Callable[[float], Any] | None = None,
) -> None:
    """Block until every InferenceService in *namespace* is Ready.

    Used after ``patch_inferenceservice_config`` triggers a predictor pod
    rollout (KServe recycles pods to inject the updated logger configuration
    and the payload-logger sidecar).  The rolled-out pods end up with three
    containers; this function ensures they are all running and Ready before
    observations are sent.

    Args:
        namespace: Namespace to check.
        timeout:   Maximum seconds to wait before raising ``TimeoutError``.
        on_tick:   Optional progress callback invoked after each poll with
                   elapsed seconds as the sole argument.

    Raises:
        TimeoutError: If any InferenceService is not Ready within ``timeout``.
    """
    log.info("Waiting for all InferenceServices in '%s' to become Ready", namespace)
    log.debug("Timeout: %ss", timeout)

    def _all_ready() -> bool:
        isvcs = resources.list_resources("InferenceService", namespace)
        if not isvcs:
            return False
        return all(
            resources.is_ready("InferenceService", isvc["metadata"]["name"], namespace)
            for isvc in isvcs
        )

    wait.wait_until(
        _all_ready,
        description=f"all InferenceServices ready in '{namespace}'",
        timeout=timeout,
        on_tick=on_tick,
    )


def get_inference_url(name: str, namespace: str) -> str:
    """Return the public inference URL from status.url or the OpenShift Route.

    Raises RuntimeError if the service has no URL yet.
    """
    url = resources.status("InferenceService", name, namespace).get("url")
    if url:
        return url
    if resources.exists("Route", name, namespace):
        route = resources.get("Route", name, namespace)
        host = route.get("spec", {}).get("host", "")
        if host:
            scheme = "https" if route.get("spec", {}).get("tls") else "http"
            return f"{scheme}://{host}"
    raise RuntimeError(f"InferenceService '{name}' has no URL — call wait_until_ready() first.")


def verify(namespace: str, name: str | None = None) -> None:
    """Assert InferenceService(s) are Ready. Raises RuntimeError if any are not.

    Args:
        namespace: Namespace to check.
        name:      If given, check only that specific InferenceService.
                   If None, check every InferenceService in the namespace.
    """
    if name is not None:
        log.info("Verifying InferenceService '%s' in '%s'", name, namespace)
        if not resources.is_ready("InferenceService", name, namespace):
            raise RuntimeError(
                f"InferenceService '{name}' is not ready in '{namespace}'."
            )
        log.info("InferenceService '%s' is Ready", name)
        return

    log.info("Verifying all InferenceServices in '%s'", namespace)
    services = resources.list_resources("InferenceService", namespace)
    not_ready = [
        svc.get("metadata", {}).get("name", "?")
        for svc in services
        if not resources.is_ready(
            "InferenceService", svc.get("metadata", {}).get("name", ""), namespace
        )
    ]
    if not_ready:
        raise RuntimeError(f"InferenceServices not ready in '{namespace}': {', '.join(not_ready)}")
    log.info("All InferenceServices in '%s' are Ready", namespace)


def send_observations(
    model_name: str,
    namespace: str,
    observation_files: list[Path],
    schema_source: Path | None = None,
) -> int:
    """Send historical observation batches to a deployed model for TrustyAI ingestion.

    Each file in observation_files must be a local KServe v2 inference request
    (JSON or CSV).

    * **CSV** — transformed into a KServe v2 payload via
      :func:`_csv_payload_from_path`, using tensor metadata resolved from
      ``schema_source``.
    * **JSON** — assumed to already be a valid KServe v2 request; loaded and
      posted as-is, with no transformation or reinterpretation.

    Expected KServe v2 envelope structure::

        {
          "inputs": [{
            "name":     "<tensor-name>",
            "shape":    [N, F],   # N rows, F features
            "datatype": "FP64",
            "data":     [[...], ...]
          }]
        }

    Files are validated locally before any network call is made.  Validation
    checks that each payload is a KServe v2 envelope (a JSON object with a
    non-empty ``inputs`` list) and that ``shape[0] == len(data)`` for the
    first tensor.

    Each file is posted as a single HTTP call to the model's KServe v2
    inference endpoint (``/v2/models/<model_name>/infer``).  The KServe
    payload-logger sidecar intercepts each call and forwards inputs + outputs
    to TrustyAI automatically — no direct TrustyAI API call is needed here.

    Args:
        model_name:        InferenceService name (also the Triton model name).
        namespace:         Namespace where the model is deployed.
        observation_files: Ordered list of local KServe v2 JSON or CSV files.
        schema_source:     KServe v2 JSON file from which ``inputs[0].name``
                           and ``inputs[0].datatype`` are read for CSV
                           conversion.  Pass the model's ``inference_request``
                           JSON.  ``None`` is safe when all files are JSON.

    Returns:
        Total number of observation rows sent across all files.

    Raises:
        FileNotFoundError: If any path does not exist.
        ValueError:        If any file fails structural validation, or if
                           ``schema_source`` is missing required fields.
        RuntimeError:      If any HTTP POST fails (non-200 response).
    """
    if not observation_files:
        raise ValueError("observation_files must not be empty")

    input_name, datatype = _read_tensor_schema(schema_source)

    # --- local validation pass (before touching the cluster) ---
    total_rows = 0
    for path in observation_files:
        total_rows += _validate_observation_file(path, input_name=input_name, datatype=datatype)

    # --- send each batch ---
    base_url  = get_inference_url(model_name, namespace)
    infer_url = base_url + _TRITON_INFER_PATH.format(model_name=model_name)

    log.info(
        "Sending %d observation rows across %d file(s) to '%s'",
        total_rows, len(observation_files), model_name,
    )
    for path in observation_files:
        payload = _load_request_payload(path, input_name=input_name, datatype=datatype)
        log.debug("POST %s  (%s)", infer_url, path.name)
        with _quiet_urllib3():
            _http_post(infer_url, payload)

    log.info("Observations sent: %d rows", total_rows)
    return total_rows


def _read_tensor_schema(schema_source: Path | None) -> tuple[str, str]:
    """Read ``inputs[0].name`` and ``inputs[0].datatype`` from a KServe v2 JSON file.

    Used to resolve tensor metadata when converting CSV files.  The caller
    passes the model's ``inference_request`` JSON as the canonical source;
    when ``csv_config`` is present in the model config the caller pre-resolves
    those values into a synthetic path or passes them directly — see
    ``_resolve_tensor_schema`` in ``deploy.py``.

    Args:
        schema_source: Path to a KServe v2 JSON file whose ``inputs[0]``
                       contains ``name`` and ``datatype``.  When ``None``,
                       defaults ``("input", "FP64")`` are returned — only safe
                       when all files are JSON.

    Returns:
        ``(input_name, datatype)`` tuple.

    Raises:
        FileNotFoundError: If ``schema_source`` is provided but does not exist.
        ValueError:        If the file is not valid JSON, or ``inputs[0]`` is
                           missing ``name`` or ``datatype``.
    """
    if schema_source is None:
        return "input", "FP64"
    if not schema_source.exists():
        raise FileNotFoundError(f"Schema source not found: {schema_source}")
    try:
        doc = json.loads(schema_source.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Schema source is not valid JSON: {schema_source}") from exc
    try:
        first = doc["inputs"][0]
        name     = first["name"]
        datatype = first["datatype"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"Schema source '{schema_source.name}': inputs[0] must have "
            "'name' and 'datatype' fields.  Expected a KServe v2 payload."
        ) from exc
    return name, datatype




def _load_request_payload(
    path: Path,
    input_name: str = "input",
    datatype: str = "FP64",
) -> dict:
    """Load a KServe v2 inference request from a JSON or CSV file.

    JSON files are loaded as-is.  CSV files are converted to a single-tensor
    KServe v2 payload via :func:`_csv_payload_from_path` using the supplied
    ``input_name`` and ``datatype``.

    Args:
        path:       Path to a ``.json`` or ``.csv`` file.
        input_name: Tensor input name.  Ignored for JSON files.
        datatype:   KServe v2 datatype string (e.g. ``"FP64"``).
                    Ignored for JSON files.

    Returns:
        KServe v2 payload dict.

    Raises:
        ValueError: If the file extension is not ``.json`` or ``.csv``.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _csv_payload_from_path(path, input_name=input_name, datatype=datatype)
    if suffix == ".json":
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Inference request file is not valid JSON: {path}") from exc
    raise ValueError(
        f"Unsupported inference request file type: '{path.name}'. "
        "Expected a .json or .csv file."
    )


def _csv_payload_from_path(
    path: Path,
    input_name: str = "input",
    datatype: str = "FP64",
) -> dict:
    """Convert a CSV file to a KServe v2 single-batch payload.

    Uses :func:`~rhoai.platform.request_generator.cast_value` and
    :func:`~rhoai.platform.request_generator.build_request` from
    :mod:`rhoai.platform.request_generator` for type casting and request
    assembly, so no request-building logic is duplicated.

    CSV rows are read with :class:`csv.reader` (not ``DictReader``) so that
    every row is treated as data.  A non-numeric first row is detected and
    skipped automatically — this preserves the header-detection contract of
    the original ``_csv_to_kserve_payload`` implementation.

    All rows form a single batch with shape ``[N, F]``.

    Args:
        path:       Path to the CSV file.
        input_name: KServe v2 tensor input name.
        datatype:   KServe v2 datatype string (e.g. ``"FP64"``).

    Returns:
        KServe v2 payload dict with shape ``[N, F]``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError:        If the CSV is empty, contains no data rows, or a
                           value in a data row cannot be cast to *datatype*.
    """
    import csv as _csv
    import io as _io

    from rhoai.platform.request_generator import (
        TensorSpec,
        build_request,
        cast_value,
    )

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    text = path.read_text()
    reader = _csv.reader(_io.StringIO(text))
    rows: list[list] = []
    for line_no, raw_row in enumerate(reader, start=1):
        if not raw_row:
            continue
        try:
            typed_row = [cast_value(v, datatype) for v in raw_row]
        except (ValueError, TypeError):
            if line_no == 1:
                # Non-castable first row → treat as a text header, skip it.
                log.debug("CSV '%s': skipping header row: %s", path.name, raw_row)
                continue
            raise ValueError(
                f"CSV '{path.name}' row {line_no}: "
                f"cannot convert to {datatype}: {raw_row!r}."
            )
        rows.append(typed_row)

    if not rows:
        raise ValueError(f"CSV '{path.name}' contains no data rows.")

    spec = TensorSpec(name=input_name, datatype=datatype, dims=[-1])
    return build_request([spec], {input_name: rows})


def _validate_observation_file(
    path: Path,
    input_name: str = "input",
    datatype: str = "FP64",
) -> int:
    """Validate a single observation file (JSON or CSV) and return its row count.

    Accepts:

    * **JSON** — must be a KServe v2 inference request payload (a JSON object
      with a non-empty ``inputs`` list). It is assumed to already be in
      KServe v2 format and is validated, never transformed. The row count
      comes from ``inputs[0].shape[0]``.  Only the first tensor is inspected;
      additional tensors are posted in full.
    * **CSV** — converted to a KServe v2 payload via ``_csv_payload_from_path``
      using ``input_name`` and ``datatype``.  The row count is the number of
      data rows after an optional header.

    Args:
        path:       Local path to a ``.json`` or ``.csv`` file.
        input_name: Tensor name passed to ``_csv_payload_from_path`` for CSV
                    files.  Ignored for JSON files.
        datatype:   KServe v2 datatype string passed to
                    ``_csv_payload_from_path`` for CSV files.  Ignored for
                    JSON files.

    Returns:
        Number of rows in the first (or only) input tensor.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError:        On any structural problem, with a descriptive message.
    """
    if not path.exists():
        raise FileNotFoundError(f"Observation file not found: {path}")

    payload = _load_request_payload(path, input_name=input_name, datatype=datatype)
    return _count_payload_rows(payload, path.name)


def _count_payload_rows(payload: object, label: str) -> int:
    """Validate a KServe v2 request payload and return its first-tensor row count.

    Shared by :func:`_validate_observation_file` (file-based observations) and
    :func:`send_observation_payloads` (in-memory batches generated from a
    dataset), so both apply identical structural checks.

    Args:
        payload: The candidate KServe v2 request (must be a dict with a
                 non-empty ``inputs`` list whose first tensor's ``shape[0]``
                 equals ``len(data)``).
        label:   Human-readable source label for error messages (a filename or
                 a synthetic identifier such as ``"batch[3]"``).

    Returns:
        ``inputs[0].shape[0]`` — the number of rows in the first tensor.

    Raises:
        ValueError: On any structural problem, with a descriptive message.
    """
    if not isinstance(payload, dict):
        raise ValueError(
            f"Observation '{label}' is not a valid KServe v2 request. "
            "Expected a JSON object "
            "({'inputs': [{'name': ..., 'shape': [N, ...], 'datatype': ..., 'data': [...]}]}), "
            f"got {type(payload).__name__}."
        )

    inputs = payload.get("inputs")
    if not inputs or not isinstance(inputs, list):
        raise ValueError(
            f"Observation '{label}' has no 'inputs' list. "
            "Expected a KServe v2 payload "
            "({'inputs': [{'name': ..., 'shape': [N, ...], 'datatype': ..., 'data': [...]}]})."
        )
    first = inputs[0]
    shape = first.get("shape")
    data  = first.get("data")
    if not shape or not isinstance(shape, list) or len(shape) < 1:
        raise ValueError(f"Observation '{label}': inputs[0] missing valid 'shape'.")
    if data is None or not isinstance(data, list):
        raise ValueError(f"Observation '{label}': inputs[0] missing 'data'.")
    declared_rows = shape[0]
    actual_rows   = len(data)
    if declared_rows != actual_rows:
        raise ValueError(
            f"Observation '{label}': shape[0]={declared_rows} "
            f"does not match len(data)={actual_rows}."
        )
    return declared_rows


def send_observation_payloads(
    model_name: str,
    namespace: str,
    payloads: list[dict],
) -> int:
    """Send pre-built KServe v2 request payloads to a model for TrustyAI ingestion.

    The in-memory counterpart to :func:`send_observations`: instead of reading
    observation files, it posts payloads generated from an ``inference_dataset``
    (see ``request_generator.iter_requests``).  Each payload is one HTTP POST to
    the model's ``/v2/models/<model_name>/infer`` endpoint; the KServe
    payload-logger sidecar forwards each call to TrustyAI automatically.

    Payloads are validated locally before any network call is made.

    Args:
        model_name: InferenceService name (also the Triton model name).
        namespace:  Namespace where the model is deployed.
        payloads:   Non-empty list of KServe v2 request dicts, each already
                    batched to the desired size.

    Returns:
        Total number of observation rows sent across all payloads.

    Raises:
        ValueError:   If ``payloads`` is empty or any payload fails validation.
        RuntimeError: If any HTTP POST fails (non-200 response).
    """
    if not payloads:
        raise ValueError("payloads must not be empty")

    # --- local validation pass (before touching the cluster) ---
    total_rows = sum(
        _count_payload_rows(p, f"batch[{i}]") for i, p in enumerate(payloads)
    )

    base_url  = get_inference_url(model_name, namespace)
    infer_url = base_url + _TRITON_INFER_PATH.format(model_name=model_name)

    log.info(
        "Sending %d observation rows across %d request(s) to '%s'",
        total_rows, len(payloads), model_name,
    )
    for i, payload in enumerate(payloads):
        log.debug("POST %s  (batch %d/%d)", infer_url, i + 1, len(payloads))
        with _quiet_urllib3():
            _http_post(infer_url, payload)

    log.info("Observations sent: %d rows", total_rows)
    return total_rows


def delete_inference_service(name: str, namespace: str) -> None:
    """Send the delete request for an InferenceService. Does not wait."""
    log.info("Deleting InferenceService '%s'", name)
    resources.delete_manifest("InferenceService", name, namespace)


def delete_serving_runtime(name: str, namespace: str) -> None:
    """Send the delete request for a ServingRuntime. Does not wait."""
    log.info("Deleting ServingRuntime '%s'", name)
    resources.delete_manifest("ServingRuntime", name, namespace)


def wait_until_inference_services_gone(names: list[str], namespace: str, timeout: int = 120) -> None:
    """Block until every named InferenceService no longer exists.

    All deletes should be issued before calling this so the removals run
    concurrently and the wait covers all of them in a single poll loop.

    Raises:
        TimeoutError: If any InferenceService is still present after timeout.
    """
    log.info("Waiting for InferenceService(s) to be removed: %s", names)
    wait.wait_until(
        lambda: all(not resources.exists("InferenceService", n, namespace) for n in names),
        f"InferenceServices gone in '{namespace}'",
        timeout=timeout,
    )


def wait_until_serving_runtimes_gone(names: list[str], namespace: str, timeout: int = 120) -> None:
    """Block until every named ServingRuntime no longer exists.

    All deletes should be issued before calling this so the removals run
    concurrently and the wait covers all of them in a single poll loop.

    Raises:
        TimeoutError: If any ServingRuntime is still present after timeout.
    """
    log.info("Waiting for ServingRuntime(s) to be removed: %s", names)
    wait.wait_until(
        lambda: all(not resources.exists("ServingRuntime", n, namespace) for n in names),
        f"ServingRuntimes gone in '{namespace}'",
        timeout=timeout,
    )


class EndpointUnreachable(RuntimeError):
    """Raised when the inference endpoint cannot be reached after retries.

    Carries the endpoint URL and the curl reproduction command so the caller
    can present a structured, user-oriented message without knowing internals.
    """

    def __init__(self, infer_url: str, curl_cmd: str) -> None:
        super().__init__(f"Inference endpoint unreachable: {infer_url}")
        self.infer_url = infer_url
        self.curl_cmd  = curl_cmd


@contextmanager
def _quiet_urllib3():
    """Temporarily raise the urllib3 log level to ERROR.

    urllib3 emits WARNING-level records for connection retries and name
    resolution failures.  These are implementation noise when we already
    handle the failure ourselves — suppress them for the duration of the
    with-block and restore the original level afterwards.
    """
    logger = logging.getLogger("urllib3")
    original = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(original)


def verify_triton_inference(
    isvc_name: str,
    namespace: str,
    model_name: str,
    sample_request: Path,
    schema_source: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Smoke-test a Triton InferenceService by submitting a sample request.

    Steps:
        1. Resolve the InferenceService URL.
        2. POST the sample payload to /v2/models/<model_name>/infer.
        3. Assert HTTP 200, model_name present, and outputs present in response.

    One retry after a short delay is attempted on connection errors to absorb
    the brief propagation gap between IS Ready and the Route becoming reachable.
    urllib3 connection-level warnings are suppressed during both attempts so that
    internal retry noise does not appear in user-facing output.

    When ``sample_request`` is a CSV file, tensor metadata is read from
    ``schema_source`` via ``_read_tensor_schema``.  Pass the model's
    ``inference_request`` JSON (overridden by ``csv_config`` when present).
    JSON request files are posted as-is; ``schema_source`` is ignored for them.

    Args:
        isvc_name:      Kubernetes name of the InferenceService.
        namespace:      Namespace where the InferenceService lives.
        model_name:     Model name as registered in Triton (used in URL and response check).
        sample_request: Path to a ``.json`` or ``.csv`` file containing the
                        KServe v2 inference payload.
        schema_source:  KServe v2 JSON file from which ``inputs[0].name`` and
                        ``inputs[0].datatype`` are read for CSV conversion.
                        ``None`` is safe for JSON request files.

    Returns:
        ``(payload, response, curl_cmd)`` for surfacing the exact input, output,
        and reproducible curl command in higher-level summaries.

    Raises:
        EndpointUnreachable: When the endpoint cannot be reached after one retry.
        RuntimeError:        If the inference request fails or the response is not valid.
    """
    base_url   = get_inference_url(isvc_name, namespace)
    infer_url  = base_url + _TRITON_INFER_PATH.format(model_name=model_name)
    input_name, datatype = _read_tensor_schema(schema_source)
    payload    = _load_request_payload(sample_request, input_name=input_name, datatype=datatype)
    curl_cmd  = (
        f"curl -sk -X POST {infer_url}"
        f" -H 'Content-Type: application/json'"
        f" -d '{json.dumps(payload, separators=(",", ":"))}'"
    )

    log.info("Running model smoke test")
    log.debug("Model:    %s", model_name)
    log.debug("Endpoint: %s", infer_url)
    log.debug("Request payload:\n%s", json.dumps(payload, indent=2))
    log.debug("Reproduce manually:\n  %s", curl_cmd)

    with _quiet_urllib3():
        try:
            response, elapsed, status = _http_post(infer_url, payload)
        except _ConnectionError as exc:
            log.debug("Connection attempt 1 failed: %s", exc)
            time.sleep(_RETRY_DELAY)
            try:
                response, elapsed, status = _http_post(infer_url, payload)
            except _ConnectionError as exc2:
                log.debug("Connection attempt 2 failed: %s", exc2)
                raise EndpointUnreachable(infer_url, curl_cmd) from exc2

    log.debug("Response status: %s  elapsed: %.2fs", status, elapsed)
    log.debug("Response body:\n%s", json.dumps(response, indent=2))

    _assert_triton_response(response, model_name)
    log.info("Model is serving inference requests")
    return payload, response, curl_cmd


# ---------------------------------------------------------------------------
# Private HTTP helpers
# ---------------------------------------------------------------------------

class _ConnectionError(RuntimeError):
    """Raised when the HTTP call cannot reach the server (not an HTTP error)."""


def _http_post(url: str, body: dict[str, Any]) -> tuple[dict[str, Any], float, int]:
    """POST a JSON body to *url* and return (decoded response, elapsed seconds, status code).

    Raises:
        _ConnectionError: On connection/timeout failure.
        RuntimeError:     On a non-200 HTTP response.
    """
    http     = urllib3.PoolManager(num_pools=_POOL_CONNECTIONS, cert_reqs="CERT_NONE")
    encoded  = json.dumps(body).encode()
    t_start  = time.monotonic()
    try:
        resp = http.request(
            "POST",
            url,
            body=encoded,
            headers={"Content-Type": "application/json"},
            timeout=_INFER_TIMEOUT,
        )
    except urllib3.exceptions.HTTPError as exc:
        raise _ConnectionError(f"Cannot reach {url}: {exc}") from exc
    elapsed = time.monotonic() - t_start

    if resp.status != 200:
        raw     = resp.data.decode(errors="replace")
        preview = raw[:200]
        log.debug("Error response body:\n%s", raw)
        raise RuntimeError(
            f"Triton inference request failed: HTTP {resp.status} — {preview}"
        )
    try:
        return json.loads(resp.data), elapsed, resp.status
    except json.JSONDecodeError as exc:
        log.debug("Raw response (non-JSON): %s", resp.data[:500].decode(errors="replace"))
        raise RuntimeError(f"Triton response is not valid JSON: {exc}") from exc


def _assert_triton_response(response: dict[str, Any], model_name: str) -> None:
    """Assert the minimal KServe v2 response structure is present.

    Raises:
        RuntimeError: If model_name or outputs are missing from the response.
    """
    if response.get("model_name") != model_name:
        raise RuntimeError(
            f"Unexpected model_name in Triton response: "
            f"expected '{model_name}', got '{response.get('model_name')}'"
        )
    if not response.get("outputs"):
        raise RuntimeError(
            f"Triton response for '{model_name}' contains no outputs."
        )
