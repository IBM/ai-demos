"""TrustyAI platform capabilities.

Organised into five sections:

    Prerequisites  — cluster and namespace setup before the TrustyAIService is deployed.
    Lifecycle      — TrustyAIService CR deploy, wait, verify, delete.
    Ingestion      — wait for observation data to be ingested after sending.
    Discovery      — resolve the TrustyAI route URL and obtain a bearer token.
    Cleanup        — delete prerequisite resources in reverse order.

Does not own model serving or observation sending — those live in platform/inference.py.
Does not contain REST API calls — those live in platform/trustyai_client.py.
"""

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rhoai.ocp import resources, wait
from rhoai.utils.logger import get_logger
from rhoai.utils.yaml_io import load_all

log = get_logger(__name__)

_SERVICE_KIND    = "TrustyAIService"
_ISVC_CONFIG_CM  = "inferenceservice-config"
_LOGGER_CA_CM    = "kserve-logger-ca-bundle"


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

def enable_user_workload_monitoring(manifest_path: Path) -> None:
    """Enable OpenShift User Workload Monitoring.

    Applies two cluster-scoped ConfigMaps from *manifest_path*:
      - cluster-monitoring-config        (openshift-monitoring)
      - user-workload-monitoring-config  (openshift-user-workload-monitoring)

    Idempotent. Must be run once per cluster before deploying TrustyAI.

    Args:
        manifest_path: Path to trustyai/service/monitoring-config.yaml.
    """
    log.info("Enabling OpenShift User Workload Monitoring")
    log.debug("Manifest: %s", manifest_path.name)
    for doc in load_all(manifest_path):
        ns = doc["metadata"]["namespace"]
        resources.apply_dict(doc, ns)
    log.info("User Workload Monitoring enabled")


def apply_rbac(manifest_path: Path, namespace: str, service_account: str = "trustyai-user") -> None:
    """Apply the TrustyAI RBAC manifest (ServiceAccount + RoleBinding). Idempotent.

    Accepts a multi-document YAML file and applies each document individually.
    The ``service_account`` name overrides the name embedded in the manifest so
    the caller can drive it from config rather than relying on the hardcoded value.
    """
    log.info("Applying TrustyAI RBAC resources (service_account=%s)", service_account)
    log.debug("Manifest: %s", manifest_path.name)
    for doc in load_all(manifest_path):
        kind = doc.get("kind", "")
        if kind == "ServiceAccount":
            doc.setdefault("metadata", {})["name"] = service_account
        elif kind == "RoleBinding":
            doc.setdefault("metadata", {})["name"] = f"{service_account}-view"
            for subject in doc.get("subjects", []):
                if subject.get("kind") == "ServiceAccount":
                    subject["name"] = service_account
        resources.apply_dict(doc, namespace)


def apply_logger_ca_bundle(manifest_path: Path, namespace: str) -> None:
    """Apply the kserve-logger-ca-bundle ConfigMap manifest. Idempotent.

    The annotation service.beta.openshift.io/inject-cabundle instructs
    the OpenShift service CA operator to populate the ConfigMap with the
    cluster CA bundle automatically after creation.

    Required before TrustyAI's payload logger can trust cluster-internal
    TLS endpoints.
    """
    log.info("Applying kserve-logger-ca-bundle ConfigMap in '%s'", namespace)
    log.debug("Manifest: %s", manifest_path.name)
    resources.apply_manifest(manifest_path, namespace)


def patch_inferenceservice_config(namespace: str) -> None:
    """Patch inferenceservice-config for TrustyAI payload logging.

    Applies two changes:
      1. Sets opendatahub.io/managed=false so RHOAI stops reverting the ConfigMap.
      2. Injects CA bundle settings into the logger JSON so the KServe payload
         logger can verify TLS against the cluster CA bundle injected by
         create_logger_ca_bundle().

    Idempotent — reads the current logger value before merging so repeated
    runs are safe.
    """
    log.info("Patching inferenceservice-config in '%s'", namespace)

    # 1 — stop RHOAI from managing (reverting) this ConfigMap.
    resources.patch(
        "ConfigMap",
        _ISVC_CONFIG_CM,
        {"metadata": {"annotations": {"opendatahub.io/managed": "false"}}},
        namespace=namespace,
    )

    # 2 — merge CA bundle settings into the logger JSON string.
    cm: dict[str, Any] = resources.get("ConfigMap", _ISVC_CONFIG_CM, namespace)
    logger_cfg: dict[str, Any] = json.loads(cm.get("data", {}).get("logger", "{}"))
    logger_cfg.update({
        "caBundle":     _LOGGER_CA_CM,
        "caCertFile":   "service-ca.crt",
        "tlsSkipVerify": False,
    })
    resources.patch(
        "ConfigMap",
        _ISVC_CONFIG_CM,
        {"data": {"logger": json.dumps(logger_cfg, indent=2)}},
        namespace=namespace,
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def apply_trustyai_service(manifest_path: Path, namespace: str) -> None:
    """Apply the TrustyAIService CR manifest. Idempotent."""
    log.info("Deploying TrustyAIService")
    log.debug("Manifest: %s", manifest_path.name)
    resources.apply_manifest(manifest_path, namespace)


def wait_until_ready(
    name: str,
    namespace: str,
    timeout: int = 300,
    on_tick: Callable[[float], Any] | None = None,
) -> None:
    """Block until the TrustyAI Deployment is available. Raises TimeoutError.

    TrustyAIService does not set a conditions[type=Ready] entry, so polling
    the CR itself never resolves.  The operator creates a same-named Deployment;
    waiting for that Deployment to become available is the reliable signal.
    """
    log.info("Waiting for TrustyAIService '%s'", name)
    log.debug("Timeout: %ss", timeout)
    wait.wait_until_ready("Deployment", name, namespace, timeout=timeout, on_tick=on_tick)


def verify(name: str, namespace: str) -> None:
    """Assert the TrustyAI Deployment is available. Raises RuntimeError if not."""
    log.info("Verifying TrustyAIService '%s'", name)
    if not resources.is_ready("Deployment", name, namespace):
        raise RuntimeError(
            f"TrustyAIService '{name}' is not ready in '{namespace}'."
        )
    log.info("TrustyAIService '%s' is Ready", name)


def delete_trustyai_service(name: str, namespace: str) -> None:
    """Delete the TrustyAIService CR and wait for full pod teardown.

    Waits in two stages:
      1. The TrustyAIService CR itself disappears (operator has processed the delete).
      2. The Deployment the operator created also disappears (pods have terminated).

    Stage 2 is required because the operator deletes the CR first and tears down
    the Deployment asynchronously.  Without it, --delete-platform can attempt to
    remove the DSC while TrustyAI pods are still running.
    """
    log.info("Deleting TrustyAIService '%s'", name)
    resources.delete_manifest(_SERVICE_KIND, name, namespace)
    wait.wait_until_deleted(_SERVICE_KIND, name, namespace)
    wait.wait_until_deleted("Deployment", name, namespace)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

#: Fraction of expected observations that must be present before monitoring
#: is considered ready.  Accounts for small numbers of requests that may be
#: dropped or delayed by the KServe payload-logger sidecar under load.
_INGESTION_THRESHOLD = 0.9


def wait_for_ingestion(
    route: str,
    token: str,
    model_id: str,
    expected: int,
    timeout: int = 300,
    on_tick: Callable[[float], Any] | None = None,
) -> None:
    """Block until TrustyAI has ingested >= 90% of the expected observations.

    The 90% threshold (``_INGESTION_THRESHOLD``) accounts for the small number
    of requests that may be dropped or delayed by the KServe payload-logger
    sidecar under load.

    Uses ``ocp.wait.wait_until`` so timeout and ``on_tick`` behaviour are
    consistent with all other wait functions in the framework.

    Args:
        route:    TrustyAI service base URL.
        token:    Bearer token for authentication.
        model_id: Model identifier to poll.
        expected: Total rows returned by ``inference.send_observations()``.
        timeout:  Maximum seconds to wait before raising ``TimeoutError``.
        on_tick:  Optional progress callback invoked after each poll with
                  elapsed seconds as the sole argument.

    Raises:
        TimeoutError: If the threshold is not reached within ``timeout`` seconds.
    """
    from rhoai.platform import trustyai_client  # local import — avoids circular dep

    threshold = max(1, int(expected * _INGESTION_THRESHOLD))
    log.info(
        "Waiting for TrustyAI to ingest %d/%d observations for '%s'",
        threshold, expected, model_id,
    )

    def _enough() -> bool:
        count = trustyai_client.get_observation_count(route, token, model_id)
        log.debug("TrustyAI observations for '%s': %d / %d", model_id, count, threshold)
        return count >= threshold

    wait.wait_until(
        _enough,
        description=f"TrustyAI ingestion for '{model_id}' ({threshold} observations)",
        timeout=timeout,
        on_tick=on_tick,
    )


# ---------------------------------------------------------------------------
# Name mapping
# ---------------------------------------------------------------------------

def resolve_and_apply_name_mapping(
    route: str,
    token: str,
    model_id: str,
    *,
    explicit_inputs: dict[str, str],
    explicit_outputs: dict[str, str],
    csv_headers: list[str] | None,
    num_input_tensors: int,
    pbtxt_output_names: list[str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve and apply the input/output name mappings to TrustyAI.

    Precedence for the *input* mapping:

        1. ``explicit_inputs`` — user-supplied ``name_mapping.inputs``; used
           verbatim, no derivation.
        2. Auto-derived from ``csv_headers`` + TrustyAI's ``inputSchema`` (only
           when observations came from a CSV dataset, so headers are available).
        3. Nothing — leave TrustyAI's original feature names in place.

    Precedence for the *output* mapping (identity — the goal is to register the
    model's own output names, not invent labels):

        1. ``explicit_outputs`` — used verbatim.
        2. ``config.pbtxt`` output tensor names → ``{name: name}``.
        3. TrustyAI's ``outputSchema`` names (i.e. the names seen in the first
           inference response) → ``{name: name}``.
        4. Nothing — leave outputs unmapped.

    The ``POST /info/names`` is issued (idempotently) only when there is a
    non-empty input or output mapping to apply.  ``GET /info`` is fetched at most
    once and only when a derivation step actually needs it.

    Returns ``(input_mapping, output_mapping)`` as resolved (either may be empty).
    """
    from rhoai.platform import request_generator, trustyai_client  # avoid cycle

    info: dict | None = None

    def _model_info() -> dict:
        nonlocal info
        if info is None:
            info = trustyai_client.get_model_info(route, token, model_id)
        return info

    # --- inputs ---
    input_mapping = dict(explicit_inputs)
    if not input_mapping and csv_headers:
        items = _model_info().get("data", {}).get("inputSchema", {}).get("items", {})
        input_mapping = request_generator.derive_input_name_mapping(
            csv_headers, items, num_input_tensors
        )
        if input_mapping:
            log.info("Auto-derived input name mapping for '%s' (%d features)",
                     model_id, len(input_mapping))
        else:
            log.info("No safe input name mapping could be derived for '%s' — "
                     "leaving original feature names", model_id)

    # --- outputs (identity from pbtxt names, else from the response schema) ---
    output_mapping = dict(explicit_outputs)
    if not output_mapping:
        names = list(pbtxt_output_names or [])
        if not names:
            out_items = _model_info().get("data", {}).get("outputSchema", {}).get("items", {})
            names = [
                str(meta["name"]) for meta in out_items.values() if meta.get("name") is not None
            ]
        output_mapping = {name: name for name in names}
        if output_mapping:
            log.info("Defaulting output name mapping for '%s' to model output names: %s",
                     model_id, ", ".join(output_mapping))

    if input_mapping or output_mapping:
        trustyai_client.apply_name_mapping(
            route, token, model_id, input_mapping, output_mapping
        )
    return input_mapping, output_mapping


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def get_url(name: str, namespace: str) -> str:
    """Return the public URL for the TrustyAI service.

    Looks up status.url on the TrustyAIService first, then falls back to
    the OpenShift Route created by the opendatahub.io/enable-route annotation.

    Raises:
        RuntimeError: If no URL can be resolved — call wait_until_ready() first.
    """
    url = resources.status(_SERVICE_KIND, name, namespace).get("url")
    if url:
        return url
    if resources.exists("Route", name, namespace):
        route = resources.get("Route", name, namespace)
        host  = route.get("spec", {}).get("host", "")
        if host:
            scheme = "https" if route.get("spec", {}).get("tls") else "http"
            return f"{scheme}://{host}"
    raise RuntimeError(
        f"TrustyAIService '{name}' has no URL — call wait_until_ready() first."
    )


def get_bearer_token(service_account: str, namespace: str) -> str:
    """Create and return a short-lived bearer token for the given ServiceAccount.

    Uses `oc create token` via the TokenRequest API.  The token is valid for
    the cluster default duration (typically 1 hour) and is not cached —
    call this function once per session and pass the result to trustyai_client.

    Args:
        service_account: Name of the ServiceAccount (e.g. "trustyai-user").
        namespace:        Namespace where the ServiceAccount lives.

    Returns:
        Bearer token string.

    Raises:
        RuntimeError: If the TokenRequest fails.
    """
    log.info("Obtaining bearer token for ServiceAccount '%s'", service_account)
    result = subprocess.run(
        [
            "oc", "create", "token", service_account,
            "-n", namespace,
            "--output", "json",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to create token for ServiceAccount '{service_account}': "
            f"{result.stderr.strip()}"
        )
    try:
        return json.loads(result.stdout)["status"]["token"]
    except (json.JSONDecodeError, KeyError):
        # Older oc versions return the token as plain text
        token = result.stdout.strip()
        if not token:
            raise RuntimeError(
                f"Empty token returned for ServiceAccount '{service_account}'."
            ) from None
        return token


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def revert_inferenceservice_config(namespace: str) -> None:
    """Revert the inferenceservice-config patch applied by patch_inferenceservice_config.

    Reversal order (opposite of apply):
      1. Remove CA bundle keys from the logger JSON string.
      2. Restore opendatahub.io/managed to "true" so RHOAI resumes managing
         the ConfigMap.

    Idempotent — safe to call even if the ConfigMap was never patched.
    """
    from kubernetes.dynamic.exceptions import NotFoundError
    log.info("Reverting inferenceservice-config patch in '%s'", namespace)
    try:
        # 1 — remove CA bundle keys from the logger JSON.
        cm: dict[str, Any] = resources.get("ConfigMap", _ISVC_CONFIG_CM, namespace)
        logger_cfg: dict[str, Any] = json.loads(cm.get("data", {}).get("logger", "{}"))
        for key in ("caBundle", "caCertFile", "tlsSkipVerify"):
            logger_cfg.pop(key, None)
        resources.patch(
            "ConfigMap",
            _ISVC_CONFIG_CM,
            {"data": {"logger": json.dumps(logger_cfg, indent=2)}},
            namespace=namespace,
        )
        # 2 — restore RHOAI management so it can reconcile the ConfigMap again.
        resources.patch(
            "ConfigMap",
            _ISVC_CONFIG_CM,
            {"metadata": {"annotations": {"opendatahub.io/managed": "true"}}},
            namespace=namespace,
        )
    except NotFoundError:
        log.debug("inferenceservice-config not found in '%s' — skipping revert", namespace)


def delete_logger_ca_bundle(namespace: str) -> None:
    """Delete the kserve-logger-ca-bundle ConfigMap."""
    log.info("Deleting kserve-logger-ca-bundle ConfigMap in '%s'", namespace)
    resources.delete_manifest("ConfigMap", _LOGGER_CA_CM, namespace)


def delete_service_account(name: str, namespace: str) -> None:
    """Delete the TrustyAI ServiceAccount."""
    log.info("Deleting ServiceAccount '%s'", name)
    resources.delete_manifest("ServiceAccount", name, namespace)


def delete_role_binding(name: str, namespace: str) -> None:
    """Delete the TrustyAI RoleBinding."""
    log.info("Deleting RoleBinding '%s'", name)
    resources.delete_manifest("RoleBinding", name, namespace)
