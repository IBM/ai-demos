"""TrustyAI REST API client.

Stateless functions that call the TrustyAI service HTTP endpoints.
Every public function takes explicit ``route`` and ``token`` parameters —
no session objects, no module-level state.

Endpoint reference (validated against ta-svc-bias branch):
    GET  /info                               → all models dict
    POST /info/names                         → apply field name mapping
    POST /metrics/group/fairness/spd         → one-shot SPD computation
    POST /metrics/group/fairness/spd/request → scheduled SPD monitoring
    POST /metrics/identity/request           → scheduled identity metric

Data ingestion is handled by platform/inference.send_observations(), which
posts KServe v2 requests directly to the model endpoint.  The KServe
payload-logger sidecar forwards inputs + outputs to TrustyAI automatically.
"""


import json
import time
from typing import Any

import requests

from rhoai.utils.logger import get_logger

log = get_logger(__name__)

_TIMEOUT = 30  # seconds

# OpenShift clusters use self-signed CA chains not present in Python's default
# CA bundle.  Matches the reference script behaviour (curl -sk).
_VERIFY_SSL = False


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _raise_with_body(response: "requests.Response") -> None:
    """Raise RuntimeError with the HTTP status and TrustyAI response body included.

    Called whenever TrustyAI returns a non-2xx response.  Including the body
    means validation errors (e.g. unknown protected attribute) are readable
    without requiring the user to reproduce the request manually.
    """
    body = response.text.strip() or "<empty body>"
    raise RuntimeError(
        f"TrustyAI request failed: HTTP {response.status_code} — {body}"
    )


def _post(route: str, token: str, path: str, body: dict[str, Any]) -> Any:
    url = f"{route.rstrip('/')}{path}"
    log.debug("POST %s", url)
    response = requests.post(
        url, json=body, headers=_headers(token), timeout=_TIMEOUT, verify=_VERIFY_SSL
    )
    if not response.ok:
        _raise_with_body(response)
    # /info/names returns 204 No Content on the first call and may return a
    # non-JSON or unparseable body on subsequent idempotent calls — skip parsing
    # in both cases.
    if not response.content or not response.content.strip():
        return None
    content_type = response.headers.get("Content-Type", "")
    if "application/json" not in content_type:
        log.debug("POST %s — ignoring non-JSON response body (Content-Type: %s)", url, content_type)
        return None
    try:
        return response.json()
    except json.JSONDecodeError:
        log.debug("POST %s — body claims application/json but failed to parse; ignoring", url)
        return None


def _get(route: str, token: str, path: str, _retries: int = 3, _delay: float = 1.0) -> Any:
    """Issue a GET request and return the parsed JSON response.

    TrustyAI can transiently return an empty body, or a non-JSON body (e.g. a
    proxy error page), on ``GET /info`` immediately after a ``POST /info/names``
    completes (the service is persisting the mapping internally).  We retry up
    to ``_retries`` times with a short delay before giving up, so the caller
    never sees a spurious ``JSONDecodeError``.
    """
    url = f"{route.rstrip('/')}{path}"
    log.debug("GET %s", url)
    last_err: str = "empty body"
    for attempt in range(1, _retries + 1):
        response = requests.get(
            url, headers=_headers(token), timeout=_TIMEOUT, verify=_VERIFY_SSL
        )
        if not response.ok:
            _raise_with_body(response)
        body = response.content.strip() if response.content else b""
        if body:
            try:
                return response.json()
            except json.JSONDecodeError:
                last_err = f"non-JSON body: {body[:80]!r}"
                log.debug(
                    "GET %s returned non-JSON body (attempt %d/%d) — retrying in %.1fs",
                    url, attempt, _retries, _delay,
                )
        else:
            last_err = "empty body"
            log.debug(
                "GET %s returned empty body (attempt %d/%d) — retrying in %.1fs",
                url, attempt, _retries, _delay,
            )
        if attempt < _retries:
            time.sleep(_delay)
    raise RuntimeError(
        f"GET {url} returned an unparseable response after {_retries} attempts "
        f"({last_err}). TrustyAI may still be processing a previous request."
    )


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------

def get_model_info(route: str, token: str, model_id: str) -> dict[str, Any]:
    """Return the TrustyAI info entry for a single model.

    Calls GET /info (returns all models) and filters client-side.

    Args:
        route:    TrustyAI service base URL (e.g. ``https://trustyai-service.apps…``).
        token:    Bearer token for authentication.
        model_id: The model identifier to look up.

    Returns:
        The info dict for ``model_id``.

    Raises:
        KeyError: If ``model_id`` is not found in the response.
        requests.HTTPError: On a non-2xx response.
    """
    log.info("Fetching TrustyAI info for model '%s'", model_id)
    all_info: dict[str, Any] = _get(route, token, "/info")
    if model_id not in all_info:
        raise KeyError(
            f"Model '{model_id}' not found in TrustyAI info response. "
            f"Available: {list(all_info.keys())}"
        )
    return all_info[model_id]


def get_observation_count(route: str, token: str, model_id: str) -> int:
    """Return the number of observations TrustyAI has ingested for a model.

    Calls GET /info and reads ``info[model_id]["data"]["observations"]``.
    Returns 0 if the model is not yet known to TrustyAI — does not raise
    KeyError, so callers can poll safely before any data has arrived.

    Args:
        route:    TrustyAI service base URL.
        token:    Bearer token for authentication.
        model_id: The model identifier to look up.

    Returns:
        Number of observations ingested, or 0 if the model is not yet registered.

    Raises:
        requests.HTTPError: On a non-2xx response from /info.
    """
    log.debug("Fetching observation count for model '%s'", model_id)
    all_info: dict[str, Any] = _get(route, token, "/info")
    entry = all_info.get(model_id, {})
    return entry.get("data", {}).get("observations", 0)


def is_name_mapping_applied(
    route: str,
    token: str,
    model_id: str,
) -> bool:
    """Return True if the name mapping is already applied for this model.

    Confirmed TrustyAI /info response structure (live cluster):

        inputSchema = {
            "items": {
                "<display_name>": {          # dict KEY is the current display name
                    "name": "<internal_name>",  # "name" value is the original internal name
                    "type": "DOUBLE",
                    "columnIndex": N,
                },
                ...
            },
            "nameMapping": {                 # only present after a mapping POST
                "<internal_name>": "<display_name>",
                ...
            }
        }

    The ``nameMapping`` field is the definitive signal: TrustyAI only
    populates it after a successful ``POST /info/names``.  A non-empty
    ``nameMapping`` on either schema means the mapping has been applied.

    Returns False when the model is not yet known to TrustyAI so the caller
    will attempt the POST (the correct safe default).
    """
    all_info: dict[str, Any] = _get(route, token, "/info")
    data = all_info.get(model_id, {}).get("data", {})
    if not data:
        return False

    input_schema  = data.get("inputSchema",  {})
    output_schema = data.get("outputSchema", {})

    already_applied = bool(
        input_schema.get("nameMapping") or output_schema.get("nameMapping")
    )
    if already_applied:
        log.debug("Name mapping already applied for '%s' — skipping", model_id)
    return already_applied


def apply_name_mapping(
    route: str,
    token: str,
    model_id: str,
    input_mapping: dict[str, str],
    output_mapping: dict[str, str],
) -> None:
    """Apply a human-readable field name mapping to a model's data.

    Idempotent — skips the POST if the mapping has already been applied
    (detected by checking whether the original internal names are still
    present in TrustyAI's /info response).

    Args:
        route:          TrustyAI service base URL.
        token:          Bearer token for authentication.
        model_id:       The model identifier.
        input_mapping:  ``{internal_name: display_name}`` for input features.
        output_mapping: ``{internal_name: display_name}`` for output features.

    Raises:
        requests.HTTPError: On a non-2xx response.
    """
    if is_name_mapping_applied(route, token, model_id):
        log.info("Name mapping already applied for '%s' — skipping", model_id)
        return
    log.info("Applying name mapping for model '%s'", model_id)
    _post(route, token, "/info/names", {
        "modelId": model_id,
        "inputMapping": input_mapping,
        "outputMapping": output_mapping,
    })


# ---------------------------------------------------------------------------
# Metrics — SPD (Statistical Parity Difference)
# ---------------------------------------------------------------------------

def compute_spd(
    route: str,
    token: str,
    model_id: str,
    protected_attribute: str,
    privileged_value: float,
    unprivileged_value: float,
    outcome_name: str,
    favorable_outcome: float,
    batch_size: int = 5000,
) -> dict[str, Any]:
    """Compute a one-shot SPD fairness metric.

    Returns:
        Dict with keys ``value``, ``id``, ``thresholds``, ``specificDefinition``.

    Raises:
        requests.HTTPError: On a non-2xx response.
    """
    log.info("Computing SPD for model '%s' (attribute='%s')", model_id, protected_attribute)
    return _post(route, token, "/metrics/group/fairness/spd", {
        "modelId": model_id,
        "protectedAttribute": protected_attribute,
        "privilegedAttribute": {"type": "DOUBLE", "value": privileged_value},
        "unprivilegedAttribute": {"type": "DOUBLE", "value": unprivileged_value},
        "outcomeName": outcome_name,
        "favorableOutcome": {"type": "DOUBLE", "value": favorable_outcome},
        "batchSize": batch_size,
    })


def schedule_spd(
    route: str,
    token: str,
    model_id: str,
    protected_attribute: str,
    privileged_value: float,
    unprivileged_value: float,
    outcome_name: str,
    favorable_outcome: float,
    batch_size: int = 5000,
) -> str:
    """Schedule recurring SPD monitoring for a model.

    Returns:
        The request ID assigned by TrustyAI (use to cancel or inspect later).

    Raises:
        requests.HTTPError: On a non-2xx response.
    """
    log.info("Scheduling SPD for model '%s' (attribute='%s')", model_id, protected_attribute)
    response = _post(route, token, "/metrics/group/fairness/spd/request", {
        "modelId": model_id,
        "protectedAttribute": protected_attribute,
        "privilegedAttribute": {"type": "DOUBLE", "value": privileged_value},
        "unprivilegedAttribute": {"type": "DOUBLE", "value": unprivileged_value},
        "outcomeName": outcome_name,
        "favorableOutcome": {"type": "DOUBLE", "value": favorable_outcome},
        "batchSize": batch_size,
    })
    # TrustyAI returns "requestId" in newer versions; fall back to "id" for older ones.
    return response.get("requestId") or response.get("id", "")


# ---------------------------------------------------------------------------
# Metrics — Identity
# ---------------------------------------------------------------------------

def schedule_identity(
    route: str,
    token: str,
    model_id: str,
    column_name: str,
    batch_size: int = 5000,
) -> str:
    """Schedule recurring identity metric tracking for a model column.

    Args:
        route:       TrustyAI service base URL.
        token:       Bearer token for authentication.
        model_id:    The model identifier.
        column_name: The feature/output column to track.
        batch_size:  Number of recent observations to include per computation.

    Returns:
        The request ID assigned by TrustyAI.

    Raises:
        requests.HTTPError: On a non-2xx response.
    """
    log.info("Scheduling identity metric for model '%s' (column='%s')", model_id, column_name)
    response = _post(route, token, "/metrics/identity/request", {
        "modelId": model_id,
        "columnName": column_name,
        "batchSize": batch_size,
    })
    # TrustyAI returns "requestId" in newer versions; fall back to "id" for older ones.
    return response.get("requestId") or response.get("id", "")
