"""Fraud Detection — verification.

Checks that all resources deployed by this use case are healthy.
Runs platform checks first, then iterates over each configured model.
"""

from typing import Any

from rhoai.platform import inference, trustyai, trustyai_client
from rhoai.platform import verify as platform_verify
from rhoai.platform.inference import EndpointUnreachable
from rhoai.usecases.fraud_detection.assets import (
    ModelResult,
    render_curl_command_file,
    validate_model_config,
)
from rhoai.usecases.fraud_detection.deploy import _resolve_request_artifacts, print_summary
from rhoai.utils.logger import get_logger
from rhoai.utils.progress import elapsed_timer, header_step, step

log = get_logger(__name__)


def _verify_bias_monitoring(
    model: dict[str, Any],
    route: str,
    token: str,
) -> None:
    """Check that TrustyAI has observations for a model.

    Reads ``model["bias_monitoring"]`` and returns immediately if absent or null.
    Confirms observation count > 0, indicating monitoring is active.

    Args:
        model: Model config dict (from deployment.models).
        route: TrustyAI service base URL.
        token: Bearer token for authentication.

    Raises:
        RuntimeError: If TrustyAI has no observations for the model.
    """
    if not model.get("bias_monitoring"):
        return

    model_id = model["name"]
    count    = trustyai_client.get_observation_count(route, token, model_id)
    if count == 0:
        raise RuntimeError(
            f"TrustyAI has no observations for model '{model_id}'. "
            "Re-run deployment to send observations."
        )
    log.info("TrustyAI observations for '%s': %d", model_id, count)


def verify(config: dict[str, Any]) -> None:
    """Verify the Fraud Detection deployment is healthy."""
    dep_cfg        = config.get("deployment", {})
    namespace      = dep_cfg.get("namespace") or config["platform"]["namespace"]
    repo_root      = config["repo_root"]
    models         = dep_cfg.get("models", [])

    trustyai_name  = dep_cfg.get("trustyai_service_name",    "trustyai-service")
    trustyai_sa    = dep_cfg.get("trustyai_service_account", "trustyai-user")

    # Fail fast on ambiguous inference-input configuration, matching deploy().
    for model in models:
        validate_model_config(model)

    log.info("Verifying Fraud Detection in '%s'", namespace)

    # Determine whether any model has bias monitoring configured.
    # TrustyAI checks are skipped entirely when no model uses it.
    bias_models = [m for m in models if m.get("bias_monitoring")]
    route = ""
    token = ""

    with elapsed_timer() as timer:
        with step("Checking RHOAI platform"):
            check_results = platform_verify.verify_platform(config)
            failed = [r for r in check_results if not r.passed]
            if failed:
                lines = "\n".join(f"  {r.name}: {r.message}" for r in failed)
                raise RuntimeError(f"Platform checks failed:\n{lines}")

        if bias_models:
            with step("Checking TrustyAI"):
                log.debug("TrustyAI service name: %s", trustyai_name)
                trustyai.verify(trustyai_name, namespace)

            route = trustyai.get_url(trustyai_name, namespace)
            token = trustyai.get_bearer_token(trustyai_sa, namespace)

        results: list[ModelResult] = []
        n_models = len(models)
        for idx, model in enumerate(models, start=1):
            name    = model["name"]
            counter = f" ({idx}/{n_models})" if n_models > 1 else ""
            with header_step(f"Verifying '{name}'{counter}", outcome=f"'{name}' healthy"):
                with step("Checking inference service"):
                    log.debug("InferenceService: %s", name)
                    inference.verify(namespace, name=name)

                # Populate source and endpoint from config + live cluster,
                # matching exactly what deploy records so the summary is consistent.
                model_uri = model.get("model_uri", "")
                endpoint  = inference.get_inference_url(name, namespace)
                result    = ModelResult(name=name, model_uri=model_uri, endpoint=endpoint)

                with step("Checking model inference") as s:
                    request_path, schema_source, payload, _ = _resolve_request_artifacts(model, repo_root)
                    result.request_path    = request_path
                    result.inference_input = payload
                    result.curl_cmd = render_curl_command_file(
                        endpoint.rstrip("/") + inference._TRITON_INFER_PATH.format(model_name=name),
                        request_path,
                    )
                    try:
                        payload, response, _ = inference.verify_triton_inference(
                            name, namespace, name,
                            request_path,
                            schema_source=schema_source,
                        )
                        result.inference_input  = payload
                        result.inference_output = response
                    except EndpointUnreachable as exc:
                        s.skip()
                        result.validation_skipped = True
                        result.unreachable        = exc
                        log.debug("Endpoint unreachable for '%s': %s", name, exc)

                if model.get("bias_monitoring"):
                    with step("Checking TrustyAI observations"):
                        _verify_bias_monitoring(model, route, token)

                results.append(result)

    print_summary(
        results,
        use_case=config.get("_use_case", "fraud-detection"),
        namespace=namespace,
        config_file=config.get("_config_file", ""),
        trustyai_route=route,
        mode="verify",
        total=timer.formatted,
    )
