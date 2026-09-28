"""Fraud Detection — cleanup.

Deletes use-case resources in reverse deploy order.
Platform resources (DSC, DSCI, operator) are managed by the CLI, not here.
Pass --delete-platform to 'rhoai usecase cleanup' to also remove them.

Each model has its own dedicated ServingRuntime (named via assets.serving_runtime_name).
Both the InferenceService and its ServingRuntime are deleted per model.

Staged-model cleanup: when a model was deployed with model_path/config_path the
framework created a staging Pod and a PVC on its behalf.  Both are deleted here
using the same name-derivation logic as deploy.py:
  - Pod: pvc-stage-<pvc-name>   (from storage._STAGING_POD_PREFIX)
  - PVC: model.pvc_name or "<model-name>-pvc"

TrustyAI cleanup is conditional: only runs when at least one model has
bias_monitoring configured (mirrors the deploy-time gate).
Resources removed when TrustyAI was deployed:
  - TrustyAIService CR + Deployment
  - inferenceservice-config patch reverted
  - kserve-logger-ca-bundle ConfigMap
  - RoleBinding  (<sa_name>-view)
  - ServiceAccount (<sa_name>)

Cleanup is structured as three explicit wait barriers so that each layer of
pods is confirmed gone before the next layer is torn down:

  Phase 1 — issue all ISVC deletes, wait for all ISVC pods gone,
             issue all SR deletes, wait for all SR pods gone.
  Phase 1b — delete staging Pods then staged PVCs (Pod deleted first so the
             volume is released before the PVC delete is issued).
  Phase 2 — delete TrustyAI CR, wait for CR gone, wait for Deployment gone,
             then revert config and remove RBAC (no pods — instant).
  Phase 3 — (CLI) delete DSC, wait for component pods gone, delete DSCI.
"""

from rich.console import Console
from typing import Any

from rhoai.platform import inference, storage, trustyai
from rhoai.utils.progress import elapsed_timer, header_step, step
from rhoai.usecases.fraud_detection import assets
from rhoai.utils.logger import get_logger

log = get_logger(__name__)
_console = Console(stderr=False, highlight=False)


def _print_cleanup_summary(
    use_case: str,
    namespace: str,
    model_names: list[str],
    staged_pvc_names: list[str],
    trustyai_removed: bool,
    total: str = "",
) -> None:
    """Print the structured cleanup summary.

    Args:
        use_case:          Use-case name shown in the scope block.
        namespace:         Workload namespace shown in the scope block.
        model_names:       Names of models whose resources were removed.
        staged_pvc_names:  Names of staged-model PVCs (and their Pods) that were removed.
        trustyai_removed:  Whether TrustyAI resources were removed.
        total:             Formatted total duration string, e.g. "1m 14s".
    """
    duration_str = f"  Total: {total}" if total else ""
    _console.print(f"\nCleanup complete.{duration_str}\n")
    _console.print(f"  Use case   : {use_case}")
    _console.print(f"  Namespace  : {namespace}\n")

    _console.print("Removed\n")
    for name in model_names:
        _console.print(f"  \u2714  {name}")
    for pvc_name in staged_pvc_names:
        _console.print(f"  \u2714  {pvc_name} (Pod + PVC)")
    if trustyai_removed:
        _console.print(f"  \u2714  TrustyAI")
    _console.print("")


def cleanup(config: dict[str, Any]) -> None:
    """Remove Fraud Detection use-case resources from the cluster."""
    dep_cfg       = config.get("deployment", {})
    namespace     = dep_cfg.get("namespace") or config["platform"]["namespace"]
    models        = dep_cfg.get("models", [])
    trustyai_name = dep_cfg.get("trustyai_service_name", "trustyai-service")
    sa_name       = dep_cfg.get("trustyai_service_account", "trustyai-user")

    log.info("Cleaning up Fraud Detection in '%s'", namespace)

    platform_namespace = config["platform"]["namespace"]

    isvc_names    = [m["name"]                              for m in models]
    runtime_names = [assets.serving_runtime_name(m["name"]) for m in models]
    bias_models   = [m for m in models if m.get("bias_monitoring")]

    # Collect PVC names for models that were staged by the framework.
    # Uses the same derivation as deploy.py: model.pvc_name or "<model-name>-pvc".
    staged_pvc_names = [
        m.get("pvc_name") or f"{m['name']}-pvc"
        for m in models
        if m.get("model_path")
    ]

    with elapsed_timer() as timer:
        # --- Phase 1: model serving -------------------------------------------
        # Issue all ISVC deletes first so the pods start terminating in parallel,
        # then wait for all of them as a single barrier before touching ServingRuntimes.
        with header_step("Removing model serving", outcome="Model serving removed"):
            for name in isvc_names:
                with step(f"Removing inference service '{name}'"):
                    log.debug("Deleting InferenceService: %s", name)
                    inference.delete_inference_service(name, namespace)

            with step("Waiting for inference service pods to terminate") as s:
                inference.wait_until_inference_services_gone(isvc_names, namespace)

            for runtime_name in runtime_names:
                with step(f"Removing serving runtime '{runtime_name}'"):
                    log.debug("Deleting ServingRuntime: %s", runtime_name)
                    inference.delete_serving_runtime(runtime_name, namespace)

            with step("Waiting for serving runtime pods to terminate") as s:
                inference.wait_until_serving_runtimes_gone(runtime_names, namespace)

        # --- Phase 1b: staging Pods + staged PVCs ----------------------------
        # The staging Pod is deleted first so the volume is released before the
        # PVC delete is issued.  Normally the Pod is gone (deleted in the deploy
        # finally block), but this handles any interrupted-deploy leftovers.
        if staged_pvc_names:
            with header_step("Removing staged models", outcome="Staged models removed"):
                for pvc_name in staged_pvc_names:
                    with step(f"Removing staging Pod for '{pvc_name}'"):
                        log.debug("Staging Pod for PVC: %s", pvc_name)
                        storage.delete_staging_pod(pvc_name, namespace)
                    with step(f"Removing PVC '{pvc_name}'"):
                        log.debug("Deleting PVC: %s", pvc_name)
                        storage.delete_pvc(pvc_name, namespace)

        # --- Phase 2: TrustyAI ------------------------------------------------
        # CR + Deployment wait is inside delete_trustyai_service; config/RBAC after.
        if bias_models:
            with header_step("Removing TrustyAI", outcome="TrustyAI removed"):
                with step("Removing TrustyAI service"):
                    log.debug("TrustyAI service name: %s", trustyai_name)
                    trustyai.delete_trustyai_service(trustyai_name, namespace)
                with step("Restoring KServe logger configuration"):
                    trustyai.revert_inferenceservice_config(platform_namespace)
                with step("Removing logger CA bundle"):
                    trustyai.delete_logger_ca_bundle(namespace)
                with step("Removing RBAC"):
                    log.debug("RoleBinding: %s-view  ServiceAccount: %s", sa_name, sa_name)
                    trustyai.delete_role_binding(f"{sa_name}-view", namespace)
                    trustyai.delete_service_account(sa_name, namespace)

    _print_cleanup_summary(
        use_case=config.get("_use_case", "fraud-detection"),
        namespace=namespace,
        model_names=isvc_names,
        staged_pvc_names=staged_pvc_names,
        trustyai_removed=bool(bias_models),
        total=timer.formatted,
    )
