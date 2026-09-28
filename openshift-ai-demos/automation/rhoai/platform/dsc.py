"""DataScienceCluster (DSC) and DSCInitialization (DSCI) lifecycle."""

from pathlib import Path

from rhoai.ocp import resources, wait
from rhoai.utils.logger import get_logger

log = get_logger(__name__)


def apply_dsci(manifest_path: Path) -> None:
    """Apply the DSCInitialization manifest. Idempotent."""
    log.info("Applying DSCInitialization from %s", manifest_path.name)
    resources.apply_manifest(manifest_path)


def apply_dsc(manifest_path: Path) -> None:
    """Apply the DataScienceCluster manifest. Idempotent."""
    log.info("Applying DataScienceCluster from %s", manifest_path.name)
    resources.apply_manifest(manifest_path)


def wait_until_ready(name: str, timeout: int) -> None:
    """Block until the DataScienceCluster is fully reconciled and Ready.

    Checks both ``status.phase == "Ready"`` and that the operator has observed
    the current spec generation (``status.observedGeneration >= metadata.generation``).
    This prevents the wait from returning on a stale Ready status that precedes
    the operator picking up a recent spec change (e.g. a component patch).

    Raises TimeoutError if the condition is not met within timeout seconds.
    """
    log.info("Waiting for DSC '%s' (timeout: %ss)", name, timeout)
    wait.wait_until(lambda: _is_reconciled(name), f"DataScienceCluster/{name} Ready", timeout)


def wait_dsci_ready(name: str, timeout: int) -> None:
    """Block until the DSCInitialization reaches Ready phase. Raises TimeoutError."""
    log.info("Waiting for DSCI '%s' (timeout: %ss)", name, timeout)
    wait.wait_until_ready("DSCInitialization", name, timeout=timeout)


def set_component_states(name: str, states: dict[str, str]) -> None:
    """Merge-patch managementState for the given components only.

    Only the components present in *states* are touched — every other
    component already configured on the DSC (via a prior full manifest
    apply or a previous call to this function) keeps its current state.
    This is what makes 'enable one more component later' safe: it never
    resets components a customer or a prior use-case deploy already turned on.

    Args:
        name:   DataScienceCluster name.
        states: e.g. {"kserve": "Managed", "trustyai": "Managed"}.
    """
    log.info("Setting component state(s) on DSC '%s': %s", name, states)
    patch_body = {
        "spec": {
            "components": {
                component: {"managementState": state}
                for component, state in states.items()
            }
        }
    }
    resources.patch("DataScienceCluster", name, patch_body, strategy="merge")


def verify_dsci(name: str) -> None:
    """Assert the DSCInitialization is in Ready phase. Raises RuntimeError if not."""
    log.info("Verifying DSCInitialization '%s'", name)
    phase = resources.status("DSCInitialization", name).get("phase", "Unknown")
    if phase != "Ready":
        raise RuntimeError(
            f"DSCInitialization '{name}' is not ready (phase={phase!r}). "
            "Run 'rhoai platform setup' to configure it."
        )
    log.info("DSCInitialization '%s' is Ready", name)


def verify(name: str) -> None:
    """Assert the DSC is in Ready phase. Raises RuntimeError if not."""
    log.info("Verifying DataScienceCluster '%s'", name)
    if not _is_ready(name):
        phase = resources.status("DataScienceCluster", name).get("phase", "Unknown")
        raise RuntimeError(
            f"DataScienceCluster '{name}' is not ready (phase={phase!r}). "
            "Run 'rhoai platform setup' to configure it."
        )
    log.info("DataScienceCluster '%s' is Ready", name)


def delete_dsc(name: str) -> None:
    """Delete the DataScienceCluster and wait for removal."""
    log.info("Deleting DataScienceCluster '%s'", name)
    resources.delete_manifest("DataScienceCluster", name)
    wait.wait_until_deleted("DataScienceCluster", name)


# Label applied by the RHOAI operator to every component Deployment it manages.
_RHOAI_PART_OF_LABEL = "app.kubernetes.io/part-of=rhods-operator"


def wait_until_no_deployments(namespace: str, timeout: int = 300) -> None:
    """Block until all RHOAI-managed Deployments in *namespace* are gone.

    Polls only Deployments labelled ``app.kubernetes.io/part-of=rhods-operator``
    so that unrelated Deployments in the same namespace do not cause a false
    block.  Called after deleting the DSC to confirm the operator has fully torn
    down kserve, trustyai, dashboard, and other component pods before the
    DSCInitialization is deleted.

    Args:
        namespace: The platform namespace (default: redhat-ods-applications).
        timeout:   Maximum seconds to wait before raising TimeoutError.
    """
    log.info("Waiting for RHOAI component Deployments to terminate in '%s'", namespace)
    wait.wait_until(
        lambda: len(resources.list_resources(
            "Deployment", namespace, label_selector=_RHOAI_PART_OF_LABEL
        )) == 0,
        f"RHOAI component Deployments removed from '{namespace}'",
        timeout=timeout,
    )


def delete_dsci(name: str) -> None:
    """Delete the DSCInitialization and wait for removal."""
    log.info("Deleting DSCInitialization '%s'", name)
    resources.delete_manifest("DSCInitialization", name)
    wait.wait_until_deleted("DSCInitialization", name)


def get_component_states(name: str) -> dict[str, str]:
    """Return a dict of component -> managementState from the DSC spec.

    Example: {"kserve": "Managed", "dashboard": "Managed", "trustyai": "Managed", ...}
    """
    obj = resources.get("DataScienceCluster", name)
    components = obj.get("spec", {}).get("components", {})
    return {
        comp: cfg.get("managementState", "Unknown")
        for comp, cfg in components.items()
        if isinstance(cfg, dict)
    }


def is_dsc_ready(name: str) -> bool:
    """Return True if the DataScienceCluster exists and is in Ready phase."""
    from kubernetes.dynamic.exceptions import NotFoundError
    try:
        return resources.status("DataScienceCluster", name).get("phase") == "Ready"
    except NotFoundError:
        return False


def is_dsci_ready(name: str) -> bool:
    """Return True if the DSCInitialization exists and is in Ready phase."""
    from kubernetes.dynamic.exceptions import NotFoundError
    try:
        return resources.status("DSCInitialization", name).get("phase") == "Ready"
    except NotFoundError:
        return False


def _is_ready(name: str) -> bool:
    return is_dsc_ready(name)


def _is_reconciled(name: str) -> bool:
    """Return True when the DSC is Ready *and* the operator has observed the current generation.

    ``metadata.generation`` is incremented by the API server on every spec
    write.  ``status.observedGeneration`` is set by the RHOAI operator each
    time it completes a reconciliation loop.  Requiring
    ``observedGeneration >= generation`` ensures the wait does not exit on a
    stale Ready status that predates a recent patch.
    """
    obj        = resources.get("DataScienceCluster", name)
    phase      = obj.get("status", {}).get("phase")
    generation = obj.get("metadata", {}).get("generation", 0)
    observed   = obj.get("status", {}).get("observedGeneration", 0)
    return phase == "Ready" and observed >= generation
