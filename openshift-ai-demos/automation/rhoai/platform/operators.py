"""RHOAI operator lifecycle via OLM.

Manages Subscription, OperatorGroup, and CSV only.
Does not touch DSC, DSCI, or operand resources.
"""

import time

from kubernetes.dynamic.exceptions import NotFoundError

from rhoai.ocp import resources, wait
from rhoai.platform import manifests
from rhoai.utils.logger import get_logger
from rhoai.utils.yaml_io import load

log = get_logger(__name__)


def is_installed(name: str, namespace: str) -> bool:
    """Return True if the RHOAI operator CSV is in Succeeded phase."""
    return _csv_phase(resolve_csv_name(name, namespace), namespace) == "Succeeded"


def install(
    name: str,
    namespace: str,
    channel: str,
    repo_root: str,
    timeout: int,
    version: str = "",
    source: str = "redhat-operators",
    source_namespace: str = "openshift-marketplace",
) -> None:
    """Apply OperatorGroup + Subscription and wait for CSV to succeed.

    The Subscription manifest is loaded from disk and every caller-supplied
    field is overwritten in-memory before applying — the file on disk is never
    modified.

    Args:
        channel:          OLM channel, e.g. ``"stable"`` or ``"stable-3.5"``.
        version:          Full versioned CSV name to pin via startingCSV, e.g.
                          ``"rhods-operator.v3.5.0"``.  When non-empty the
                          Subscription is created with
                          ``installPlanApproval: Manual`` so OLM stops at
                          exactly this CSV and does not auto-upgrade.  When
                          empty, ``installPlanApproval`` stays ``Automatic``.
        source:           CatalogSource name, e.g. ``"redhat-operators"`` or
                          ``"cs-rhoai-fbc-fragment"`` for beta/EA builds.
        source_namespace: Namespace that hosts the CatalogSource, usually
                          ``"openshift-marketplace"``.
    """
    log.info(
        "Installing RHOAI operator (channel: %s, source: %s%s)",
        channel, source,
        f", version: {version}" if version else "",
    )

    # Load the Subscription from disk and overwrite every runtime-variable
    # field in-memory so the mutated dict (not the file) reaches the cluster.
    sub = load(manifests.get_subscription(repo_root))
    spec = sub.setdefault("spec", {})
    spec["channel"]             = channel
    spec["source"]              = source
    spec["sourceNamespace"]     = source_namespace
    spec["installPlanApproval"] = "Automatic"
    if version:
        spec["startingCSV"]         = version
        spec["installPlanApproval"] = "Manual"

    resources.apply_manifest(manifests.get_operator_group(repo_root), namespace)
    resources.apply_dict(sub, namespace)

    if version:
        _approve_install_plan(name, namespace, version)

    wait_until_ready(name, namespace, timeout)


def upgrade(name: str, namespace: str, target_channel: str, timeout: int) -> None:
    """Switch OLM channel and wait for the new CSV to succeed."""
    log.info("Upgrading RHOAI operator to channel: %s", target_channel)
    resources.patch("Subscription", name, {"spec": {"channel": target_channel}}, namespace)
    wait_until_ready(name, namespace, timeout)


def wait_until_ready(name: str, namespace: str, timeout: int) -> None:
    """Block until the CSV for 'name' reaches Succeeded. Raises TimeoutError."""
    log.info("Waiting for operator '%s' (timeout: %ss)", name, timeout)
    # Resolve lazily inside the loop: OLM creates the versioned CSV name only
    # after the Subscription is processed, so resolving once up-front would
    # always fall back to the bare package name and never match.
    def _is_ready() -> bool:
        resolved = resolve_csv_name(name, namespace)
        return _csv_phase(resolved, namespace) == "Succeeded"

    wait.wait_until(_is_ready, f"CSV {name} Succeeded", timeout)


def resolve_csv_name(package_name: str, namespace: str) -> str:
    """Return the versioned CSV name for a package, e.g. 'rhods-operator.3.5.0'.

    Looks up CSVs in the namespace and finds the one whose name starts with
    package_name. Falls back to package_name as-is if none is found, so that
    an already-versioned name passed directly still works.
    """
    csvs = resources.list_resources("ClusterServiceVersion", namespace)
    for csv in csvs:
        csv_name = csv.get("metadata", {}).get("name", "")
        if csv_name.startswith(package_name + ".") or csv_name == package_name:
            return csv_name
    return package_name


def verify(name: str, namespace: str) -> None:
    """Assert the operator CSV is Succeeded. Raises RuntimeError if not."""
    log.info("Verifying RHOAI operator")
    resolved = resolve_csv_name(name, namespace)
    phase = _csv_phase(resolved, namespace)
    if phase != "Succeeded":
        raise RuntimeError(
            f"Operator '{resolved}' is not ready (phase={phase!r}). "
            "Run 'rhoai platform deploy' to install it."
        )
    log.info("RHOAI operator is Succeeded")


def get_csv_info(name: str, namespace: str) -> dict[str, str]:
    """Return display info for the resolved CSV: name, version, phase."""
    resolved = resolve_csv_name(name, namespace)
    s = resources.status("ClusterServiceVersion", resolved, namespace)
    spec = resources.get("ClusterServiceVersion", resolved, namespace).get("spec", {})
    return {
        "name":    resolved,
        "version": spec.get("version", "unknown"),
        "phase":   s.get("phase", "Unknown"),
    }


def _approve_install_plan(sub_name: str, namespace: str, csv_name: str) -> None:
    """Approve the pending InstallPlan that references csv_name.

    When a Subscription uses ``installPlanApproval: Manual``, OLM creates an
    InstallPlan but leaves it in ``RequiresApproval`` phase.  This function
    finds that plan and patches ``spec.approved: true`` so OLM proceeds.

    It polls until the InstallPlan appears (OLM may take a few seconds to
    create it after the Subscription is applied) and raises RuntimeError if
    none is found within the timeout.
    """
    log.info("Approving InstallPlan for CSV '%s' in '%s'", csv_name, namespace)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        plans = resources.list_resources("InstallPlan", namespace)
        for plan in plans:
            clustersvs = plan.get("spec", {}).get("clusterServiceVersionNames", [])
            phase      = plan.get("status", {}).get("phase", "")
            approved   = plan.get("spec", {}).get("approved", False)
            if csv_name in clustersvs and phase == "RequiresApproval" and not approved:
                plan_name = plan["metadata"]["name"]
                log.info("Approving InstallPlan '%s'", plan_name)
                resources.patch(
                    "InstallPlan", plan_name,
                    {"spec": {"approved": True}},
                    namespace,
                )
                return
        log.debug("InstallPlan for '%s' not yet available — retrying", csv_name)
        time.sleep(5)
    raise RuntimeError(
        f"Timed out waiting for InstallPlan referencing '{csv_name}' "
        f"in namespace '{namespace}'."
    )


def _csv_phase(name: str, namespace: str) -> str:
    try:
        return resources.status("ClusterServiceVersion", name, namespace).get("phase", "Unknown")
    except NotFoundError:
        return "Unknown"
