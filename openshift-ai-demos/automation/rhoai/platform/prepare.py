"""Platform preparation — cluster validation, namespace setup, and bootstrap."""

from typing import Any

import typer

from rhoai.ocp import resources
from rhoai.utils.logger import get_logger

log = get_logger(__name__)

# Exhaustive list of components the DataScienceCluster CR accepts.
# Used to give an immediate, actionable error for typos.
VALID_COMPONENTS = frozenset({
    "aipipelines",
    "dashboard",
    "feastoperator",
    "kserve",
    "kueue",
    "llamastackoperator",
    "mlflowoperator",
    "modelregistry",
    "ogx",
    "ray",
    "sparkoperator",
    "trainer",
    "trainingoperator",
    "trustyai",
    "workbenches",
})


def prepare_platform(config: dict[str, Any]) -> None:
    """Validate login, storage, and namespace — in that order."""
    log.info("Starting platform preparation")
    validate_login()
    validate_storage(config.get("storage", {}).get("class_name", ""))
    validate_namespace(config["operator"]["namespace"])
    validate_namespace(config["platform"]["namespace"])
    log.info("Platform preparation complete")


def init_platform(config: dict[str, Any]) -> None:
    """Validate prerequisites, install the RHOAI operator, and initialize DSCI.

    This is the one-time, cluster-wide bootstrap step — it does not touch the
    DataScienceCluster or any component state:
        1. prepare_platform  — login, RBAC, storage, namespace
        2. operator          — install or wait for the existing CSV
        3. DSCI              — apply the manifest and wait for Ready

    Backs 'rhoai platform init'. Use install_component() afterwards to turn on
    specific components, or bootstrap_platform() to do both in one call.
    """
    from rhoai.platform import dsc, manifests, operators

    repo_root  = config["repo_root"]
    op_name    = config["operator"]["name"]
    op_ns      = config["operator"]["namespace"]
    op_timeout = config["timeouts"]["operator_ready"]

    prepare_platform(config)

    channel          = config["operator"]["channel"]
    csv_version      = config["operator"].get("version", "")
    source           = config["operator"].get("source", "redhat-operators")
    source_namespace = config["operator"].get("source_namespace", "openshift-marketplace")

    if not operators.is_installed(op_name, op_ns):
        # Pass all Subscription fields atomically so whatever is in config
        # (channel, source, version) all land in the same server-side apply.
        operators.install(
            op_name, op_ns, channel, repo_root, op_timeout,
            version=csv_version,
            source=source,
            source_namespace=source_namespace,
        )
    else:
        operators.wait_until_ready(op_name, op_ns, op_timeout)

    # Always apply DSCI: idempotent, only initialisation settings.
    dsc.apply_dsci(manifests.get_dsci(repo_root))
    dsc.wait_dsci_ready(config["dsc"]["dsci_name"], config["timeouts"]["dsc_ready"])


def install_component(config: dict[str, Any], components: list[str]) -> None:
    """Turn on one or more DSC components. Requires init_platform to have run first.

    Idempotent and additive — only the named components are patched to Managed;
    every other component's current state is left untouched.
    """
    from rhoai.platform import dsc, manifests, operators

    op_name  = config["operator"]["name"]
    op_ns    = config["operator"]["namespace"]
    dsc_name = config["dsc"]["name"]

    invalid = sorted(c for c in components if c not in VALID_COMPONENTS)
    if invalid:
        valid_list = ", ".join(sorted(VALID_COMPONENTS))
        raise typer.BadParameter(
            f"unknown component(s): {', '.join(invalid)}.\n"
            f"Valid components:\n  {valid_list}"
        )

    if not operators.is_installed(op_name, op_ns):
        raise RuntimeError(
            "Operator not found. Run 'rhoai platform init' first."
        )
    try:
        dsc.verify_dsci(config["dsc"]["dsci_name"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc} Run 'rhoai platform init' first."
        ) from exc

    if not resources.exists("DataScienceCluster", dsc_name):
        log.info("No existing DSC '%s' — creating from base manifest", dsc_name)
        dsc.apply_dsc(manifests.get_dsc(config["repo_root"]))

    dsc.set_component_states(dsc_name, {c: "Managed" for c in components})
    dsc.wait_until_ready(dsc_name, config["timeouts"]["dsc_ready"])


def remove_component(config: dict[str, Any], components: list[str]) -> None:
    """Set one or more DSC components to Removed. Requires init_platform to have run first.

    Idempotent — components already Removed stay Removed; unrelated components
    are left exactly as they are.
    """
    from rhoai.platform import dsc, operators

    op_name  = config["operator"]["name"]
    op_ns    = config["operator"]["namespace"]
    dsc_name = config["dsc"]["name"]

    invalid = sorted(c for c in components if c not in VALID_COMPONENTS)
    if invalid:
        valid_list = ", ".join(sorted(VALID_COMPONENTS))
        raise typer.BadParameter(
            "unknown component(s): " + ", ".join(invalid) + ".\n"
            "Valid components:\n  " + valid_list
        )

    if not operators.is_installed(op_name, op_ns):
        raise RuntimeError(
            "Operator not found. Run 'rhoai platform init' first."
        )

    if not resources.exists("DataScienceCluster", dsc_name):
        raise RuntimeError(
            f"DataScienceCluster '{dsc_name}' not found — nothing to disable. "
            "Run 'rhoai platform enable' to create it first."
        )

    dsc.set_component_states(dsc_name, {c: "Removed" for c in components})
    dsc.wait_until_ready(dsc_name, config["timeouts"]["dsc_ready"])

def platform_needs_reconciliation(config: dict[str, Any]) -> bool:
    """Return True if bootstrap_platform() must modify the cluster to reach the desired state.

    Performs read-only checks only — never writes or waits.  Called by
    bootstrap_platform() to skip the full reconciliation cycle when the
    platform is already healthy and the requested component states already match.

    Returns False (fast-path) when ALL of the following hold:
        1. Operator CSV is Succeeded.
        2. DSCInitialization is Ready.
        3. DataScienceCluster is Ready.
        4. Every component in config["components"] is already Managed
           (or no specific components are requested).
    """
    from rhoai.platform import dsc, operators

    op_name   = config["operator"]["name"]
    op_ns     = config["operator"]["namespace"]
    dsc_name  = config["dsc"]["name"]
    dsci_name = config["dsc"]["dsci_name"]

    if not operators.is_installed(op_name, op_ns):
        log.debug("platform_needs_reconciliation: operator not installed")
        return True

    if not dsc.is_dsci_ready(dsci_name):
        log.debug("platform_needs_reconciliation: DSCI not ready")
        return True

    if not dsc.is_dsc_ready(dsc_name):
        log.debug("platform_needs_reconciliation: DSC not ready")
        return True

    components = config.get("components") or []
    if components:
        try:
            states = dsc.get_component_states(dsc_name)
        except Exception:  # noqa: BLE001
            log.debug("platform_needs_reconciliation: could not read component states")
            return True
        not_managed = [c for c in components if states.get(c) != "Managed"]
        if not_managed:
            log.debug("platform_needs_reconciliation: not yet Managed: %s", not_managed)
            return True

    log.debug("platform_needs_reconciliation: platform already in desired state")
    return False


def bootstrap_platform(
    config: dict[str, Any],
    _needs_reconciliation: bool | None = None,
) -> None:
    """Full platform bootstrap: init_platform, then enable DSC components.

    On an already-configured cluster this is a fast verification pass —
    reconciliation is skipped when the platform is already in the desired state.

    Backs 'rhoai platform setup'. Two modes controlled by config["components"]:
    - Non-empty list → patch only those components to Managed.
    - Empty (default) → apply the full base DSC manifest as-is.

    Args:
        _needs_reconciliation: Pre-computed result of platform_needs_reconciliation().
            Pass this when the caller has already called platform_needs_reconciliation()
            to avoid a redundant second check. If None (default), the check runs here.
    """
    from rhoai.platform import dsc, manifests

    needs_recon = (
        _needs_reconciliation
        if _needs_reconciliation is not None
        else platform_needs_reconciliation(config)
    )
    if not needs_recon:
        log.info("Platform already in desired state — skipping reconciliation")
        return

    init_platform(config)

    components = config.get("components") or []
    if components:
        install_component(config, components)
    else:
        dsc.apply_dsc(manifests.get_dsc(config["repo_root"]))
        dsc.wait_until_ready(config["dsc"]["name"], config["timeouts"]["dsc_ready"])


# Backward-compatible alias — prefer bootstrap_platform() in new code.
deploy_platform = bootstrap_platform


def uninstall_platform(config: dict[str, Any], keep_workload_ns: bool = False) -> None:
    """Remove all RHOAI platform resources installed by init/setup.

    Deletion order (reverse of install):
        1. DataScienceCluster   — trigger operator component teardown, then wait
                                  for workload pods to drain before continuing
        2. DSCInitialization
        3. CSV, Subscription, OperatorGroup, InstallPlans (cluster-wide sweep)
        4. Dependency operators auto-detected on the cluster:
             servicemeshoperator3 (NIM / gateway routing)
             servicemeshoperator  (legacy Maistra v2)
           Each is only cleaned up when its Subscription is present —
           we never touch operators we did not install.
        5. CRDs registered by the RHOAI operator label + explicit RHOAI CRD list
        6. Stale APIServices (block namespace deletion with DiscoveryFailure)
        7. Webhooks (validating + mutating)
        8. Cluster-scoped RBAC (ClusterRoles, ClusterRoleBindings)
        9. Operator namespace   (redhat-ods-operator)
       10. Workload namespaces  — discovered dynamically via operator labels
                                  plus a static fallback list.
                                  Deleted by default; use --keep-workload-ns to
                                  preserve user notebooks and pipelines.

    Every step is safe on a partially-installed cluster — missing resources are
    skipped with an info log rather than raising an error.
    """
    import subprocess
    import time

    from rhoai.ocp import wait
    from rhoai.platform import dsc, operators

    op_name    = config["operator"]["name"]
    op_ns      = config["operator"]["namespace"]
    cluster_ns = config["platform"]["namespace"]
    dsc_name   = config["dsc"]["name"]
    dsci_name  = config["dsc"]["dsci_name"]

    # Patterns used for webhook / RBAC name matching
    _RHOAI_PATTERNS = (
        "rhods", "rhoai", "opendatahub", "odh",
        "kserve", "trustyai", "notebook", "dashboard",
        "model-registry", "modelregistry", "data-science",
        "istio", "maistra", "knative",
    )

    def _oc(*args: str) -> None:
        subprocess.run(["oc", *args], capture_output=True)

    def _oc_out(*args: str) -> str:
        r = subprocess.run(["oc", *args], capture_output=True, text=True)
        return r.stdout

    def _oc_lines(*args: str) -> list[str]:
        return [l for l in _oc_out(*args).splitlines() if l.strip()]

    def _matches(name: str) -> bool:
        return any(p in name for p in _RHOAI_PATTERNS)

    def _subscription_exists(pattern: str) -> bool:
        """Return True if any Subscription name matches pattern cluster-wide."""
        return any(
            pattern in line
            for line in _oc_lines("get", "subscriptions", "-A", "--no-headers")
        )

    def _force_delete_crd(crd: str) -> None:
        """Delete a CRD, stripping its finalizers first if needed."""
        _oc("delete", "crd", crd, "--ignore-not-found", "--timeout=30s")
        _oc("patch", "crd", crd, "--type=merge",
            "-p", '{"metadata":{"finalizers":null}}')

    def _delete_resources(kind: str) -> None:
        """Strip finalizers then delete all instances of a CRD kind cluster-wide."""
        # Strip finalizers
        for line in _oc_lines(
            "get", kind, "--all-namespaces", "--ignore-not-found", "--no-headers",
            "-o", "custom-columns=KIND:.kind,NAME:.metadata.name,NS:.metadata.namespace",
        ):
            parts = line.split()
            if len(parts) >= 3:
                res_kind, name, ns = parts[0], parts[1], parts[2]
                _oc("patch", res_kind, name, "-n", ns,
                    "--type=merge", "-p", '{"metadata":{"finalizers":null}}')
        _oc("delete", kind, "--all", "-A", "--ignore-not-found", "--timeout=30s")

    def _delete_olm_for(ns: str, sub_pattern: str, csv_pattern: str) -> None:
        """Delete Subscription + InstallPlan + CSV matching patterns in a namespace."""
        for sub in _oc_lines(
            "get", "subscription", "-n", ns, "--no-headers", "--ignore-not-found",
            "-o", "custom-columns=NAME:.metadata.name",
        ):
            if sub_pattern in sub:
                _oc("delete", "subscription", sub, "-n", ns, "--ignore-not-found")
        for ip in _oc_lines(
            "get", "installplan", "-n", ns, "--no-headers", "--ignore-not-found",
            "-o", "custom-columns=NAME:.metadata.name",
        ):
            _oc("delete", "installplan", ip, "-n", ns, "--ignore-not-found")
        for csv in _oc_lines(
            "get", "csv", "-n", ns, "--no-headers", "--ignore-not-found",
            "-o", "custom-columns=NAME:.metadata.name",
        ):
            if csv_pattern in csv:
                _oc("delete", "csv", csv, "-n", ns, "--ignore-not-found")

    def _cleanup_stale_apiservices() -> None:
        """Remove APIServices whose Available condition is not True.

        Stale APIServices cause NamespaceDeletionDiscoveryFailure and keep
        namespaces stuck in Terminating indefinitely.
        """
        log.info("Checking for stale APIServices")
        for api in _oc_lines(
            "get", "apiservice",
            "-o", "jsonpath={range .items[?(@.status.conditions[0].status!='True')]}{.metadata.name}{'\\n'}{end}",
        ):
            log.info("  Deleting stale APIService %s", api)
            _oc("delete", "apiservice", api, "--ignore-not-found", "--timeout=15s")

    def _cleanup_servicemesh3() -> None:
        """Remove Service Mesh 3 (Sail / Istio) operator and all its resources."""
        log.info("Cleaning up ServiceMesh 3 (servicemeshoperator3)")
        _delete_resources("gateways.gateway.networking.k8s.io")
        _delete_resources("gatewayclasses.gateway.networking.k8s.io")
        _delete_resources("istios.sailoperator.io")
        _delete_resources("istiorevisions.sailoperator.io")
        _delete_resources("istiorevisiontags.sailoperator.io")
        _delete_resources("istiocnis.sailoperator.io")
        _delete_resources("ztunnels.sailoperator.io")

        _delete_olm_for("openshift-operators", "servicemeshoperator3", "servicemeshoperator3")

        for crd in [
            "authorizationpolicies.security.istio.io",
            "destinationrules.networking.istio.io",
            "envoyfilters.networking.istio.io",
            "gateways.networking.istio.io",
            "peerauthentications.security.istio.io",
            "proxyconfigs.networking.istio.io",
            "requestauthentications.security.istio.io",
            "serviceentries.networking.istio.io",
            "sidecars.networking.istio.io",
            "telemetries.telemetry.istio.io",
            "virtualservices.networking.istio.io",
            "wasmplugins.extensions.istio.io",
            "workloadentries.networking.istio.io",
            "workloadgroups.networking.istio.io",
            "istiocsrs.operator.openshift.io",
            "istios.sailoperator.io",
            "istiorevisions.sailoperator.io",
            "istiorevisiontags.sailoperator.io",
            "istiocnis.sailoperator.io",
            "ztunnels.sailoperator.io",
        ]:
            _force_delete_crd(crd)

        _oc("delete", "operator", "servicemeshoperator3.openshift-operators", "--ignore-not-found")
        log.info("ServiceMesh 3 cleanup complete")

    def _cleanup_servicemesh2() -> None:
        """Remove Service Mesh 2 (Maistra) operator and all its resources."""
        log.info("Cleaning up ServiceMesh 2 (Maistra / servicemeshoperator)")
        _delete_resources("servicemeshcontrolplanes.maistra.io")
        _delete_resources("servicemeshmemberrolls.maistra.io")
        _delete_resources("servicemeshmembers.maistra.io")

        _delete_olm_for("openshift-operators", "servicemeshoperator", "servicemeshoperator")

        _oc("delete", "namespace", "istio-system", "--ignore-not-found", "--timeout=60s")

        for crd in [
            "servicemeshcontrolplanes.maistra.io",
            "servicemeshmemberrolls.maistra.io",
            "servicemeshmembers.maistra.io",
            "exportedservicesets.federation.maistra.io",
            "importedservicesets.federation.maistra.io",
            "servicemeshpeers.federation.maistra.io",
        ]:
            _force_delete_crd(crd)

        _oc("delete", "operator", "servicemeshoperator.openshift-operators", "--ignore-not-found")
        log.info("ServiceMesh 2 cleanup complete")

    def _strip_resource_finalizers(ns: str) -> None:
        """Remove finalizers from every object inside a namespace."""
        log.info("  Stripping resource finalizers in '%s'", ns)
        for api_resource in _oc_lines(
            "api-resources", "--verbs=list", "--namespaced", "-o", "name",
        ):
            raw = _oc_out(
                "get", api_resource, "-n", ns,
                "--ignore-not-found",
                "-o", "jsonpath={range .items[?(@.metadata.finalizers)]}{.metadata.name}{'\\n'}{end}",
            )
            for obj_name in (l for l in raw.splitlines() if l.strip()):
                log.info("    Removing finalizers on %s/%s", api_resource, obj_name)
                _oc("patch", api_resource, obj_name, "-n", ns,
                    "--type=merge", "-p", '{"metadata":{"finalizers":[]}}')

    def _delete_namespace(ns: str) -> None:
        """Delete a namespace robustly, handling stuck-Terminating cases."""
        if not resources.exists("Namespace", ns):
            log.info("Namespace '%s' not found — skipping", ns)
            return
        log.info("Deleting namespace '%s'", ns)
        _oc("delete", "namespace", ns, "--ignore-not-found", "--timeout=60s")
        time.sleep(3)
        phase = subprocess.run(
            ["oc", "get", "ns", ns, "-o", "jsonpath={.status.phase}"],
            capture_output=True, text=True,
        ).stdout.strip()
        if phase == "Terminating":
            log.info("  Namespace '%s' stuck Terminating — force-clearing", ns)
            _cleanup_stale_apiservices()
            _oc("delete", "all", "--all", "-n", ns, "--ignore-not-found", "--timeout=30s")
            _strip_resource_finalizers(ns)
            _oc("patch", "namespace", ns,
                "--type=merge", "-p", '{"spec":{"finalizers":[]}}')
        wait.wait_until_deleted("Namespace", ns, timeout=180)

    # ------------------------------------------------------------------
    # 1. DataScienceCluster — delete then wait for component pods to drain
    # ------------------------------------------------------------------
    if resources.exists("DataScienceCluster", dsc_name):
        dsc.delete_dsc(dsc_name)
        log.info("Waiting for component pods to drain from workload namespaces")
        _drain_deadline = time.monotonic() + 120
        _drain_namespaces = [
            cluster_ns, "rhods-notebooks",
            "rhoai-model-registries", "redhat-ods-monitoring",
        ]
        while time.monotonic() < _drain_deadline:
            pod_count = sum(
                len(_oc_lines(
                    "get", "pods", "-n", _ns,
                    "--no-headers", "--ignore-not-found",
                    "--field-selector=status.phase!=Succeeded,status.phase!=Failed",
                ))
                for _ns in _drain_namespaces
            )
            if pod_count == 0:
                log.info("  All component pods drained")
                break
            log.info("  %d pod(s) still running — waiting", pod_count)
            time.sleep(10)
        else:
            log.info("  Drain timeout reached — proceeding with force cleanup")
    else:
        log.info("DataScienceCluster '%s' not found — skipping", dsc_name)

    # 2. DSCInitialization
    if resources.exists("DSCInitialization", dsci_name):
        dsc.delete_dsci(dsci_name)
    else:
        log.info("DSCInitialization '%s' not found — skipping", dsci_name)

    # 3. OLM resources: CSV, Subscription, OperatorGroup, InstallPlans
    csv_name = operators.resolve_csv_name(op_name, op_ns)
    if resources.exists("ClusterServiceVersion", csv_name, op_ns):
        log.info("Deleting CSV '%s'", csv_name)
        resources.delete_manifest("ClusterServiceVersion", csv_name, op_ns)
    else:
        log.info("CSV '%s' not found — skipping", csv_name)

    if resources.exists("Subscription", op_name, op_ns):
        log.info("Deleting Subscription '%s'", op_name)
        resources.delete_manifest("Subscription", op_name, op_ns)
    else:
        log.info("Subscription '%s' not found — skipping", op_name)

    og_name = config.get("operator", {}).get("group_name", op_name)
    if resources.exists("OperatorGroup", og_name, op_ns):
        log.info("Deleting OperatorGroup '%s'", og_name)
        resources.delete_manifest("OperatorGroup", og_name, op_ns)
    else:
        log.info("OperatorGroup '%s' not found — skipping", og_name)

    _oc("delete", "installplan", "--all", "-n", op_ns, "--ignore-not-found")

    # Cluster-wide sweep: stale CSVs/Subs matching rhoai/rhods/opendatahub
    log.info("Sweeping remaining rhods/rhoai CSVs cluster-wide")
    for ns_csv in _oc_lines(
        "get", "csv", "-A", "--no-headers",
        "-o", "custom-columns=NS:.metadata.namespace,NAME:.metadata.name",
    ):
        parts = ns_csv.split()
        if len(parts) == 2 and any(p in parts[1] for p in ("rhods", "rhoai", "opendatahub")):
            _oc("delete", "csv", parts[1], "-n", parts[0], "--ignore-not-found")

    # 4. Dependency operators — only if their Subscription is present on the cluster
    if _subscription_exists("servicemeshoperator3"):
        _cleanup_servicemesh3()
    else:
        log.info("servicemeshoperator3 not found — skipping")

    if _subscription_exists("servicemeshoperator") and not _subscription_exists("servicemeshoperator3"):
        _cleanup_servicemesh2()
    else:
        log.info("servicemeshoperator (v2/Maistra) not found — skipping")

    # 5. CRDs — operator-label-based + explicit RHOAI CRD list
    log.info("Deleting operator-owned CRDs")
    _oc("delete", "crd", "-l",
        f"operators.coreos.com/{op_name}.{op_ns}",
        "--ignore-not-found", "--timeout=60s")

    # Explicit sweep for CRDs the label may miss (e.g. created by components
    # whose controllers register CRDs under their own labels)
    log.info("Sweeping remaining RHOAI/ODH CRDs by name pattern")
    for crd_line in _oc_lines("get", "crd", "--no-headers", "-o", "custom-columns=NAME:.metadata.name"):
        if any(p in crd_line for p in (
            "opendatahub.io", "datasciencecluster", "dscinitialization",
            "kserve.io", "knative", "maistra.io", "sailoperator.io",
            "kubeflow.org", "ray.io", "codeflare.dev",
            "feast.dev", "llamastack.io", "mlflow", "ogx.io",
            "nim.opendatahub", "trustyai.opendatahub",
        )):
            _force_delete_crd(crd_line.strip())

    # 6. Stale APIServices — do this before namespace deletion
    _cleanup_stale_apiservices()

    # 7. Webhooks
    log.info("Removing validating and mutating webhooks")
    for wh_type in ("validatingwebhookconfigurations", "mutatingwebhookconfigurations"):
        for wh_name in _oc_lines(
            "get", wh_type, "--no-headers",
            "-o", "custom-columns=NAME:.metadata.name",
        ):
            if _matches(wh_name):
                log.info("  Deleting %s/%s", wh_type, wh_name)
                _oc("delete", wh_type, wh_name, "--ignore-not-found")

    # 8. Cluster-scoped RBAC
    log.info("Removing cluster-scoped RBAC")
    for rbac_kind in ("clusterrolebindings", "clusterroles"):
        for rbac_name in _oc_lines(
            "get", rbac_kind, "--no-headers",
            "-o", "custom-columns=NAME:.metadata.name",
        ):
            if _matches(rbac_name):
                _oc("delete", rbac_kind, rbac_name, "--ignore-not-found")

    # 9. Operator namespace (redhat-ods-operator)
    _delete_namespace(op_ns)

    # 10. Workload namespaces
    _static_workload_ns = [
        cluster_ns,               # redhat-ods-applications
        "rhods-notebooks",        # workbenches default
        "rhoai-model-registries", # modelregistry default
        "redhat-ods-monitoring",  # DSCI monitoring namespace
    ]

    if keep_workload_ns:
        log.info(
            "Keeping workload namespaces (--keep-workload-ns set): %s",
            ", ".join(_static_workload_ns),
        )
    else:
        _labelled = _oc_lines(
            "get", "namespaces",
            "-l", "opendatahub.io/generated-namespace=true",
            "--no-headers", "-o", "custom-columns=NAME:.metadata.name",
        )
        for ns in list(dict.fromkeys(_labelled + _static_workload_ns)):
            _delete_namespace(ns)

    log.info("Platform uninstall complete")


def validate_login() -> None:
    """Confirm the CLI is authenticated. Raises RuntimeError if unreachable."""
    log.info("Validating cluster login")
    try:
        resources.get("ClusterVersion", "version")
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Cannot reach cluster — run 'oc login' first. Detail: {exc}") from exc
    log.info("Cluster login confirmed")


def validate_permissions(operator_namespace: str) -> None:
    """Confirm create access on Subscriptions in the operator namespace.

    Uses SelfSubjectAccessReview — does not require cluster-admin explicitly,
    but RHOAI installation in practice does.
    """
    log.info("Validating RBAC permissions")
    review = {
        "apiVersion": "authorization.k8s.io/v1",
        "kind": "SelfSubjectAccessReview",
        "spec": {
            "resourceAttributes": {
                "namespace": operator_namespace,
                "verb": "create",
                "group": "operators.coreos.com",
                "resource": "subscriptions",
            }
        },
    }
    try:
        result = resources.create_dict(review)
        allowed = result.get("status", {}).get("allowed", False)
    except RuntimeError as exc:
        raise RuntimeError(f"Permission check failed: {exc}") from exc

    if not allowed:
        raise RuntimeError(
            f"Insufficient permissions: cannot create Subscriptions in '{operator_namespace}'. "
            "Cluster-admin role is required."
        )
    log.info("RBAC permissions confirmed")


def validate_storage(class_name: str) -> None:
    """Confirm a suitable StorageClass exists.

    If class_name is set, verifies that exact class. Otherwise, verifies
    at least one StorageClass is present on the cluster.
    """
    log.info("Validating storage classes")
    if class_name:
        if not resources.exists("StorageClass", class_name):
            raise RuntimeError(
                f"StorageClass '{class_name}' not found. "
                "Update storage.class_name in your config or leave it empty "
                "to use the cluster default."
            )
        log.info("StorageClass '%s' confirmed", class_name)
    else:
        classes = resources.list_resources("StorageClass")
        if not classes:
            raise RuntimeError(
                "No StorageClasses found. Create one before deploying RHOAI workloads."
            )
        names = [sc.get("metadata", {}).get("name", "?") for sc in classes]
        log.info("StorageClasses available: %s", ", ".join(names))


def validate_namespace(namespace: str) -> None:
    """Ensure the namespace exists, creating it if absent."""
    if not resources.exists("Namespace", namespace):
        log.info("Namespace '%s' not found — creating", namespace)
        resources.apply_dict({
            "apiVersion": "v1", "kind": "Namespace", "metadata": {"name": namespace},
        })
    else:
        log.info("Namespace '%s' exists", namespace)


def get_cluster_info() -> dict[str, Any]:
    """Return cluster metadata: version, topology, per-role CPU/memory/GPU, storage summary."""
    log.info("Collecting cluster info")
    cv      = resources.get("ClusterVersion", "version")
    nodes   = resources.list_resources("Node")
    classes = resources.list_resources("StorageClass")
    pvs     = resources.list_resources("PersistentVolume")

    worker_nodes = [n for n in nodes if _has_role(n, "worker")] or nodes

    return {
        "openshift_version": _openshift_version(cv),
        "topology":          "SNO" if len(nodes) == 1 else "Multi-node",
        "node_count":        len(nodes),
        "worker_count":      len(worker_nodes),
        "worker_nodes":      _worker_details(worker_nodes),
        "storage_summary":   _storage_summary(classes, pvs),
    }


def _openshift_version(cluster_version: dict[str, Any]) -> str:
    return cluster_version.get("status", {}).get("desired", {}).get("version", "unknown")


def _worker_details(worker_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return per-node CPU/memory/GPU breakdown for each worker node."""
    result = []
    for n in worker_nodes:
        name = n.get("metadata", {}).get("name", "?")
        cap  = n.get("status", {}).get("capacity", {})
        result.append({
            "name":   name,
            "cpu":    f"{_parse_cpu(cap.get('cpu', '0')) // 1000} cores",
            "memory": f"{_parse_memory_ki(cap.get('memory', '0Ki')) // (1024 * 1024)} GiB",
            "gpu":    int(cap.get("nvidia.com/gpu", 0)),
        })
    return result


def _storage_summary(
    classes: list[dict[str, Any]],
    pvs: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """Return per-StorageClass used/available GiB aggregated from PV phases."""
    # Phase "Bound"               = in use by a PVC
    # Phase "Available"/"Released"= pre-provisioned or pending recycle
    pv_bound_ki:     dict[str, int] = {}
    pv_available_ki: dict[str, int] = {}

    for pv in pvs:
        sc    = pv.get("spec", {}).get("storageClassName", "unknown")
        ki    = _parse_memory_ki(pv.get("spec", {}).get("capacity", {}).get("storage", "0"))
        phase = pv.get("status", {}).get("phase", "Unknown")
        if phase == "Bound":
            pv_bound_ki[sc]     = pv_bound_ki.get(sc, 0) + ki
        elif phase in ("Available", "Released"):
            pv_available_ki[sc] = pv_available_ki.get(sc, 0) + ki

    summary: dict[str, dict[str, str]] = {}
    for sc_obj in classes:
        sc = sc_obj.get("metadata", {}).get("name", "?")
        summary[sc] = {
            "used":      f"{pv_bound_ki.get(sc, 0) // (1024 * 1024)} GiB",
            "available": f"{pv_available_ki.get(sc, 0) // (1024 * 1024)} GiB",
        }
    return summary


def _has_role(node: dict[str, Any], role: str) -> bool:
    labels = node.get("metadata", {}).get("labels", {})
    return f"node-role.kubernetes.io/{role}" in labels


def _parse_cpu(value: str) -> int:
    """Return CPU as millicores. Handles '4' (cores) and '4000m' (millicores)."""
    value = value.strip()
    if value.endswith("m"):
        return int(value[:-1])
    return int(value) * 1000


def _parse_memory_ki(value: str) -> int:
    """Return memory in kibibytes. Handles Ki, Mi, Gi, and plain bytes."""
    value = value.strip()
    if value.endswith("Ki"):
        return int(value[:-2])
    if value.endswith("Mi"):
        return int(value[:-2]) * 1024
    if value.endswith("Gi"):
        return int(value[:-2]) * 1024 * 1024
    if value.endswith("Ti"):
        return int(value[:-2]) * 1024 * 1024 * 1024
    # plain bytes
    return int(value) // 1024
