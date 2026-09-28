"""Platform commands — rhoai platform <subcommand>.

    rhoai platform init      validate cluster, install operator, initialize DSCI
    rhoai platform enable    enable one or more DSC components
    rhoai platform disable   disable one or more DSC components
    rhoai platform setup     full one-shot bootstrap (calls init then enable)
    rhoai platform uninstall remove all RHOAI platform resources
    rhoai platform status    report RHOAI platform health
    rhoai platform inspect   display factual cluster information (read-only)

Output verbosity
----------------
  Default (INFO log level):  Structured summary output only.
  --log-level DEBUG:          Full log stream + structured output.
"""

from pathlib import Path

import typer

from rhoai.config.loader import load_config
from rhoai.ocp import resources
from rhoai.platform import dsc, operators, prepare
from rhoai.platform import verify as platform_verify
from rhoai.utils.logger import get_logger

app = typer.Typer(help="Manage the RHOAI platform.")
log = get_logger(__name__)

_config_option  = typer.Option(None, "--config", "-c", help="Path to config YAML.")
_components_arg = typer.Argument(..., help="Component names, e.g. kserve trustyai.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _step(msg: str) -> None:
    """Print an indented progress line."""
    typer.echo(f"    {msg}")


def _print_platform_summary(config: dict) -> None:
    """Print the RHOAI Platform status table (operator + DSCI + DSC + components)."""
    op_name   = config["operator"]["name"]
    op_ns     = config["operator"]["namespace"]
    dsc_name  = config["dsc"]["name"]
    dsci_name = config["dsc"]["dsci_name"]

    results     = platform_verify.verify_platform(config)
    op_result   = results[0]
    dsci_result = results[1]
    dsc_result  = results[2]
    op_failed   = not op_result.passed

    op_display = op_name
    if op_result.passed:
        try:
            csv = operators.get_csv_info(op_name, op_ns)
            op_display = f"{csv['name']}  {csv['version']}"
        except Exception:  # noqa: BLE001
            pass

    typer.echo("\nRHOAI Platform")
    typer.echo(_status_row("Operator",       op_display,  op_result))
    typer.echo(_status_row("Initialization", dsci_name,   dsci_result, skip=op_failed))
    typer.echo(_status_row("DataScienceCluster", dsc_name, dsc_result,  skip=op_failed))

    if dsc_result.passed:
        try:
            states  = dsc.get_component_states(dsc_name)
            managed = sorted(c for c, s in states.items() if s == "Managed")
            if managed:
                typer.echo("\n  Components")
                for comp in managed:
                    typer.echo(f"    \u2714  {comp}")
        except Exception:  # noqa: BLE001
            pass


def _status_row(
    label: str,
    name: str,
    result: platform_verify.CheckResult,
    skip: bool = False,
) -> str:
    if skip:
        mark   = "\u2013"
        detail = "(skipped)"
    elif result.passed:
        mark   = "\u2714"
        detail = "Ready" if label != "Operator" else "Succeeded"
    else:
        mark   = "\u2718"
        detail = result.message or "not installed"
    return f"  {label:<16}  {name:<36}  {mark}  {detail}"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command(name="init")
def init_cmd(
    config_file: Path | None = _config_option,
    channel: str | None = typer.Option(
        None, "--channel",
        help=(
            "OLM channel, e.g. 'stable-3.x', 'stable-3.4', 'fast-3.x', 'beta'. "
            "Run: oc get packagemanifest rhods-operator "
            "-o jsonpath='{.status.channels[*].name}' to list available channels."
        ),
    ),
    version: str | None = typer.Option(
        None, "--version",
        help="Pin to a specific CSV, e.g. '3.4.2' or 'rhods-operator.3.4.2'.",
    ),
    source: str | None = typer.Option(
        None, "--source",
        help="CatalogSource name. Defaults to 'redhat-operators'.",
    ),
) -> None:
    """Validate prerequisites, install the operator, and initialize DSCI.

    Does not enable any DSC components — run 'enable' afterwards,
    or use 'setup' to do everything in one call.

    Examples:
        rhoai platform init --channel stable-3.x
        rhoai platform init --channel stable-3.4 --version 3.4.2
        rhoai platform init --channel beta --source cs-rhoai-fbc-fragment
    """
    config = load_config(config_file)

    if channel:
        config["operator"]["channel"] = channel
    if version:
        if version[0].isdigit():
            version = f"rhods-operator.{version}"
        config["operator"]["version"] = version
    if source:
        config["operator"]["source"] = source

    typer.echo("\nInitializing RHOAI platform...")
    prepare.init_platform(config)

    op_name   = config["operator"]["name"]
    op_ns     = config["operator"]["namespace"]
    dsci_name = config["dsc"]["dsci_name"]

    op_display = op_name
    try:
        csv = operators.get_csv_info(op_name, op_ns)
        op_display = f"{csv['name']}  {csv['version']}"
    except Exception:  # noqa: BLE001
        pass

    dsci_phase = resources.status("DSCInitialization", dsci_name).get("phase", "Unknown")

    typer.echo("\n  ✔  RHOAI platform initialized")
    typer.echo(f"     Operator  : {op_display}")
    typer.echo(f"     Platform  : {dsci_phase}")
    typer.echo("")


@app.command(name="enable")
def enable_cmd(
    components: list[str] = _components_arg,
    config_file: Path | None = _config_option,
) -> None:
    """Enable one or more DSC components. Requires 'init' to have run first.

    Additive and idempotent — only the named components are changed;
    existing Managed components stay Managed.

    Example:
        rhoai platform enable kserve trustyai
    """
    config = load_config(config_file)
    comp_list = ", ".join(components)

    typer.echo(f"\nEnabling: {comp_list}")
    prepare.install_component(config, components)
    typer.echo(f"\n  ✔  Enabled: {comp_list}")

    # Show all currently-managed components
    dsc_name = config["dsc"]["name"]
    try:
        states  = dsc.get_component_states(dsc_name)
        managed = sorted(c for c, s in states.items() if s == "Managed")
        if managed:
            typer.echo("\n  Components")
            for comp in managed:
                typer.echo(f"    \u2714  {comp}")
    except Exception:  # noqa: BLE001
        pass
    typer.echo("")


@app.command(name="disable")
def disable_cmd(
    components: list[str] = _components_arg,
    config_file: Path | None = _config_option,
) -> None:
    """Disable one or more DSC components (set managementState to Removed).

    Idempotent — components already Removed stay Removed;
    unrelated components are left exactly as they are.

    Example:
        rhoai platform disable trustyai ray
    """
    config = load_config(config_file)
    comp_list = ", ".join(components)

    typer.echo(f"\nDisabling: {comp_list}")
    prepare.remove_component(config, components)
    typer.echo(f"\n  ✔  Disabled: {comp_list}")

    dsc_name = config["dsc"]["name"]
    try:
        states  = dsc.get_component_states(dsc_name)
        managed = sorted(c for c, s in states.items() if s == "Managed")
        if managed:
            typer.echo("\n  Components still enabled")
            for comp in managed:
                typer.echo(f"    \u2714  {comp}")
        else:
            typer.echo("\n  No components currently enabled")
    except Exception:  # noqa: BLE001
        pass
    typer.echo("")


@app.command(name="uninstall")
def uninstall_cmd(
    config_file: Path | None = _config_option,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
    keep_workload_ns: bool = typer.Option(
        False, "--keep-workload-ns",
        help=(
            "Preserve workload namespaces (redhat-ods-applications, rhods-notebooks, "
            "rhoai-model-registries, redhat-ods-monitoring). "
            "By default they are deleted for a clean reinstall."
        ),
    ),
) -> None:
    """Remove all RHOAI platform resources for a clean cluster state.

    Workload namespaces are deleted by default. Pass --keep-workload-ns to
    preserve user notebooks and pipelines.
    """
    if not yes:
        typer.confirm(
            "This will remove all RHOAI platform resources. Continue?",
            abort=True,
        )
    config = load_config(config_file)

    import rhoai.platform.dsc as _dsc
    from rhoai.ocp import resources as _res

    R = "⟳"  # ⟳  fires before work starts
    D = "✔"  # ✔  fires after work completes

    _orig_delete_dsc      = _dsc.delete_dsc
    _orig_delete_dsci     = _dsc.delete_dsci
    _orig_delete_manifest = _res.delete_manifest
    _orig_exists          = _res.exists

    # Each flag ensures its step line prints exactly once
    _printed = {"dsc": False, "olm": False, "ns": False}

    def _once(key: str, msg: str) -> None:
        if not _printed[key]:
            typer.echo(f"  {R}  {msg}")
            _printed[key] = True

    # Hook 1: DSC deletion (first thing uninstall_platform does)
    def _patched_delete_dsc(name):
        _once("dsc", "Removing DataScienceCluster and DSCInitialization")
        _orig_delete_dsc(name)

    # Hook 1b: DSCI-only path (when DSC was already absent)
    def _patched_delete_dsci(name):
        _once("dsc", "Removing DSCInitialization")
        _orig_delete_dsci(name)

    # Hook 2: OLM phase — first CSV delete triggers the step line
    def _patched_delete_manifest(kind, name, namespace=None):
        if kind == "ClusterServiceVersion":
            _once("olm", "Removing operator (CSV, Subscription, OperatorGroup, InstallPlans)")
        _orig_delete_manifest(kind, name, namespace)

    # Hook 3: Namespace phase — _delete_namespace inside uninstall_platform
    # calls resources.exists("Namespace", ns) before every deletion attempt.
    # The first call where the namespace actually exists triggers the step line,
    # so it prints right before the first namespace deletion starts.
    def _patched_exists(kind, name, namespace=None):
        result = _orig_exists(kind, name, namespace)
        if kind == "Namespace" and result:
            if keep_workload_ns:
                _once("ns", "Removing operator namespace  (workload namespaces preserved)")
            else:
                _once("ns", "Removing namespaces")
        return result

    _dsc.delete_dsc      = _patched_delete_dsc
    _dsc.delete_dsci     = _patched_delete_dsci
    _res.delete_manifest = _patched_delete_manifest
    _res.exists          = _patched_exists

    typer.echo("\nRemoving RHOAI platform...")

    try:
        prepare.uninstall_platform(config, keep_workload_ns=keep_workload_ns)
    finally:
        _dsc.delete_dsc      = _orig_delete_dsc
        _dsc.delete_dsci     = _orig_delete_dsci
        _res.delete_manifest = _orig_delete_manifest
        _res.exists          = _orig_exists

    # CRDs / webhooks / RBAC run as batch oc calls with no Python-level hook.
    # Print with ✔ after completion — they always run.
    typer.echo(f"  {D}  Removed CRDs, webhooks and cluster-scoped RBAC")

    typer.echo("\n  ✔  RHOAI platform removed.")


@app.command(name="setup")
def setup_cmd(
    config_file: Path | None = _config_option,
    channel: str | None = typer.Option(None, "--channel", help="OLM channel. Overrides config file."),
    version: str | None = typer.Option(None, "--version", help="Pin to a specific CSV version."),
    source:  str | None = typer.Option(None, "--source",  help="CatalogSource name."),
) -> None:
    """Full one-shot bootstrap: init + enable DSC components.

    Equivalent to 'init' followed by applying the base DSC manifest
    (or the components list in your config file).
    For fine-grained control use 'init' + 'enable' separately.
    """
    config = load_config(config_file)
    if channel:
        config["operator"]["channel"] = channel
    if version:
        if version[0].isdigit():
            version = f"rhods-operator.{version}"
        config["operator"]["version"] = version
    if source:
        config["operator"]["source"] = source

    ch  = config["operator"]["channel"]
    src = config["operator"].get("source", "redhat-operators")

    typer.echo("\nSetting up RHOAI platform...")
    prepare.bootstrap_platform(config)

    _print_platform_summary(config)
    typer.echo("")


@app.command()
def status(
    config_file: Path | None = _config_option,
) -> None:
    """Report the health of the installed RHOAI platform."""
    config = load_config(config_file)

    results     = platform_verify.verify_platform(config)
    op_result   = results[0]
    dsci_result = results[1]
    dsc_result  = results[2]
    op_failed   = not op_result.passed

    op_name   = config["operator"]["name"]
    op_ns     = config["operator"]["namespace"]
    dsc_name  = config["dsc"]["name"]
    dsci_name = config["dsc"]["dsci_name"]

    op_display = op_name
    if op_result.passed:
        try:
            csv = operators.get_csv_info(op_name, op_ns)
            op_display = f"{csv['name']}  {csv['version']}"
        except Exception:  # noqa: BLE001
            pass

    typer.echo("\nRHOAI Platform")
    typer.echo(_status_row("Operator",             op_display,  op_result))
    typer.echo(_status_row("Initialization",       dsci_name,   dsci_result, skip=op_failed))
    typer.echo(_status_row("DataScienceCluster",   dsc_name,    dsc_result,  skip=op_failed))

    if dsc_result.passed:
        try:
            states  = dsc.get_component_states(dsc_name)
            managed = sorted(c for c, s in states.items() if s == "Managed")
            if managed:
                typer.echo("\n  Components")
                for comp in managed:
                    typer.echo(f"    \u2714  {comp}")
        except Exception:  # noqa: BLE001
            pass

    failed = [r for r in results if not r.passed]
    typer.echo("")
    if not failed:
        typer.echo("Platform is healthy.")
        typer.echo("")
        return

    typer.echo("Platform is not healthy.", err=True)

    # Generic actionable hints — keeps the message correct regardless of
    # which combination of checks failed or what command was used to install.
    auth_failed = any(
        r.message and ("401" in r.message or "Unauthorized" in r.message)
        for r in failed
    )
    if auth_failed:
        typer.echo("  \u2139  Run 'oc login <cluster-url>' to authenticate.", err=True)
    else:
        typer.echo("  \u2139  Run 'rhoai platform setup' to reinstall.", err=True)
    typer.echo("  \u2139  For details run: rhoai --log-level DEBUG platform status", err=True)
    typer.echo("", err=True)

    raise typer.Exit(code=1)


@app.command()
def inspect(
    config_file: Path | None = _config_option,
) -> None:
    """Display factual cluster information. Read-only, no modifications."""
    typer.echo("\nDumping cluster info...")
    info         = prepare.get_cluster_info()
    worker_nodes = info["worker_nodes"]
    has_gpu      = any(n["gpu"] for n in worker_nodes)

    node_count   = info["node_count"]
    worker_count = info["worker_count"]
    master_count = node_count - worker_count
    topology_detail = (
        f"({node_count} node)"
        if node_count == 1
        else f"({node_count} nodes: {master_count} master, {worker_count} workers)"
    )

    typer.echo("\n\u2699  Cluster")
    typer.echo(f"  OpenShift    {info['openshift_version']}")
    typer.echo(f"  Topology     {info['topology']}  {topology_detail}")

    typer.echo("\n\u26f6  Worker Nodes")
    if has_gpu:
        typer.echo(f"  {'NAME':<48}  {'CPU':<10}  {'MEMORY':<10}  GPU")
        for node in worker_nodes:
            gpu_str = str(node["gpu"]) if node["gpu"] else "\u2014"
            typer.echo(f"  {node['name']:<48}  {node['cpu']:<10}  {node['memory']:<10}  {gpu_str}")
    else:
        typer.echo(f"  {'NAME':<48}  {'CPU':<10}  MEMORY")
        for node in worker_nodes:
            typer.echo(f"  {node['name']:<48}  {node['cpu']:<10}  {node['memory']}")

    typer.echo("\n💾  Storage Classes")
    storage = info["storage_summary"]
    if storage:
        typer.echo(f"  {'NAME':<40}  BOUND")
        for sc_name in sorted(storage):
            bound = storage[sc_name]["used"]
            typer.echo(f"  {sc_name:<40}  {bound}")
    else:
        typer.echo("  (none)")

    typer.echo("")
