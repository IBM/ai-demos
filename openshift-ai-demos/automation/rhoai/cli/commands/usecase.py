"""Use case commands — rhoai usecase <name> <subcommand>.

    rhoai usecase deploy   <name>   deploy the named use case
    rhoai usecase verify  <name>   verify the deployment
    rhoai usecase cleanup <name>   remove use-case resources
    rhoai usecase list             show all registered use cases
"""

from pathlib import Path

import typer

from rhoai.config.loader import load_config
from rhoai.usecases import registry
from rhoai.utils.errors import friendly_error
from rhoai.utils.logger import get_logger

app = typer.Typer(help="Deploy and manage AI use cases.")
log = get_logger(__name__)

_config_option = typer.Option(None, "--config", "-c", help="Path to config YAML.")
_name_argument = typer.Argument("fraud-detection", help="Use case name.")


@app.command()
def deploy(
    name: str = _name_argument,
    config_file: Path | None = _config_option,
) -> None:
    """Deploy the named use case."""
    config    = load_config(config_file)
    dep_cfg   = config.get("deployment", {})
    namespace = dep_cfg.get("namespace") or config["platform"]["namespace"]

    typer.echo(f"\nDeploying  : {name}")
    typer.echo(f"Namespace  : {namespace}\n")

    if config_file:
        config["_config_file"] = str(config_file)
    config["_use_case"] = name

    try:
        registry.get(name).deploy(config)
    except Exception as exc:  # noqa: BLE001
        _exit_with_error(exc)


@app.command()
def verify(
    name: str = _name_argument,
    config_file: Path | None = _config_option,
) -> None:
    """Verify the named use case deployment."""
    config    = load_config(config_file)
    dep_cfg   = config.get("deployment", {})
    namespace = dep_cfg.get("namespace") or config["platform"]["namespace"]

    typer.echo(f"\nVerifying  : {name}")
    typer.echo(f"Namespace  : {namespace}\n")

    if config_file:
        config["_config_file"] = str(config_file)
    config["_use_case"] = name

    try:
        registry.get(name).verify(config)
    except Exception as exc:  # noqa: BLE001
        _exit_with_error(exc)


@app.command()
def cleanup(
    name: str = _name_argument,
    config_file: Path | None = _config_option,
    delete_platform: bool = typer.Option(
        False,
        "--delete-platform",
        help="Also remove DSC and DSCI after use-case cleanup. Use with caution.",
    ),
) -> None:
    """Remove the named use case resources."""
    from rhoai.platform import dsc

    config    = load_config(config_file)
    dep_cfg   = config.get("deployment", {})
    namespace = dep_cfg.get("namespace") or config["platform"]["namespace"]

    typer.echo(f"\nRemoving   : {name}")
    typer.echo(f"Namespace  : {namespace}\n")

    try:
        registry.get(name).cleanup(config)
        if delete_platform:
            from rhoai.utils.progress import header_step, step
            platform_namespace = config["platform"]["namespace"]
            dsc_timeout        = config["timeouts"].get("dsc_ready", 600)
            with header_step("Removing platform resources", outcome="Platform resources removed"):
                with step("Removing DSC"):
                    dsc.delete_dsc(config["dsc"]["name"])
                with step("Waiting for component pods to terminate") as s:
                    dsc.wait_until_no_deployments(platform_namespace, timeout=dsc_timeout)
                with step("Removing DSCI"):
                    dsc.delete_dsci(config["dsc"]["dsci_name"])
    except Exception as exc:  # noqa: BLE001
        _exit_with_error(exc)


@app.command(name="list")
def list_cmd() -> None:
    """List all registered use case names."""
    for name in registry.list_available():
        typer.echo(f"  {name}")




def _exit_with_error(exc: Exception) -> None:
    """Print a clean one-line error and exit 1. Never shows a traceback."""
    log.debug("Unhandled exception", exc_info=exc)
    typer.echo(f"\n✖  {friendly_error(exc)}", err=True)
    log.debug("Exception type: %s", type(exc).__name__)
    raise typer.Exit(code=1)
