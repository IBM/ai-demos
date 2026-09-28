"""Platform health verification.

Checks only the core platform components: operator CSV, DSCI, and DSC.
InferenceServices and storage are use-case concerns — each use case's
verify() is responsible for those.
"""

from dataclasses import dataclass
from typing import Any

from rhoai.platform import dsc, operators
from rhoai.utils.errors import friendly_error
from rhoai.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str = ""


def verify_platform(config: dict[str, Any]) -> list[CheckResult]:
    """Run core platform health checks: operator, DSCI, and DSC.

    Returns the list of CheckResults. Does not raise — callers inspect
    the results and decide how to exit.
    """
    op_name = config["operator"]["name"]
    op_ns   = config["operator"]["namespace"]
    return [
        _run("RHOAI Operator",     lambda: operators.verify(op_name, op_ns)),
        _run("DSCInitialization",  lambda: dsc.verify_dsci(config["dsc"]["dsci_name"])),
        _run("DataScienceCluster", lambda: dsc.verify(config["dsc"]["name"])),
    ]


def _run(name: str, check_fn) -> CheckResult:
    try:
        check_fn()
        return CheckResult(name=name, passed=True)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name=name, passed=False, message=friendly_error(exc))
