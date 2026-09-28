"""Polling helpers for Kubernetes resource state transitions.

All reads delegate to ocp.resources.
Dependency is one-way: wait.py → resources.py (never the reverse).
"""

import time
from collections.abc import Callable
from typing import Any

from rhoai.ocp import resources
from rhoai.utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_INTERVAL = 10  # seconds between polls


def wait_until_ready(
    kind: str,
    name: str,
    namespace: str | None = None,
    timeout: int = 300,
    interval: int = _DEFAULT_INTERVAL,
    on_tick: Callable[[float], Any] | None = None,
) -> None:
    """Block until ocp.resources.is_ready() returns True. Raises TimeoutError."""
    wait_until(
        lambda: resources.is_ready(kind, name, namespace),
        f"{kind}/{name} ready",
        timeout,
        interval,
        on_tick=on_tick,
    )


def wait_until_deleted(
    kind: str,
    name: str,
    namespace: str | None = None,
    timeout: int = 120,
    interval: int = 5,
    on_tick: Callable[[float], Any] | None = None,
) -> None:
    """Block until the resource no longer exists. Raises TimeoutError."""
    wait_until(
        lambda: not resources.exists(kind, name, namespace),
        f"{kind}/{name} deleted",
        timeout,
        interval,
        on_tick=on_tick,
    )


def wait_until(
    condition: Callable[[], bool],
    description: str,
    timeout: int = 300,
    interval: int = _DEFAULT_INTERVAL,
    on_tick: Callable[[float], Any] | None = None,
) -> None:
    """Block until condition() returns True or timeout is exceeded.

    Args:
        condition:   Called each interval; returns True when done.
        description: Human-readable label used in log messages and TimeoutError.
        timeout:     Maximum seconds to wait before raising TimeoutError.
        interval:    Seconds between condition checks.
        on_tick:     Optional callback invoked after each sleep with elapsed
                     seconds as the sole argument.  Use this to push live
                     elapsed-time updates into a progress spinner.

    Raises:
        TimeoutError: If condition() does not return True within timeout seconds.
    """
    start    = time.monotonic()
    deadline = start + timeout
    while time.monotonic() < deadline:
        if condition():
            log.debug("Ready: %s", description)
            return
        elapsed = time.monotonic() - start
        log.debug("Waiting: %s  (%.0fs)", description, elapsed)
        if on_tick:
            on_tick(elapsed)
        time.sleep(interval)
    raise TimeoutError(f"Timed out after {timeout}s waiting for: {description}")
