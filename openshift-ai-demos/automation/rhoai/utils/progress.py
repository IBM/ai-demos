"""Reusable progress indicators for long-running CLI operations.

Three primitives:

step() — phase indicator
    Owns a spinner and elapsed time.  Use for any operation where the
    user should see activity while waiting.

    with step("Applying S3 storage credentials"):
        do_work()

    with step("Waiting for InferenceService") as s:
        wait_until_ready(..., on_tick=s.tick)

header_step() + sub_step() — phased grouping
    header_step() prints the phase label immediately and owns elapsed time.
    sub_step() prints a single completed milestone line the instant its
    block exits — no spinner, no Live context, no buffering.

    with header_step("Checking RHOAI platform", outcome="Platform ready"):
        with sub_step("Operator ready"):
            check_operator()
        with sub_step("DSCI 'default-dsci' ready"):
            check_dsci()

elapsed_timer() — total command duration
    Captures wall-clock time for an entire command so callers can display
    a "Total: Xm Ys" line in the summary without threading timing through
    every function call.

    with elapsed_timer() as t:
        do_all_the_work()
    summary(total=t.seconds)

on_tick protocol
----------------
Pass ``on_tick=spinner.tick`` to any long-running call that accepts it.
The callee calls ``on_tick(elapsed_seconds)`` so the spinner text shows
live elapsed time.
"""

import sys
import time
from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.status import Status

_console = Console(stderr=False, highlight=False)


class _Spinner:
    """Thin wrapper around a Rich Status so callers can push tick updates."""

    def __init__(self, status: Status, label: str) -> None:
        self._status = status
        self._label  = label
        self._skipped = False

    def tick(self, elapsed: float) -> None:
        """Update the spinner text with current elapsed time."""
        self._status.update(f"{self._label}  ({int(elapsed)}s)...")

    def skip(self) -> None:
        """Mark this step as unavailable so the outcome line shows ⚠ instead of ✔."""
        self._skipped = True


@contextmanager
def step(label: str) -> Generator[_Spinner, None, None]:
    """Context manager: spin while work runs, print outcome when done.

    Args:
        label: Human-readable description shown next to the spinner.

    Yields:
        _Spinner — call ``spinner.tick(elapsed)`` to update elapsed time,
                   or ``spinner.skip()`` to mark the step as skipped (⚠).

    Example::

        with step("Waiting for InferenceService") as s:
            wait_until_ready(..., on_tick=s.tick)

        with step("Validating model inference") as s:
            if unreachable:
                s.skip()
    """
    start = time.monotonic()

    if not sys.stdout.isatty():
        # Non-TTY (CI, pipe): print static lines, no spinner codes.
        _console.print(f"{label}...")
        spinner = _Spinner(Status(""), label)
        try:
            yield spinner
            elapsed = int(time.monotonic() - start)
            if spinner._skipped:
                _console.print(f"\u26a0  {label}  unavailable  ({elapsed}s)")
            else:
                _console.print(f"\u2714  {label}  ({elapsed}s)")
        except Exception:
            _console.print(f"\u2716  {label}  failed")
            raise
        return

    with _console.status(f"{label}...", spinner="dots") as rich_status:
        spinner = _Spinner(rich_status, label)
        try:
            yield spinner
        except Exception:
            elapsed = int(time.monotonic() - start)
            _console.print(f"\u2716  {label}  failed  ({elapsed}s)")
            raise

    elapsed = int(time.monotonic() - start)
    if spinner._skipped:
        _console.print(f"\u26a0  {label}  unavailable  ({elapsed}s)")
    else:
        _console.print(f"\u2714  {label}  ({elapsed}s)")


@contextmanager
def header_step(label: str, outcome: str | None = None) -> Generator[None, None, None]:
    """Phase indicator that prints its label immediately and owns elapsed time.

    Prints ``<label>...`` before any work starts.  Sub-steps nested inside
    use sub_step(), which prints each milestone the instant it completes.
    No spinner or Live context is created here — output streams freely.

    Args:
        label:   Header text printed before work starts.
        outcome: Text for the final ✔ / ✖ line.  Defaults to *label*.

    Example::

        with header_step("Checking RHOAI platform", outcome="Platform ready"):
            with sub_step("Operator ready"):
                check_operator()
    """
    resolved = outcome if outcome is not None else label
    start    = time.monotonic()

    _console.print(f"\n{label}...\n")

    try:
        yield
    except Exception:
        elapsed = int(time.monotonic() - start)
        _console.print(f"\n\u2716  {resolved}  failed  ({elapsed}s)")
        raise

    elapsed = int(time.monotonic() - start)
    _console.print(f"\n\u2714  {resolved}  ({elapsed}s)")


@contextmanager
def sub_step(label: str) -> Generator[None, None, None]:
    """Milestone indicator for use inside a header_step() block.

    Prints "  ✔  <label>" the moment its block exits cleanly, or
    "  ✖  <label>  failed" on exception.  No spinner, no Live context —
    _console.print() outside any active Live flushes immediately.

    Args:
        label: Completion description, e.g. "Operator ready".

    Example::

        with header_step("Checking RHOAI platform", outcome="Platform ready"):
            with sub_step("Operator ready"):
                check_operator()
    """
    try:
        yield
        _console.print(f"  \u2714  {label}")
    except Exception:
        _console.print(f"  \u2716  {label}  failed")
        raise


class _Timer:
    """Holds the elapsed seconds after an elapsed_timer() block exits."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.seconds: float = 0.0

    def _begin(self) -> None:
        self._start = time.monotonic()

    def _end(self) -> None:
        self.seconds = time.monotonic() - self._start

    @property
    def formatted(self) -> str:
        """Return elapsed time as 'Xm Ys' (minutes omitted when < 60 s)."""
        total = int(self.seconds)
        minutes, secs = divmod(total, 60)
        if minutes:
            return f"{minutes}m {secs}s"
        return f"{secs}s"


@contextmanager
def elapsed_timer() -> Generator[_Timer, None, None]:
    """Context manager that measures total wall-clock time for a command.

    Yields a ``_Timer`` whose ``seconds`` and ``formatted`` attributes are
    populated once the block exits (cleanly or via exception).  Use this
    at the outermost level of a command to produce a "Total: Xm Ys" line
    in the summary.

    Example::

        with elapsed_timer() as t:
            do_all_the_work()
        print_summary(total=t.formatted)
    """
    timer = _Timer()
    timer._begin()
    try:
        yield timer
    finally:
        timer._end()
