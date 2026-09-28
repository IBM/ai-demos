"""Logging configuration.

Call configure() once at startup (from cli/main.py).
All other modules call get_logger(__name__).

Log-level contract
------------------
  INFO    (default)  Logger output suppressed — only structured typer.echo
                     output is shown. Clean UX for day-to-day / demo use.
  DEBUG              All log.debug / log.info lines printed alongside the
                     structured output. Use for development / troubleshooting.
  WARNING / ERROR    Only that severity and above from the logger.
"""

import logging
import sys


def configure(level: str = "INFO") -> None:
    """Configure root logger. Called once from cli/main.py.

    At INFO the handler threshold is raised to WARNING so all log.info()
    calls inside platform/ocp modules are silent — typer.echo() structured
    output is the primary UX.  --log-level DEBUG re-enables the full stream.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    ))

    # INFO  → handler emits WARNING+ only (log.info is silent)
    # DEBUG → everything flows through
    # WARNING/ERROR → only those levels
    handler.setLevel(logging.WARNING if numeric == logging.INFO else numeric)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # root accepts all; handler filters
    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module."""
    return logging.getLogger(name)
