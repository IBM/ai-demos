"""CLI entry point — rhoai-automation.

Registers command groups:
    rhoai platform ...
    rhoai usecase  ...

Declared as the console script in pyproject.toml:
    [project.scripts]
    rhoai = "cli.main:app"

Log-level behaviour
-------------------
  (default / INFO)  Logger output is suppressed — only structured typer.echo
                    output is shown.  Clean UX for day-to-day use.
  --log-level DEBUG All log.info / log.debug lines are printed alongside the
                    structured output.  Useful for developers and CI debugging.
"""

import typer

from rhoai.cli.commands import platform as platform_cmd
from rhoai.cli.commands import usecase as usecase_cmd
from rhoai.utils import logger

app = typer.Typer(
    name="rhoai",
    help="rhoai-automation — deploy and manage Red Hat OpenShift AI use cases.",
    no_args_is_help=True,
)

app.add_typer(platform_cmd.app, name="platform")
app.add_typer(usecase_cmd.app,  name="usecase")


@app.callback()
def _callback(
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l",
        help=(
            "Logger verbosity. "
            "INFO (default): logger lines hidden, only structured output shown. "
            "DEBUG: all log lines printed alongside structured output."
        ),
    ),
) -> None:
    logger.configure(level=log_level)


if __name__ == "__main__":
    app()
