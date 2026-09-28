"""Unit tests for rhoai.usecases.fraud_detection.deploy.print_summary."""

import importlib
from pathlib import Path

from rhoai.usecases.fraud_detection.assets import ModelResult

# The fraud_detection package re-exports the deploy() function under the name
# "deploy", shadowing the submodule; import the module object explicitly.
deploy = importlib.import_module("rhoai.usecases.fraud_detection.deploy")


def _summary_text(results: list[ModelResult]) -> str:
    """Render print_summary to a string via the module's rich Console."""
    with deploy._console.capture() as cap:
        deploy.print_summary(results, "fraud-detection", "ns")
    return cap.get()


class TestPrintSummaryGeneratedArtifacts:
    def test_shows_generated_config_and_request_paths(self) -> None:
        r = ModelResult(name="loan", endpoint="https://loan.example")
        r.generated_config_path = Path("/repo/inputs/loan/config.pbtxt")
        r.generated_request_path = Path("/repo/inputs/loan_generated_request.json")

        out = _summary_text([r])

        assert "/repo/inputs/loan/config.pbtxt" in out
        assert "/repo/inputs/loan_generated_request.json" in out
        assert "(generated)" in out

    def test_omits_lines_when_nothing_generated(self) -> None:
        """User-supplied artifacts leave both fields None — no path lines shown."""
        r = ModelResult(name="fraud", endpoint="https://fraud.example")

        out = _summary_text([r])

        assert "(generated)" not in out
        assert "Config       :" not in out
        assert "Request      :" not in out
