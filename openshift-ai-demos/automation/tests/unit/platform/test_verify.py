"""Unit tests for rhoai.platform.verify and rhoai.utils.errors."""

from unittest.mock import MagicMock

import pytest

from rhoai.platform.verify import _run, verify_platform
from rhoai.utils.errors import friendly_error

# ---------------------------------------------------------------------------
# friendly_error
# ---------------------------------------------------------------------------

class TestFriendlyError:
    def test_kubernetes_api_exception_uses_reason_and_status(self) -> None:
        exc = Exception("ignored")
        exc.reason = "Unauthorized"  # type: ignore[attr-defined]
        exc.status = 401             # type: ignore[attr-defined]
        assert friendly_error(exc) == "Unauthorized (401)"

    def test_reason_only_when_no_status(self) -> None:
        exc = Exception("ignored")
        exc.reason = "Forbidden"     # type: ignore[attr-defined]
        assert friendly_error(exc) == "Forbidden"

    def test_first_line_only_for_plain_exceptions(self) -> None:
        exc = RuntimeError("line one\nline two\nline three")
        assert friendly_error(exc) == "line one"

    def test_plain_single_line_exception(self) -> None:
        exc = RuntimeError("something went wrong")
        assert friendly_error(exc) == "something went wrong"

    def test_empty_reason_falls_back_to_str(self) -> None:
        exc = RuntimeError("fallback message")
        exc.reason = ""              # type: ignore[attr-defined]
        assert friendly_error(exc) == "fallback message"


# ---------------------------------------------------------------------------
# _run
# ---------------------------------------------------------------------------

class TestRun:
    def test_passes_when_no_exception(self) -> None:
        result = _run("MyCheck", lambda: None)
        assert result.passed is True
        assert result.name == "MyCheck"
        assert result.message == ""

    def test_fails_when_exception_raised(self) -> None:
        result = _run("MyCheck", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert result.passed is False
        assert result.name == "MyCheck"
        assert result.message == "boom"

    def test_message_uses_friendly_error(self) -> None:
        exc_holder: list[Exception] = []

        def raiser() -> None:
            e = Exception("ignored body")
            e.reason = "Unauthorized"   # type: ignore[attr-defined]
            e.status = 401              # type: ignore[attr-defined]
            exc_holder.append(e)
            raise e

        result = _run("Cluster", raiser)
        assert result.message == "Unauthorized (401)"


# ---------------------------------------------------------------------------
# verify_platform
# ---------------------------------------------------------------------------

class TestVerifyPlatform:
    def _config(self) -> dict:
        return {
            "operator": {"name": "rhods-operator", "namespace": "redhat-ods-operator"},
            "dsc":      {"name": "default-dsc", "dsci_name": "default-dsci"},
        }

    def test_returns_three_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("rhoai.platform.verify.operators", MagicMock())
        monkeypatch.setattr("rhoai.platform.verify.dsc", MagicMock())
        results = verify_platform(self._config())
        assert len(results) == 3

    def test_all_pass_when_no_exceptions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("rhoai.platform.verify.operators", MagicMock())
        monkeypatch.setattr("rhoai.platform.verify.dsc", MagicMock())
        results = verify_platform(self._config())
        assert all(r.passed for r in results)

    def test_does_not_raise_when_checks_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ops_mock = MagicMock()
        ops_mock.verify.side_effect = RuntimeError("not ready")
        monkeypatch.setattr("rhoai.platform.verify.operators", ops_mock)
        monkeypatch.setattr("rhoai.platform.verify.dsc", MagicMock())
        # must NOT raise — callers inspect results themselves
        results = verify_platform(self._config())
        assert any(not r.passed for r in results)

    def test_failed_result_carries_friendly_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ops_mock = MagicMock()
        exc = Exception("body")
        exc.reason = "Unauthorized"   # type: ignore[attr-defined]
        exc.status = 401              # type: ignore[attr-defined]
        ops_mock.verify.side_effect = exc
        monkeypatch.setattr("rhoai.platform.verify.operators", ops_mock)
        monkeypatch.setattr("rhoai.platform.verify.dsc", MagicMock())
        results = verify_platform(self._config())
        operator_result = next(r for r in results if r.name == "RHOAI Operator")
        assert operator_result.message == "Unauthorized (401)"

    def test_result_names_are_correct(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("rhoai.platform.verify.operators", MagicMock())
        monkeypatch.setattr("rhoai.platform.verify.dsc", MagicMock())
        results = verify_platform(self._config())
        names = [r.name for r in results]
        assert "RHOAI Operator"     in names
        assert "DSCInitialization"  in names
        assert "DataScienceCluster" in names
