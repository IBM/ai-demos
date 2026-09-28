"""Unit tests for rhoai.ocp.wait."""


import pytest

from rhoai.ocp import wait


class TestWaitUntil:
    def test_succeeds_immediately(self) -> None:
        wait.wait_until(lambda: True, "instant condition", timeout=5)

    def test_succeeds_after_retries(self) -> None:
        calls = {"n": 0}

        def flaky() -> bool:
            calls["n"] += 1
            return calls["n"] >= 3

        wait.wait_until(flaky, "flaky condition", timeout=60, interval=0)
        assert calls["n"] == 3

    def test_raises_timeout_error(self) -> None:
        with pytest.raises(TimeoutError, match="always false"):
            wait.wait_until(lambda: False, "always false", timeout=0, interval=0)

    def test_timeout_error_message_contains_description(self) -> None:
        with pytest.raises(TimeoutError, match="my description"):
            wait.wait_until(lambda: False, "my description", timeout=0, interval=0)


class TestWaitUntilReady:
    def test_delegates_to_is_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("rhoai.ocp.resources.is_ready", lambda *_: True)
        wait.wait_until_ready("Pod", "my-pod", "ns", timeout=10)

    def test_raises_when_never_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("rhoai.ocp.resources.is_ready", lambda *_: False)
        with pytest.raises(TimeoutError):
            wait.wait_until_ready("Pod", "stuck", "ns", timeout=0, interval=0)


class TestWaitUntilDeleted:
    def test_delegates_to_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("rhoai.ocp.resources.exists", lambda *_: False)
        wait.wait_until_deleted("Pod", "gone", "ns", timeout=10)

    def test_raises_when_still_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("rhoai.ocp.resources.exists", lambda *_: True)
        with pytest.raises(TimeoutError):
            wait.wait_until_deleted("Pod", "stubborn", "ns", timeout=0, interval=0)
