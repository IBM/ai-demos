"""Unit tests for rhoai.platform.trustyai_client."""

import json
from unittest.mock import MagicMock, patch

import pytest

from rhoai.platform import trustyai_client

_ROUTE = "https://trustyai.apps.example.com"
_TOKEN = "test-token"
_MODEL = "fraud-detection"


def _mock_response(
    json_data: object,
    status_code: int = 200,
    text: str = "",
    content: bytes = b"{}",
    content_type: str = "application/json",
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok          = status_code < 400
    resp.text        = text
    resp.content     = content
    resp.headers     = {"Content-Type": content_type}
    resp.json.return_value = json_data
    return resp


# ---------------------------------------------------------------------------
# _get retry behaviour
# ---------------------------------------------------------------------------

class TestGetRetry:
    """_get must retry when TrustyAI transiently returns an empty body."""

    def test_retries_on_empty_body_then_succeeds(self) -> None:
        """First call returns empty body; second returns JSON — must succeed."""
        empty = _mock_response(None, content=b"")
        ok    = _mock_response({_MODEL: {}})
        with patch("rhoai.platform.trustyai_client.requests.get", side_effect=[empty, ok]), \
             patch("rhoai.platform.trustyai_client.time.sleep"):
            result = trustyai_client.get_model_info(_ROUTE, _TOKEN, _MODEL)
        assert result == {}

    def test_raises_after_all_retries_empty(self) -> None:
        """All retries return empty body — must raise RuntimeError, not JSONDecodeError."""
        empty = _mock_response(None, content=b"")
        with patch("rhoai.platform.trustyai_client.requests.get", return_value=empty), \
             patch("rhoai.platform.trustyai_client.time.sleep"):
            with pytest.raises(RuntimeError, match="unparseable response"):
                trustyai_client.get_model_info(_ROUTE, _TOKEN, _MODEL)

    def test_retries_on_non_json_body_then_succeeds(self) -> None:
        """First call returns a non-JSON body (e.g. proxy error page); second returns JSON."""
        non_json = MagicMock()
        non_json.status_code = 200
        non_json.ok          = True
        non_json.content     = b"<html>Bad Gateway</html>"
        non_json.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        ok = _mock_response({_MODEL: {}})
        with patch("rhoai.platform.trustyai_client.requests.get", side_effect=[non_json, ok]), \
             patch("rhoai.platform.trustyai_client.time.sleep"):
            result = trustyai_client.get_model_info(_ROUTE, _TOKEN, _MODEL)
        assert result == {}

    def test_raises_after_all_retries_non_json(self) -> None:
        """All retries return non-JSON body — must raise RuntimeError, not JSONDecodeError."""
        non_json = MagicMock()
        non_json.status_code = 200
        non_json.ok          = True
        non_json.content     = b"<html>Bad Gateway</html>"
        non_json.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        with patch("rhoai.platform.trustyai_client.requests.get", return_value=non_json), \
             patch("rhoai.platform.trustyai_client.time.sleep"):
            with pytest.raises(RuntimeError, match="unparseable response"):
                trustyai_client.get_model_info(_ROUTE, _TOKEN, _MODEL)

    def test_no_sleep_on_success(self) -> None:
        """No sleep when the first call succeeds."""
        ok = _mock_response({_MODEL: {}})
        with patch("rhoai.platform.trustyai_client.requests.get", return_value=ok), \
             patch("rhoai.platform.trustyai_client.time.sleep") as mock_sleep:
            trustyai_client.get_model_info(_ROUTE, _TOKEN, _MODEL)
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# get_model_info
# ---------------------------------------------------------------------------

class TestGetModelInfo:
    def test_returns_model_entry(self) -> None:
        info_response = {_MODEL: {"name": _MODEL, "data": {}}, "other": {}}
        with patch("rhoai.platform.trustyai_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response(info_response)
            result = trustyai_client.get_model_info(_ROUTE, _TOKEN, _MODEL)
        assert result == {"name": _MODEL, "data": {}}

    def test_calls_correct_url(self) -> None:
        with patch("rhoai.platform.trustyai_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response({_MODEL: {}})
            trustyai_client.get_model_info(_ROUTE, _TOKEN, _MODEL)
        url = mock_get.call_args[0][0]
        assert url == f"{_ROUTE}/info"

    def test_raises_key_error_when_model_not_found(self) -> None:
        with patch("rhoai.platform.trustyai_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response({"other-model": {}})
            with pytest.raises(KeyError, match=_MODEL):
                trustyai_client.get_model_info(_ROUTE, _TOKEN, _MODEL)

    def test_sends_bearer_token(self) -> None:
        with patch("rhoai.platform.trustyai_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response({_MODEL: {}})
            trustyai_client.get_model_info(_ROUTE, _TOKEN, _MODEL)
        headers = mock_get.call_args[1]["headers"]
        assert headers["Authorization"] == f"Bearer {_TOKEN}"


# ---------------------------------------------------------------------------
# apply_name_mapping
# ---------------------------------------------------------------------------

# /info response where nameMapping is absent — mapping not yet applied.
# Confirmed live-cluster structure: dict KEY is the display name (== internal name
# before any mapping), "name" value is the original internal name.
_INFO_NOT_MAPPED = {
    _MODEL: {
        "data": {
            "inputSchema":  {"items": {"input-0":  {"name": "input-0",  "type": "DOUBLE", "columnIndex": 0}}},
            "outputSchema": {"items": {"output-0": {"name": "output-0", "type": "DOUBLE", "columnIndex": 0}}},
        }
    }
}

# /info response where mapping is already applied.
# TrustyAI populates nameMapping only after a successful POST /info/names.
_INFO_ALREADY_MAPPED = {
    _MODEL: {
        "data": {
            "inputSchema": {
                "items": {"age":   {"name": "input-0",  "type": "DOUBLE", "columnIndex": 0}},
                "nameMapping": {"input-0": "age"},
            },
            "outputSchema": {
                "items": {"score": {"name": "output-0", "type": "DOUBLE", "columnIndex": 0}},
                "nameMapping": {"output-0": "score"},
            },
        }
    }
}


class TestApplyNameMapping:
    def test_posts_to_info_names(self) -> None:
        with patch("rhoai.platform.trustyai_client.requests.get") as mock_get, \
             patch("rhoai.platform.trustyai_client.requests.post") as mock_post:
            mock_get.return_value  = _mock_response(_INFO_NOT_MAPPED)
            mock_post.return_value = _mock_response({})
            trustyai_client.apply_name_mapping(
                _ROUTE, _TOKEN, _MODEL,
                input_mapping={"input-0": "age"},
                output_mapping={"output-0": "fraud_score"},
            )
        url = mock_post.call_args[0][0]
        assert url.endswith("/info/names")

    def test_payload_shape(self) -> None:
        with patch("rhoai.platform.trustyai_client.requests.get") as mock_get, \
             patch("rhoai.platform.trustyai_client.requests.post") as mock_post:
            mock_get.return_value  = _mock_response(_INFO_NOT_MAPPED)
            mock_post.return_value = _mock_response({})
            trustyai_client.apply_name_mapping(
                _ROUTE, _TOKEN, _MODEL,
                input_mapping={"input-0": "age"},
                output_mapping={"output-0": "score"},
            )
        body = mock_post.call_args[1]["json"]
        assert body["modelId"] == _MODEL
        assert body["inputMapping"] == {"input-0": "age"}
        assert body["outputMapping"] == {"output-0": "score"}

    def test_204_no_content_does_not_raise(self) -> None:
        """TrustyAI /info/names returns 204 No Content — must not crash on empty body."""
        with patch("rhoai.platform.trustyai_client.requests.get") as mock_get, \
             patch("rhoai.platform.trustyai_client.requests.post") as mock_post:
            mock_get.return_value  = _mock_response(_INFO_NOT_MAPPED)
            mock_post.return_value = _mock_response(None, status_code=204, content=b"")
            trustyai_client.apply_name_mapping(
                _ROUTE, _TOKEN, _MODEL,
                input_mapping={"input-0": "age"},
                output_mapping={"output-0": "score"},
            )

    def test_200_non_json_body_does_not_raise(self) -> None:
        """TrustyAI /info/names returns 200 with whitespace-only body — must not crash."""
        with patch("rhoai.platform.trustyai_client.requests.get") as mock_get, \
             patch("rhoai.platform.trustyai_client.requests.post") as mock_post:
            mock_get.return_value  = _mock_response(_INFO_NOT_MAPPED)
            mock_post.return_value = _mock_response(None, content=b" ", content_type="")
            trustyai_client.apply_name_mapping(
                _ROUTE, _TOKEN, _MODEL,
                input_mapping={"input-0": "age"},
                output_mapping={"output-0": "score"},
            )

    def test_200_json_content_type_but_unparseable_body_does_not_raise(self) -> None:
        """TrustyAI /info/names returns 200 application/json but with a body that
        cannot be parsed as JSON (observed on the second model when TrustyAI is
        still writing state from the first mapping).  Must not crash."""
        bad_json_resp = MagicMock()
        bad_json_resp.status_code = 200
        bad_json_resp.ok          = True
        bad_json_resp.content     = b"Mapping applied"
        bad_json_resp.headers     = {"Content-Type": "application/json"}
        bad_json_resp.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        with patch("rhoai.platform.trustyai_client.requests.get") as mock_get, \
             patch("rhoai.platform.trustyai_client.requests.post") as mock_post:
            mock_get.return_value  = _mock_response(_INFO_NOT_MAPPED)
            mock_post.return_value = bad_json_resp
            trustyai_client.apply_name_mapping(
                _ROUTE, _TOKEN, _MODEL,
                input_mapping={"input-0": "age"},
                output_mapping={"output-0": "score"},
            )

    def test_skips_when_already_applied(self) -> None:
        """Must not POST to /info/names when the mapping is already in place."""
        with patch("rhoai.platform.trustyai_client.requests.get") as mock_get, \
             patch("rhoai.platform.trustyai_client.requests.post") as mock_post:
            mock_get.return_value = _mock_response(_INFO_ALREADY_MAPPED)
            trustyai_client.apply_name_mapping(
                _ROUTE, _TOKEN, _MODEL,
                input_mapping={"input-0": "age"},
                output_mapping={"output-0": "score"},
            )
        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# compute_spd
# ---------------------------------------------------------------------------

class TestComputeSpd:
    _spd_response = {
        "value": 0.05,
        "id": "spd-123",
        "thresholds": {"lowerBound": -0.1, "upperBound": 0.1},
        "specificDefinition": "…",
    }

    def test_returns_response_dict(self) -> None:
        with patch("rhoai.platform.trustyai_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response(self._spd_response)
            result = trustyai_client.compute_spd(
                _ROUTE, _TOKEN, _MODEL,
                protected_attribute="gender", privileged_value=1.0,
                unprivileged_value=0.0, outcome_name="fraud_score",
                favorable_outcome=0.0,
            )
        assert result["value"] == 0.05

    def test_posts_to_spd_endpoint(self) -> None:
        with patch("rhoai.platform.trustyai_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response(self._spd_response)
            trustyai_client.compute_spd(
                _ROUTE, _TOKEN, _MODEL,
                protected_attribute="gender", privileged_value=1.0,
                unprivileged_value=0.0, outcome_name="fraud_score",
                favorable_outcome=0.0,
            )
        url = mock_post.call_args[0][0]
        assert url.endswith("/metrics/group/fairness/spd")

    def test_payload_contains_protected_attribute(self) -> None:
        with patch("rhoai.platform.trustyai_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response(self._spd_response)
            trustyai_client.compute_spd(
                _ROUTE, _TOKEN, _MODEL,
                protected_attribute="gender", privileged_value=1.0,
                unprivileged_value=0.0, outcome_name="fraud_score",
                favorable_outcome=0.0, batch_size=1000,
            )
        body = mock_post.call_args[1]["json"]
        assert body["protectedAttribute"] == "gender"
        assert body["privilegedAttribute"] == {"type": "DOUBLE", "value": 1.0}
        assert body["batchSize"] == 1000


# ---------------------------------------------------------------------------
# schedule_spd
# ---------------------------------------------------------------------------

class TestScheduleSpd:
    def test_returns_request_id(self) -> None:
        """Accepts both "requestId" (newer TrustyAI) and "id" (older TrustyAI)."""
        with patch("rhoai.platform.trustyai_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({"requestId": "req-abc"})
            result = trustyai_client.schedule_spd(
                _ROUTE, _TOKEN, _MODEL,
                protected_attribute="gender", privileged_value=1.0,
                unprivileged_value=0.0, outcome_name="fraud_score",
                favorable_outcome=0.0,
            )
        assert result == "req-abc"

    def test_returns_request_id_legacy_key(self) -> None:
        """Falls back to "id" when "requestId" is absent."""
        with patch("rhoai.platform.trustyai_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({"id": "req-abc"})
            result = trustyai_client.schedule_spd(
                _ROUTE, _TOKEN, _MODEL,
                protected_attribute="gender", privileged_value=1.0,
                unprivileged_value=0.0, outcome_name="fraud_score",
                favorable_outcome=0.0,
            )
        assert result == "req-abc"

    def test_posts_to_spd_request_endpoint(self) -> None:
        with patch("rhoai.platform.trustyai_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({"id": "req-abc"})
            trustyai_client.schedule_spd(
                _ROUTE, _TOKEN, _MODEL,
                protected_attribute="gender", privileged_value=1.0,
                unprivileged_value=0.0, outcome_name="fraud_score",
                favorable_outcome=0.0,
            )
        url = mock_post.call_args[0][0]
        assert url.endswith("/metrics/group/fairness/spd/request")


# ---------------------------------------------------------------------------
# schedule_identity
# ---------------------------------------------------------------------------

class TestScheduleIdentity:
    def test_returns_request_id(self) -> None:
        """Accepts both "requestId" (newer TrustyAI) and "id" (older TrustyAI)."""
        with patch("rhoai.platform.trustyai_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({"requestId": "id-xyz"})
            result = trustyai_client.schedule_identity(
                _ROUTE, _TOKEN, _MODEL, column_name="gender"
            )
        assert result == "id-xyz"

    def test_returns_request_id_legacy_key(self) -> None:
        """Falls back to "id" when "requestId" is absent."""
        with patch("rhoai.platform.trustyai_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({"id": "id-xyz"})
            result = trustyai_client.schedule_identity(
                _ROUTE, _TOKEN, _MODEL, column_name="gender"
            )
        assert result == "id-xyz"

    def test_posts_to_identity_request_endpoint(self) -> None:
        with patch("rhoai.platform.trustyai_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({"id": "id-xyz"})
            trustyai_client.schedule_identity(
                _ROUTE, _TOKEN, _MODEL, column_name="gender", batch_size=2000
            )
        url = mock_post.call_args[0][0]
        assert url.endswith("/metrics/identity/request")

    def test_payload_shape(self) -> None:
        with patch("rhoai.platform.trustyai_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({"id": "id-xyz"})
            trustyai_client.schedule_identity(
                _ROUTE, _TOKEN, _MODEL, column_name="gender", batch_size=2000
            )
        body = mock_post.call_args[1]["json"]
        assert body["modelId"] == _MODEL
        assert body["columnName"] == "gender"
        assert body["batchSize"] == 2000


# ---------------------------------------------------------------------------
# Error reporting (_raise_with_body via _post and _get)
# ---------------------------------------------------------------------------

class TestErrorReporting:
    """Verify that non-2xx responses include the TrustyAI body in the message."""

    def test_post_400_includes_body(self) -> None:
        error_body = '{"error": "Unknown protected attribute: gender"}'
        with patch("rhoai.platform.trustyai_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({}, status_code=400, text=error_body)
            with pytest.raises(RuntimeError, match="HTTP 400"):
                trustyai_client.compute_spd(
                    _ROUTE, _TOKEN, _MODEL,
                    protected_attribute="gender", privileged_value=1.0,
                    unprivileged_value=0.0, outcome_name="score",
                    favorable_outcome=0.0,
                )

    def test_post_error_message_includes_trustyai_body(self) -> None:
        error_body = "Unknown protected attribute: foobar"
        with patch("rhoai.platform.trustyai_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({}, status_code=400, text=error_body)
            with pytest.raises(RuntimeError, match="foobar"):
                trustyai_client.compute_spd(
                    _ROUTE, _TOKEN, _MODEL,
                    protected_attribute="foobar", privileged_value=1.0,
                    unprivileged_value=0.0, outcome_name="score",
                    favorable_outcome=0.0,
                )

    def test_get_400_includes_body(self) -> None:
        error_body = "model not found"
        with patch("rhoai.platform.trustyai_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response({}, status_code=404, text=error_body)
            with pytest.raises(RuntimeError, match="HTTP 404"):
                trustyai_client.get_model_info(_ROUTE, _TOKEN, _MODEL)

    def test_get_error_message_includes_trustyai_body(self) -> None:
        error_body = "service temporarily unavailable"
        with patch("rhoai.platform.trustyai_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response({}, status_code=503, text=error_body)
            with pytest.raises(RuntimeError, match="temporarily unavailable"):
                trustyai_client.get_model_info(_ROUTE, _TOKEN, _MODEL)

    def test_empty_body_shows_placeholder(self) -> None:
        with patch("rhoai.platform.trustyai_client.requests.post") as mock_post:
            mock_post.return_value = _mock_response({}, status_code=500, text="")
            with pytest.raises(RuntimeError, match="<empty body>"):
                trustyai_client.schedule_identity(_ROUTE, _TOKEN, _MODEL, column_name="x")
