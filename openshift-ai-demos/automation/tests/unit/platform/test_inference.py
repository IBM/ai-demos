"""Unit tests for rhoai.platform.inference."""

from unittest.mock import MagicMock

import pytest

from rhoai.platform import inference


@pytest.fixture(autouse=True)
def patch_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rhoai.platform.inference.resources", MagicMock())
    monkeypatch.setattr("rhoai.platform.inference.wait", MagicMock())


class TestGetInferenceUrl:
    def test_returns_status_url_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.status.return_value = {"url": "https://model.example.com"}
        monkeypatch.setattr("rhoai.platform.inference.resources", mock_resources)
        assert inference.get_inference_url("qwen", "ns") == "https://model.example.com"

    def test_falls_back_to_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.status.return_value = {}
        mock_resources.exists.return_value = True
        mock_resources.get.return_value = {
            "spec": {"host": "qwen.apps.example.com", "tls": {"termination": "edge"}}
        }
        monkeypatch.setattr("rhoai.platform.inference.resources", mock_resources)
        url = inference.get_inference_url("qwen", "ns")
        assert url == "https://qwen.apps.example.com"

    def test_raises_when_no_url_and_no_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.status.return_value = {}
        mock_resources.exists.return_value = False
        monkeypatch.setattr("rhoai.platform.inference.resources", mock_resources)
        with pytest.raises(RuntimeError, match="no URL"):
            inference.get_inference_url("qwen", "ns")


class TestVerify:
    # --- namespace-wide (name=None) ---
    def test_passes_when_all_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.list_resources.return_value = [{"metadata": {"name": "qwen"}}]
        mock_resources.is_ready.return_value = True
        monkeypatch.setattr("rhoai.platform.inference.resources", mock_resources)
        inference.verify("ns")  # should not raise

    def test_raises_when_service_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.list_resources.return_value = [{"metadata": {"name": "qwen"}}]
        mock_resources.is_ready.return_value = False
        monkeypatch.setattr("rhoai.platform.inference.resources", mock_resources)
        with pytest.raises(RuntimeError, match="not ready"):
            inference.verify("ns")

    def test_passes_with_no_services(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.list_resources.return_value = []
        monkeypatch.setattr("rhoai.platform.inference.resources", mock_resources)
        inference.verify("ns")  # no services = nothing to fail

    # --- named ISVC path ---
    def test_named_passes_when_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.is_ready.return_value = True
        monkeypatch.setattr("rhoai.platform.inference.resources", mock_resources)
        inference.verify("ns", name="qwen")  # should not raise
        mock_resources.is_ready.assert_called_once_with("InferenceService", "qwen", "ns")

    def test_named_raises_when_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.is_ready.return_value = False
        monkeypatch.setattr("rhoai.platform.inference.resources", mock_resources)
        with pytest.raises(RuntimeError, match="'qwen' is not ready"):
            inference.verify("ns", name="qwen")

    def test_named_does_not_call_list_resources(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.is_ready.return_value = True
        monkeypatch.setattr("rhoai.platform.inference.resources", mock_resources)
        inference.verify("ns", name="qwen")
        mock_resources.list_resources.assert_not_called()


# ---------------------------------------------------------------------------
# verify_triton_inference
# ---------------------------------------------------------------------------

class TestVerifyTritonInference:
    """Tests for verify_triton_inference and its private helpers.

    _http_post is patched at the module level so no real HTTP calls are made.
    get_inference_url is patched to return a fixed base URL.
    """

    _BASE_URL  = "https://fraud-detection.apps.example.com"
    _MODEL     = "fraud-detection"
    _GOOD_BODY = {"model_name": "fraud-detection", "outputs": [{"name": "output", "data": [0.1]}]}
    _GOOD_RESP = (_GOOD_BODY, 0.05, 200)   # (response_dict, elapsed, status)

    def _patch(self, monkeypatch: pytest.MonkeyPatch, post_return=None, post_side_effect=None):
        monkeypatch.setattr(
            "rhoai.platform.inference.get_inference_url",
            lambda *_: self._BASE_URL,
        )
        post_mock = MagicMock(return_value=post_return, side_effect=post_side_effect)
        monkeypatch.setattr("rhoai.platform.inference._http_post", post_mock)
        return post_mock

    def _sample_request(self, tmp_path):
        import json as _json
        p = tmp_path / "sample.json"
        p.write_text(_json.dumps({"inputs": [{"name": "dense_input", "data": [0.1] * 10}]}))
        return p

    # --- happy path ---

    def test_passes_on_valid_response(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        self._patch(monkeypatch, post_return=self._GOOD_RESP)
        payload, response, curl_cmd = inference.verify_triton_inference(
            "fraud-detection", "ns", self._MODEL, self._sample_request(tmp_path)
        )
        assert payload["inputs"][0]["name"] == "dense_input"
        assert response == self._GOOD_BODY
        assert "curl -sk -X POST" in curl_cmd

    def test_posts_to_correct_url(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        post_mock = self._patch(monkeypatch, post_return=self._GOOD_RESP)
        inference.verify_triton_inference(
            "fraud-detection", "ns", self._MODEL, self._sample_request(tmp_path)
        )
        called_url = post_mock.call_args[0][0]
        assert called_url == f"{self._BASE_URL}/v2/models/{self._MODEL}/infer"

    # --- response validation ---

    def test_raises_on_wrong_model_name_in_response(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        bad_resp = ({"model_name": "wrong-model", "outputs": [{}]}, 0.05, 200)
        self._patch(monkeypatch, post_return=bad_resp)
        with pytest.raises(RuntimeError, match="Unexpected model_name"):
            inference.verify_triton_inference(
                "fraud-detection", "ns", self._MODEL, self._sample_request(tmp_path)
            )

    def test_raises_when_outputs_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        bad_resp = ({"model_name": "fraud-detection", "outputs": []}, 0.05, 200)
        self._patch(monkeypatch, post_return=bad_resp)
        with pytest.raises(RuntimeError, match="no outputs"):
            inference.verify_triton_inference(
                "fraud-detection", "ns", self._MODEL, self._sample_request(tmp_path)
            )

    def test_raises_when_outputs_key_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        bad_resp = ({"model_name": "fraud-detection"}, 0.05, 200)
        self._patch(monkeypatch, post_return=bad_resp)
        with pytest.raises(RuntimeError, match="no outputs"):
            inference.verify_triton_inference(
                "fraud-detection", "ns", self._MODEL, self._sample_request(tmp_path)
            )

    # --- retry on connection error ---

    def test_retries_once_on_connection_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setattr("rhoai.platform.inference.time.sleep", MagicMock())
        monkeypatch.setattr(
            "rhoai.platform.inference.get_inference_url",
            lambda *_: self._BASE_URL,
        )
        # First call raises _ConnectionError; second call succeeds (returns a 3-tuple).
        post_mock = MagicMock(
            side_effect=[inference._ConnectionError("timeout"), self._GOOD_RESP]
        )
        monkeypatch.setattr("rhoai.platform.inference._http_post", post_mock)
        inference.verify_triton_inference(
            "fraud-detection", "ns", self._MODEL, self._sample_request(tmp_path)
        )
        assert post_mock.call_count == 2

    def test_raises_endpoint_unreachable_after_both_attempts_fail(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """When both connection attempts fail, EndpointUnreachable is raised.

        The caller (deploy/verify) is responsible for handling it gracefully.
        This keeps the platform layer free of progress/UI concerns.
        """
        monkeypatch.setattr("rhoai.platform.inference.time.sleep", MagicMock())
        monkeypatch.setattr(
            "rhoai.platform.inference.get_inference_url",
            lambda *_: self._BASE_URL,
        )
        post_mock = MagicMock(
            side_effect=inference._ConnectionError("still down")
        )
        monkeypatch.setattr("rhoai.platform.inference._http_post", post_mock)
        with pytest.raises(inference.EndpointUnreachable) as exc_info:
            inference.verify_triton_inference(
                "fraud-detection", "ns", self._MODEL, self._sample_request(tmp_path)
            )
        assert post_mock.call_count == 2
        assert self._BASE_URL in exc_info.value.infer_url
        assert "curl" in exc_info.value.curl_cmd
        assert '"inputs"' in exc_info.value.curl_cmd


# ---------------------------------------------------------------------------
# _quiet_urllib3
# ---------------------------------------------------------------------------

class TestQuietUrllib3:
    def test_suppresses_urllib3_warnings_within_block(self) -> None:
        import logging
        logger = logging.getLogger("urllib3")
        logger.setLevel(logging.WARNING)
        # Capture the level as _quiet_urllib3 will see it (after we set it).
        level_before_block = logger.level
        with inference._quiet_urllib3():
            assert logging.getLogger("urllib3").level == logging.ERROR
        assert logging.getLogger("urllib3").level == level_before_block

    def test_restores_level_on_exception(self) -> None:
        import logging
        logger = logging.getLogger("urllib3")
        logger.setLevel(logging.WARNING)
        try:
            with inference._quiet_urllib3():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert logging.getLogger("urllib3").level == logging.WARNING


# ---------------------------------------------------------------------------
# _http_post
# ---------------------------------------------------------------------------

class TestHttpPost:
    """Unit tests for the _http_post helper (urllib3 pool mocked)."""

    def _patch_pool(self, monkeypatch: pytest.MonkeyPatch, status: int, data: bytes):
        resp_mock = MagicMock()
        resp_mock.status = status
        resp_mock.data   = data
        pool_mock = MagicMock()
        pool_mock.request.return_value = resp_mock
        monkeypatch.setattr("rhoai.platform.inference.urllib3.PoolManager", lambda **_: pool_mock)
        return pool_mock

    def test_returns_decoded_json_on_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import json as _json
        body = _json.dumps({"model_name": "x", "outputs": []}).encode()
        self._patch_pool(monkeypatch, 200, body)
        result, elapsed, status = inference._http_post("https://host/infer", {"inputs": []})
        assert result["model_name"] == "x"
        assert status == 200
        assert elapsed >= 0.0

    def test_raises_runtime_error_on_non_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_pool(monkeypatch, 503, b"Service Unavailable")
        with pytest.raises(RuntimeError, match="HTTP 503"):
            inference._http_post("https://host/infer", {})

    def test_raises_connection_error_on_urllib3_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import urllib3 as _urllib3
        pool_mock = MagicMock()
        pool_mock.request.side_effect = _urllib3.exceptions.HTTPError("connect failed")
        monkeypatch.setattr("rhoai.platform.inference.urllib3.PoolManager", lambda **_: pool_mock)
        with pytest.raises(inference._ConnectionError, match="connect failed"):
            inference._http_post("https://host/infer", {})

    def test_raises_on_invalid_json_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_pool(monkeypatch, 200, b"not-json{{")
        with pytest.raises(RuntimeError, match="not valid JSON"):
            inference._http_post("https://host/infer", {})


# ---------------------------------------------------------------------------
# delete_inference_service / delete_serving_runtime
# ---------------------------------------------------------------------------

class TestDeleteInferenceService:
    def test_deletes_resource(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.inference.resources", mock_resources)
        inference.delete_inference_service("fraud-detection", "ns")
        mock_resources.delete_manifest.assert_called_once_with(
            "InferenceService", "fraud-detection", "ns"
        )

    def test_does_not_wait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.inference.wait", mock_wait)
        inference.delete_inference_service("fraud-detection", "ns")
        mock_wait.wait_until_deleted.assert_not_called()
        mock_wait.wait_until.assert_not_called()


class TestDeleteServingRuntime:
    def test_deletes_resource(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        monkeypatch.setattr("rhoai.platform.inference.resources", mock_resources)
        inference.delete_serving_runtime("triton-fraud-detection", "ns")
        mock_resources.delete_manifest.assert_called_once_with(
            "ServingRuntime", "triton-fraud-detection", "ns"
        )

    def test_does_not_wait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.inference.wait", mock_wait)
        inference.delete_serving_runtime("triton-fraud-detection", "ns")
        mock_wait.wait_until_deleted.assert_not_called()
        mock_wait.wait_until.assert_not_called()


# ---------------------------------------------------------------------------
# wait_until_inference_services_gone / wait_until_serving_runtimes_gone
# ---------------------------------------------------------------------------

class TestWaitUntilInferenceServicesGone:
    def test_calls_wait_until(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.inference.wait", mock_wait)
        inference.wait_until_inference_services_gone(["alpha", "beta"], "ns")
        mock_wait.wait_until.assert_called_once()

    def test_condition_true_when_all_gone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.exists.return_value = False
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.inference.resources", mock_resources)
        monkeypatch.setattr("rhoai.platform.inference.wait", mock_wait)
        inference.wait_until_inference_services_gone(["alpha", "beta"], "ns")
        condition_fn = mock_wait.wait_until.call_args[0][0]
        assert condition_fn() is True

    def test_condition_false_when_one_remains(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.exists.side_effect = lambda kind, name, ns: name == "alpha"
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.inference.resources", mock_resources)
        monkeypatch.setattr("rhoai.platform.inference.wait", mock_wait)
        inference.wait_until_inference_services_gone(["alpha", "beta"], "ns")
        condition_fn = mock_wait.wait_until.call_args[0][0]
        assert condition_fn() is False

    def test_passes_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.inference.wait", mock_wait)
        inference.wait_until_inference_services_gone(["alpha"], "ns", timeout=60)
        assert mock_wait.wait_until.call_args[1]["timeout"] == 60


class TestWaitUntilServingRuntimesGone:
    def test_calls_wait_until(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.inference.wait", mock_wait)
        inference.wait_until_serving_runtimes_gone(["triton-alpha"], "ns")
        mock_wait.wait_until.assert_called_once()

    def test_condition_true_when_all_gone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.exists.return_value = False
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.inference.resources", mock_resources)
        monkeypatch.setattr("rhoai.platform.inference.wait", mock_wait)
        inference.wait_until_serving_runtimes_gone(["triton-alpha", "triton-beta"], "ns")
        condition_fn = mock_wait.wait_until.call_args[0][0]
        assert condition_fn() is True

    def test_condition_false_when_one_remains(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resources = MagicMock()
        mock_resources.exists.side_effect = lambda kind, name, ns: name == "triton-alpha"
        mock_wait = MagicMock()
        monkeypatch.setattr("rhoai.platform.inference.resources", mock_resources)
        monkeypatch.setattr("rhoai.platform.inference.wait", mock_wait)
        inference.wait_until_serving_runtimes_gone(["triton-alpha", "triton-beta"], "ns")
        condition_fn = mock_wait.wait_until.call_args[0][0]
        assert condition_fn() is False


# ---------------------------------------------------------------------------
# _csv_to_kserve_payload
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _csv_payload_from_path
# ---------------------------------------------------------------------------

class TestCsvPayloadFromPath:
    """Tests for the CSV → KServe v2 conversion helper (replaces _csv_to_kserve_payload)."""

    def _write(self, tmp_path, name: str, content: str):
        p = tmp_path / name
        p.write_text(content)
        return p

    def test_converts_numeric_csv_without_header(self, tmp_path) -> None:
        p = self._write(tmp_path, "data.csv", "1.0,2.0,3.0\n4.0,5.0,6.0\n")
        payload = inference._csv_payload_from_path(p)
        inputs = payload["inputs"][0]
        assert inputs["shape"] == [2, 3]
        assert inputs["data"] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        assert inputs["datatype"] == "FP64"
        assert inputs["name"] == "input"

    def test_custom_input_name_and_datatype(self, tmp_path) -> None:
        p = self._write(tmp_path, "data.csv", "1.0,2.0\n3.0,4.0\n")
        payload = inference._csv_payload_from_path(
            p, input_name="customer_data_input", datatype="FP32"
        )
        inputs = payload["inputs"][0]
        assert inputs["name"] == "customer_data_input"
        assert inputs["datatype"] == "FP32"

    def test_skips_non_numeric_header_row(self, tmp_path) -> None:
        p = self._write(tmp_path, "data.csv", "col_a,col_b\n1.0,2.0\n3.0,4.0\n")
        payload = inference._csv_payload_from_path(p)
        inputs = payload["inputs"][0]
        assert inputs["shape"] == [2, 2]
        assert inputs["data"] == [[1.0, 2.0], [3.0, 4.0]]

    def test_single_row(self, tmp_path) -> None:
        p = self._write(tmp_path, "data.csv", "0.0,202500.0,1.0\n")
        payload = inference._csv_payload_from_path(p)
        assert payload["inputs"][0]["shape"] == [1, 3]

    def test_raises_on_empty_file(self, tmp_path) -> None:
        p = self._write(tmp_path, "data.csv", "")
        with pytest.raises(ValueError, match="no data rows"):
            inference._csv_payload_from_path(p)

    def test_raises_on_header_only(self, tmp_path) -> None:
        p = self._write(tmp_path, "data.csv", "col_a,col_b\n")
        with pytest.raises(ValueError, match="no data rows"):
            inference._csv_payload_from_path(p)

    def test_raises_on_non_numeric_data_row(self, tmp_path) -> None:
        p = self._write(tmp_path, "data.csv", "1.0,2.0\nbad,value\n")
        with pytest.raises(ValueError, match="cannot convert to FP64"):
            inference._csv_payload_from_path(p)

    def test_raises_on_missing_file(self, tmp_path) -> None:
        p = tmp_path / "missing.csv"
        with pytest.raises(FileNotFoundError):
            inference._csv_payload_from_path(p)

    def test_ignores_blank_lines(self, tmp_path) -> None:
        p = self._write(tmp_path, "data.csv", "1.0,2.0\n\n3.0,4.0\n")
        payload = inference._csv_payload_from_path(p)
        assert payload["inputs"][0]["shape"] == [2, 2]


# ---------------------------------------------------------------------------
# _load_request_payload
# ---------------------------------------------------------------------------

class TestLoadRequestPayload:
    """Tests for the format-dispatching _load_request_payload helper."""

    def _write(self, tmp_path, name: str, content: str):
        p = tmp_path / name
        p.write_text(content)
        return p

    def test_loads_json_file(self, tmp_path) -> None:
        import json as _json
        body = {"inputs": [{"name": "x", "shape": [1, 2], "datatype": "FP64", "data": [[1.0, 2.0]]}]}
        p = self._write(tmp_path, "req.json", _json.dumps(body))
        result = inference._load_request_payload(p)
        assert result["inputs"][0]["name"] == "x"

    def test_loads_csv_file(self, tmp_path) -> None:
        p = self._write(tmp_path, "req.csv", "1.0,2.0,3.0\n")
        result = inference._load_request_payload(p)
        assert result["inputs"][0]["data"] == [[1.0, 2.0, 3.0]]

    def test_csv_passes_input_name_and_datatype(self, tmp_path) -> None:
        p = self._write(tmp_path, "req.csv", "1.0,2.0\n")
        result = inference._load_request_payload(
            p, input_name="my_tensor", datatype="FP32"
        )
        assert result["inputs"][0]["name"] == "my_tensor"
        assert result["inputs"][0]["datatype"] == "FP32"

    def test_json_ignores_csv_params(self, tmp_path) -> None:
        import json as _json
        body = {"inputs": [{"name": "orig", "shape": [1, 2], "datatype": "FP64", "data": [[1.0, 2.0]]}]}
        p = self._write(tmp_path, "req.json", _json.dumps(body))
        # input_name / datatype are silently ignored for JSON files
        result = inference._load_request_payload(p, input_name="ignored", datatype="INT64")
        assert result["inputs"][0]["name"] == "orig"
        assert result["inputs"][0]["datatype"] == "FP64"

    def test_raises_on_invalid_json(self, tmp_path) -> None:
        p = self._write(tmp_path, "req.json", "not-json{{")
        with pytest.raises(ValueError, match="not valid JSON"):
            inference._load_request_payload(p)

    def test_raises_on_unsupported_extension(self, tmp_path) -> None:
        p = self._write(tmp_path, "req.txt", "some text")
        with pytest.raises(ValueError, match="Unsupported"):
            inference._load_request_payload(p)

    def test_case_insensitive_extension(self, tmp_path) -> None:
        p = self._write(tmp_path, "req.CSV", "1.0,2.0\n")
        result = inference._load_request_payload(p)
        assert result["inputs"][0]["shape"] == [1, 2]


# ---------------------------------------------------------------------------
# _validate_observation_file — CSV path
# ---------------------------------------------------------------------------

class TestValidateObservationFileCsv:
    """Tests for the CSV branch of _validate_observation_file."""

    def _write(self, tmp_path, name: str, content: str):
        p = tmp_path / name
        p.write_text(content)
        return p

    def test_returns_row_count_for_csv(self, tmp_path) -> None:
        p = self._write(tmp_path, "obs.csv", "1.0,2.0\n3.0,4.0\n5.0,6.0\n")
        assert inference._validate_observation_file(p) == 3

    def test_skips_header_in_row_count(self, tmp_path) -> None:
        p = self._write(tmp_path, "obs.csv", "a,b\n1.0,2.0\n3.0,4.0\n")
        assert inference._validate_observation_file(p) == 2

    def test_raises_on_missing_csv(self, tmp_path) -> None:
        p = tmp_path / "missing.csv"
        with pytest.raises(FileNotFoundError):
            inference._validate_observation_file(p)

    def test_raises_on_empty_csv(self, tmp_path) -> None:
        p = self._write(tmp_path, "empty.csv", "")
        with pytest.raises(ValueError, match="no data rows"):
            inference._validate_observation_file(p)

    def test_passes_input_name_and_datatype_to_payload(self, tmp_path) -> None:
        """Custom csv_config values must appear in the validated payload."""
        p = self._write(tmp_path, "obs.csv", "1.0,2.0\n3.0,4.0\n")
        # _validate_observation_file returns row count; we verify the side-effect
        # indirectly by checking it does not raise and returns the correct count.
        count = inference._validate_observation_file(
            p, input_name="my_tensor", datatype="FP32"
        )
        assert count == 2


# ---------------------------------------------------------------------------
# _read_tensor_schema
# ---------------------------------------------------------------------------

class TestReadTensorSchema:
    """Tests for the JSON-based tensor schema reader."""

    def _schema_file(self, tmp_path, name: str, datatype: str) -> "Path":
        import json as _json
        p = tmp_path / "schema.json"
        p.write_text(_json.dumps({
            "inputs": [{"name": name, "datatype": datatype, "shape": [1, 2], "data": [[0, 0]]}]
        }))
        return p

    def test_reads_name_and_datatype(self, tmp_path) -> None:
        p = self._schema_file(tmp_path, "customer_data_input", "FP64")
        assert inference._read_tensor_schema(p) == ("customer_data_input", "FP64")

    def test_none_returns_defaults(self) -> None:
        assert inference._read_tensor_schema(None) == ("input", "FP64")

    def test_raises_on_missing_file(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            inference._read_tensor_schema(tmp_path / "missing.json")

    def test_raises_on_invalid_json(self, tmp_path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("not-json{{")
        with pytest.raises(ValueError, match="not valid JSON"):
            inference._read_tensor_schema(p)

    def test_raises_when_inputs_missing(self, tmp_path) -> None:
        import json as _json
        p = tmp_path / "bad.json"
        p.write_text(_json.dumps({"no_inputs": []}))
        with pytest.raises(ValueError, match="inputs\\[0\\]"):
            inference._read_tensor_schema(p)

    def test_raises_when_name_missing(self, tmp_path) -> None:
        import json as _json
        p = tmp_path / "bad.json"
        p.write_text(_json.dumps({"inputs": [{"datatype": "FP64"}]}))
        with pytest.raises(ValueError, match="inputs\\[0\\]"):
            inference._read_tensor_schema(p)



# ---------------------------------------------------------------------------
# _validate_observation_file / send_observations — JSON arrays are rejected
# ---------------------------------------------------------------------------

class TestValidateObservationFileJsonArray:
    """JSON observation files must be a KServe v2 object, not a bare array.

    JSON inputs are never transformed — a bare array is not a valid KServe
    v2 request and must be rejected with a clear error.
    """

    def _write(self, tmp_path, name: str, content) -> "Path":
        import json as _json
        p = tmp_path / name
        p.write_text(_json.dumps(content))
        return p

    def test_raises_on_json_array(self, tmp_path) -> None:
        doc = [{"inputs": [{"name": "x", "shape": [1, 1], "datatype": "FP64", "data": [[1.0]]}]}]
        p = self._write(tmp_path, "obs.json", doc)
        with pytest.raises(ValueError, match="not a valid KServe v2 request"):
            inference._validate_observation_file(p)

    def test_raises_on_non_dict_json(self, tmp_path) -> None:
        p = self._write(tmp_path, "obs.json", "just a string")
        with pytest.raises(ValueError, match="not a valid KServe v2 request"):
            inference._validate_observation_file(p)


class TestSendObservationsJsonArray:
    """send_observations must reject a JSON array file with a clear error,
    rather than merging or transforming it."""

    def test_raises_on_json_array(self, tmp_path) -> None:
        doc = [{"inputs": [{"name": "x", "shape": [1, 1], "datatype": "FP64", "data": [[1.0]]}]}]
        p = tmp_path / "obs.json"
        p.write_text(__import__("json").dumps(doc))
        with pytest.raises(ValueError, match="not a valid KServe v2 request"):
            inference.send_observations("model", "ns", [p])

    def test_json_object_still_posts_once_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """A valid JSON object envelope is sent as-is, unchanged."""
        captured: list[dict] = []
        monkeypatch.setattr(
            "rhoai.platform.inference.get_inference_url",
            lambda *_: "https://model.example.com",
        )
        monkeypatch.setattr(
            "rhoai.platform.inference._http_post",
            lambda url, body: (captured.append(body), 0.01, 200)[1:],
        )
        env = {
            "inputs": [{
                "name": "x", "shape": [10, 1],
                "datatype": "FP64",
                "data": [[float(i)] for i in range(10)],
            }]
        }
        p = tmp_path / "obs.json"
        p.write_text(__import__("json").dumps(env))
        total = inference.send_observations("model", "ns", [p])
        assert total == 10
        assert len(captured) == 1
        assert captured[0] == env


# ---------------------------------------------------------------------------
# send_observations — schema_source param
# ---------------------------------------------------------------------------

class TestSendObservationsSchemaSource:
    """Verify that send_observations threads schema_source to _load_request_payload."""

    def _patch(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "rhoai.platform.inference.get_inference_url",
            lambda *_: "https://model.example.com",
        )
        monkeypatch.setattr(
            "rhoai.platform.inference._http_post",
            MagicMock(return_value=({"outputs": []}, 0.01, 200)),
        )

    def _schema_file(self, tmp_path, name: str, datatype: str) -> "Path":
        import json as _json
        p = tmp_path / "schema.json"
        p.write_text(_json.dumps({
            "inputs": [{"name": name, "datatype": datatype, "shape": [1, 2], "data": [[0, 0]]}]
        }))
        return p

    def test_csv_with_no_schema_source_uses_defaults(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        self._patch(monkeypatch)
        p = tmp_path / "obs.csv"
        p.write_text("1.0,2.0\n3.0,4.0\n")
        assert inference.send_observations("model", "ns", [p]) == 2

    def test_csv_uses_name_from_schema_source(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        self._patch(monkeypatch)
        captured: list[dict] = []

        def capture_post(url, body):
            captured.append(body)
            return ({"outputs": []}, 0.01, 200)

        monkeypatch.setattr("rhoai.platform.inference._http_post", capture_post)
        schema = self._schema_file(tmp_path, "customer_data_input", "FP32")
        p = tmp_path / "obs.csv"
        p.write_text("1.0,2.0\n")
        inference.send_observations("model", "ns", [p], schema_source=schema)
        assert captured[0]["inputs"][0]["name"] == "customer_data_input"
        assert captured[0]["inputs"][0]["datatype"] == "FP32"

    def test_none_schema_source_uses_defaults(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        self._patch(monkeypatch)
        captured: list[dict] = []

        def capture_post(url, body):
            captured.append(body)
            return ({"outputs": []}, 0.01, 200)

        monkeypatch.setattr("rhoai.platform.inference._http_post", capture_post)
        p = tmp_path / "obs.csv"
        p.write_text("1.0,2.0\n")
        inference.send_observations("model", "ns", [p], schema_source=None)
        assert captured[0]["inputs"][0]["name"] == "input"
        assert captured[0]["inputs"][0]["datatype"] == "FP64"


class TestSendObservationPayloads:
    """send_observation_payloads posts in-memory batches generated from a dataset."""

    def _env(self, rows: int) -> dict:
        return {
            "inputs": [{
                "name": "x", "shape": [rows, 1], "datatype": "FP64",
                "data": [[float(i)] for i in range(rows)],
            }]
        }

    def _patch(self, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
        captured: list[dict] = []
        monkeypatch.setattr(
            "rhoai.platform.inference.get_inference_url",
            lambda *_: "https://model.example.com",
        )

        def capture_post(url, body):
            captured.append(body)
            return ({"outputs": []}, 0.01, 200)

        monkeypatch.setattr("rhoai.platform.inference._http_post", capture_post)
        return captured

    def test_posts_each_payload_and_sums_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._patch(monkeypatch)
        payloads = [self._env(50), self._env(50), self._env(23)]
        total = inference.send_observation_payloads("model", "ns", payloads)
        assert total == 123
        assert len(captured) == 3
        assert captured == payloads

    def test_raises_on_empty_payloads(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            inference.send_observation_payloads("model", "ns", [])

    def test_validates_before_any_post(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = self._patch(monkeypatch)
        # Second payload has a shape[0]/len(data) mismatch — must fail with no POSTs.
        bad = {"inputs": [{"name": "x", "shape": [5, 1], "datatype": "FP64", "data": [[1.0]]}]}
        with pytest.raises(ValueError, match="does not match len"):
            inference.send_observation_payloads("model", "ns", [self._env(2), bad])
        assert captured == []


# ---------------------------------------------------------------------------
# verify_triton_inference — schema_source param
# ---------------------------------------------------------------------------

class TestVerifyTritonInferenceSchemaSource:
    """Verify that verify_triton_inference threads schema_source for CSV requests."""

    _BASE_URL  = "https://fraud-detection.apps.example.com"
    _GOOD_RESP = (
        {"model_name": "fraud-detection", "outputs": [{"name": "output", "data": [0.1]}]},
        0.05, 200,
    )

    def _patch(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "rhoai.platform.inference.get_inference_url",
            lambda *_: self._BASE_URL,
        )
        return MagicMock(return_value=self._GOOD_RESP)

    def _schema_file(self, tmp_path, name: str, datatype: str) -> "Path":
        import json as _json
        p = tmp_path / "schema.json"
        p.write_text(_json.dumps({
            "inputs": [{"name": name, "datatype": datatype, "shape": [1, 3], "data": [[0, 0, 0]]}]
        }))
        return p

    def test_csv_request_reads_name_from_schema_source(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        post_mock = self._patch(monkeypatch)
        monkeypatch.setattr("rhoai.platform.inference._http_post", post_mock)
        schema = self._schema_file(tmp_path, "customer_data_input", "FP32")
        p = tmp_path / "req.csv"
        p.write_text("0.1,0.2,0.3\n")
        inference.verify_triton_inference(
            "fraud-detection", "ns", "fraud-detection", p, schema_source=schema,
        )
        payload = post_mock.call_args[0][1]
        assert payload["inputs"][0]["name"] == "customer_data_input"
        assert payload["inputs"][0]["datatype"] == "FP32"

    def test_json_request_ignores_schema_source(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import json as _json
        post_mock = self._patch(monkeypatch)
        monkeypatch.setattr("rhoai.platform.inference._http_post", post_mock)
        body = {"inputs": [{"name": "orig", "shape": [1, 3], "datatype": "FP64", "data": [[0.1, 0.2, 0.3]]}]}
        p = tmp_path / "req.json"
        p.write_text(_json.dumps(body))
        schema = self._schema_file(tmp_path, "ignored", "INT64")
        inference.verify_triton_inference(
            "fraud-detection", "ns", "fraud-detection", p, schema_source=schema,
        )
        payload = post_mock.call_args[0][1]
        assert payload["inputs"][0]["name"] == "orig"
        assert payload["inputs"][0]["datatype"] == "FP64"
