"""Unit tests for rhoai.platform.request_generator — iter_requests and generate_batched_requests."""

import json
import textwrap
from pathlib import Path

import pytest

from rhoai.platform.request_generator import (
    derive_input_name_mapping,
    generate_batched_requests,
    humanize_feature_name,
    iter_requests,
    normalize_feature_name,
    read_csv_headers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_pbtxt(tmp_path: Path, name: str = "mymodel", datatype: str = "TYPE_FP64") -> Path:
    """Write a minimal single-input config.pbtxt and return its path."""
    p = tmp_path / "config.pbtxt"
    p.write_text(
        textwrap.dedent(f"""\
            name: "{name}"
            max_batch_size: 0
            input [
              {{
                name: "dense_input"
                data_type: {datatype}
                dims: [ -1 ]
              }}
            ]
        """)
    )
    return p


def _write_pbtxt_multi(tmp_path: Path, name: str = "multi") -> Path:
    """Write a two-input config.pbtxt whose tensor names are 'a' and 'b'."""
    p = tmp_path / "config.pbtxt"
    p.write_text(textwrap.dedent(f"""\
        name: "{name}"
        max_batch_size: 0
        input [
          {{ name: "a" data_type: TYPE_FP32 dims: [ -1 ] }}
          {{ name: "b" data_type: TYPE_FP32 dims: [ -1 ] }}
        ]
    """))
    return p


def _write_pbtxt_multi_types(tmp_path: Path) -> Path:
    """Write a three-input config.pbtxt with mixed types (FP32, INT32, BYTES)."""
    p = tmp_path / "config.pbtxt"
    p.write_text(textwrap.dedent("""\
        name: "mixed"
        max_batch_size: 0
        input [
          { name: "amount"   data_type: TYPE_FP32   dims: [ -1 ] }
          { name: "count"    data_type: TYPE_INT32   dims: [ -1 ] }
          { name: "category" data_type: TYPE_STRING  dims: [ -1 ] }
        ]
    """))
    return p


def _write_csv(tmp_path: Path, content: str, name: str = "data.csv") -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def _write_json(tmp_path: Path, doc: dict | list, name: str = "dataset.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


def _kserve_envelope(
    tensor_name: str = "dense_input",
    n_rows: int = 2,
    n_cols: int = 3,
    datatype: str = "FP64",
) -> dict:
    """Return a minimal valid KServe v2 request envelope."""
    data = [[float(i * n_cols + j) for j in range(n_cols)] for i in range(n_rows)]
    return {
        "inputs": [{
            "name":     tensor_name,
            "shape":    [n_rows, n_cols],
            "datatype": datatype,
            "data":     data,
        }]
    }


# ---------------------------------------------------------------------------
# iter_requests — CSV: basic batching behaviour
# ---------------------------------------------------------------------------

class TestIterRequestsCsvBasic:

    def test_single_row_default_batch_size(self, tmp_path: Path) -> None:
        """batch_size=1 yields one request per CSV data row."""
        pbtxt = _write_pbtxt(tmp_path)
        csv   = _write_csv(tmp_path, "col_a,col_b,col_c\n1.0,2.0,3.0\n")
        batches = list(iter_requests(pbtxt, csv, batch_size=1))
        assert len(batches) == 1
        _, req = batches[0]
        assert req["inputs"][0]["shape"] == [1, 3]
        assert req["inputs"][0]["data"] == [[1.0, 2.0, 3.0]]

    def test_returns_model_name_from_pbtxt(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path, name="fraud-detection")
        csv   = _write_csv(tmp_path, "a,b\n1.0,2.0\n")
        model_name, _ = next(iter_requests(pbtxt, csv))
        assert model_name == "fraud-detection"

    def test_batch_size_one_yields_one_request_per_row(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        csv   = _write_csv(tmp_path, "a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n")
        batches = list(iter_requests(pbtxt, csv, batch_size=1))
        assert len(batches) == 3
        for _, req in batches:
            assert req["inputs"][0]["shape"][0] == 1

    def test_batch_size_equals_row_count_yields_one_request(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        csv   = _write_csv(tmp_path, "a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n")
        batches = list(iter_requests(pbtxt, csv, batch_size=3))
        assert len(batches) == 1
        _, req = batches[0]
        assert req["inputs"][0]["shape"] == [3, 2]

    def test_batch_size_larger_than_rows_yields_one_partial_batch(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        csv   = _write_csv(tmp_path, "a,b\n1.0,2.0\n3.0,4.0\n")
        batches = list(iter_requests(pbtxt, csv, batch_size=100))
        assert len(batches) == 1
        _, req = batches[0]
        assert req["inputs"][0]["shape"][0] == 2

    def test_final_batch_contains_remainder_rows(self, tmp_path: Path) -> None:
        """5 rows with batch_size=2 → batches of [2, 2, 1]."""
        pbtxt = _write_pbtxt(tmp_path)
        rows  = "\n".join(f"{i}.0,{i}.0" for i in range(5))
        csv   = _write_csv(tmp_path, f"a,b\n{rows}\n")
        batches = list(iter_requests(pbtxt, csv, batch_size=2))
        assert len(batches) == 3
        sizes = [req["inputs"][0]["shape"][0] for _, req in batches]
        assert sizes == [2, 2, 1]

    def test_no_padding_or_duplication(self, tmp_path: Path) -> None:
        """Total rows across all batches equals the CSV data rows."""
        pbtxt = _write_pbtxt(tmp_path)
        csv   = _write_csv(tmp_path, "a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n")
        batches = list(iter_requests(pbtxt, csv, batch_size=2))
        total = sum(req["inputs"][0]["shape"][0] for _, req in batches)
        assert total == 3


# ---------------------------------------------------------------------------
# iter_requests — CSV: request structure
# ---------------------------------------------------------------------------

class TestIterRequestsCsvStructure:

    def test_kserve_v2_envelope(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        csv   = _write_csv(tmp_path, "a,b\n1.0,2.0\n")
        _, req = next(iter_requests(pbtxt, csv))
        assert "inputs" in req
        inp = req["inputs"][0]
        assert "name"     in inp
        assert "shape"    in inp
        assert "datatype" in inp
        assert "data"     in inp

    def test_tensor_name_matches_pbtxt(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        csv   = _write_csv(tmp_path, "a,b\n1.0,2.0\n")
        _, req = next(iter_requests(pbtxt, csv))
        assert req["inputs"][0]["name"] == "dense_input"

    def test_datatype_fp32_cast(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path, datatype="TYPE_FP32")
        csv   = _write_csv(tmp_path, "a,b\n1,2\n")
        _, req = next(iter_requests(pbtxt, csv))
        assert req["inputs"][0]["datatype"] == "FP32"
        assert all(isinstance(v, float) for v in req["inputs"][0]["data"][0])

    def test_datatype_int32_cast(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path, datatype="TYPE_INT32")
        csv   = _write_csv(tmp_path, "a,b\n1.0,2.0\n")
        _, req = next(iter_requests(pbtxt, csv))
        assert req["inputs"][0]["datatype"] == "INT32"
        assert all(isinstance(v, int) for v in req["inputs"][0]["data"][0])

    def test_shape_reflects_batch_size(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        csv   = _write_csv(tmp_path, "a,b,c\n1.0,2.0,3.0\n4.0,5.0,6.0\n")
        _, req = next(iter_requests(pbtxt, csv, batch_size=2))
        assert req["inputs"][0]["shape"] == [2, 3]

    def test_data_is_list_of_rows(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        csv   = _write_csv(tmp_path, "a,b\n1.0,2.0\n3.0,4.0\n")
        _, req = next(iter_requests(pbtxt, csv, batch_size=2))
        assert req["inputs"][0]["data"] == [[1.0, 2.0], [3.0, 4.0]]

    def test_result_is_json_serialisable(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        csv   = _write_csv(tmp_path, "a,b\n1.0,2.0\n")
        _, req = next(iter_requests(pbtxt, csv))
        json.dumps(req)  # must not raise


# ---------------------------------------------------------------------------
# iter_requests — JSON dataset
# ---------------------------------------------------------------------------

class TestIterRequestsJson:

    def test_yields_once_for_json_dataset(self, tmp_path: Path) -> None:
        """A JSON pre-built batch produces exactly one yield regardless of row count."""
        pbtxt = _write_pbtxt(tmp_path)
        doc   = _kserve_envelope(n_rows=50)
        jf    = _write_json(tmp_path, doc)
        batches = list(iter_requests(pbtxt, jf))
        assert len(batches) == 1

    def test_json_payload_is_returned_verbatim(self, tmp_path: Path) -> None:
        """The yielded request dict is identical to the JSON file content."""
        pbtxt = _write_pbtxt(tmp_path)
        doc   = _kserve_envelope(tensor_name="my_tensor", n_rows=3, n_cols=4, datatype="FP32")
        jf    = _write_json(tmp_path, doc)
        _, req = next(iter_requests(pbtxt, jf))
        assert req == doc

    def test_json_model_name_comes_from_pbtxt(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path, name="loan-model")
        jf    = _write_json(tmp_path, _kserve_envelope())
        model_name, _ = next(iter_requests(pbtxt, jf))
        assert model_name == "loan-model"

    def test_json_batch_size_is_ignored(self, tmp_path: Path) -> None:
        """batch_size has no effect on JSON datasets — always one yield."""
        pbtxt = _write_pbtxt(tmp_path)
        doc   = _kserve_envelope(n_rows=10)
        jf    = _write_json(tmp_path, doc)
        for bs in (1, 3, 100):
            batches = list(iter_requests(pbtxt, jf, batch_size=bs))
            assert len(batches) == 1, f"expected 1 batch for batch_size={bs}"

    def test_json_multi_tensor_model_allowed(self, tmp_path: Path) -> None:
        """JSON datasets are not restricted to single-tensor models."""
        pbtxt = _write_pbtxt_multi(tmp_path)
        doc   = {
            "inputs": [
                {"name": "a", "shape": [2, 2], "datatype": "FP32", "data": [[1.0, 2.0], [3.0, 4.0]]},
                {"name": "b", "shape": [2, 2], "datatype": "FP32", "data": [[5.0, 6.0], [7.0, 8.0]]},
            ]
        }
        jf = _write_json(tmp_path, doc)
        batches = list(iter_requests(pbtxt, jf))
        assert len(batches) == 1
        assert batches[0][1] == doc

    def test_json_is_json_serialisable(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        jf    = _write_json(tmp_path, _kserve_envelope())
        _, req = next(iter_requests(pbtxt, jf))
        json.dumps(req)  # must not raise


# ---------------------------------------------------------------------------
# iter_requests — JSON error handling
# ---------------------------------------------------------------------------

class TestIterRequestsJsonErrors:

    def test_raises_on_missing_json(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            list(iter_requests(pbtxt, tmp_path / "missing.json"))

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        jf    = tmp_path / "bad.json"
        jf.write_text("not-json{{")
        with pytest.raises(ValueError, match="not valid JSON"):
            list(iter_requests(pbtxt, jf))

    def test_raises_when_inputs_key_missing(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        jf    = _write_json(tmp_path, {"outputs": [{"name": "o", "data": [1]}]})
        with pytest.raises(ValueError, match="no 'inputs' list"):
            list(iter_requests(pbtxt, jf))

    def test_raises_when_inputs_is_empty_list(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        jf    = _write_json(tmp_path, {"inputs": []})
        with pytest.raises(ValueError, match="no 'inputs' list"):
            list(iter_requests(pbtxt, jf))


# ---------------------------------------------------------------------------
# iter_requests — CSV error handling
# ---------------------------------------------------------------------------

class TestIterRequestsCsvErrors:

    def test_raises_on_batch_size_zero(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        csv   = _write_csv(tmp_path, "a,b\n1.0,2.0\n")
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            list(iter_requests(pbtxt, csv, batch_size=0))

    def test_raises_on_negative_batch_size(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        csv   = _write_csv(tmp_path, "a,b\n1.0,2.0\n")
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            list(iter_requests(pbtxt, csv, batch_size=-5))

    def test_raises_on_missing_pbtxt(self, tmp_path: Path) -> None:
        csv = _write_csv(tmp_path, "a,b\n1.0,2.0\n")
        with pytest.raises(FileNotFoundError):
            list(iter_requests(tmp_path / "missing.pbtxt", csv))

    def test_raises_on_missing_csv(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            list(iter_requests(pbtxt, tmp_path / "missing.csv"))

    def test_raises_on_empty_csv(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        csv   = _write_csv(tmp_path, "")
        with pytest.raises(ValueError, match="no data rows"):
            list(iter_requests(pbtxt, csv))

    def test_raises_on_multi_tensor_model_flat_csv_no_matching_columns(
        self, tmp_path: Path
    ) -> None:
        """Flat-tensor fallback requires single-tensor model when no column matches a tensor."""
        pbtxt = _write_pbtxt_multi(tmp_path)
        csv   = _write_csv(tmp_path, "x,y\n1.0,2.0\n")
        with pytest.raises(ValueError, match="no columns matching"):
            list(iter_requests(pbtxt, csv))

    def test_raises_on_unsupported_extension(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        jf    = tmp_path / "data.parquet"
        jf.write_bytes(b"PAR1")
        with pytest.raises(ValueError, match="Unsupported dataset file type"):
            list(iter_requests(pbtxt, jf))


# ---------------------------------------------------------------------------
# iter_requests — JSON array dataset (list of KServe v2 envelopes)
# ---------------------------------------------------------------------------

class TestIterRequestsJsonArray:

    def test_yields_one_per_element(self, tmp_path: Path) -> None:
        """A JSON array of N envelopes yields N requests."""
        pbtxt = _write_pbtxt(tmp_path)
        doc   = [_kserve_envelope(n_rows=1) for _ in range(5)]
        jf    = _write_json(tmp_path, doc)
        batches = list(iter_requests(pbtxt, jf))
        assert len(batches) == 5

    def test_each_element_returned_verbatim(self, tmp_path: Path) -> None:
        """Each yielded request is the exact element from the array."""
        pbtxt = _write_pbtxt(tmp_path)
        envs  = [_kserve_envelope(n_rows=i + 1) for i in range(3)]
        jf    = _write_json(tmp_path, envs)
        batches = list(iter_requests(pbtxt, jf))
        for (_, req), expected in zip(batches, envs):
            assert req == expected

    def test_model_name_comes_from_pbtxt(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path, name="upi-fraud")
        doc   = [_kserve_envelope()]
        jf    = _write_json(tmp_path, doc)
        model_name, _ = next(iter_requests(pbtxt, jf))
        assert model_name == "upi-fraud"

    def test_batch_size_ignored_for_array(self, tmp_path: Path) -> None:
        """batch_size has no effect on JSON array datasets — one yield per element."""
        pbtxt = _write_pbtxt(tmp_path)
        doc   = [_kserve_envelope() for _ in range(4)]
        jf    = _write_json(tmp_path, doc)
        for bs in (1, 2, 100):
            batches = list(iter_requests(pbtxt, jf, batch_size=bs))
            assert len(batches) == 4, f"expected 4 for batch_size={bs}"

    def test_single_element_array(self, tmp_path: Path) -> None:
        """An array with one element yields exactly one request."""
        pbtxt = _write_pbtxt(tmp_path)
        doc   = [_kserve_envelope(n_rows=10, n_cols=5)]
        jf    = _write_json(tmp_path, doc)
        batches = list(iter_requests(pbtxt, jf))
        assert len(batches) == 1
        assert batches[0][1] == doc[0]

    def test_multi_tensor_elements_allowed(self, tmp_path: Path) -> None:
        """JSON array elements may contain multi-tensor envelopes."""
        pbtxt = _write_pbtxt_multi(tmp_path)
        element = {
            "inputs": [
                {"name": "a", "shape": [1, 2], "datatype": "FP32", "data": [[1.0, 2.0]]},
                {"name": "b", "shape": [1, 2], "datatype": "FP32", "data": [[3.0, 4.0]]},
            ]
        }
        jf = _write_json(tmp_path, [element, element])
        batches = list(iter_requests(pbtxt, jf))
        assert len(batches) == 2
        assert batches[0][1] == element

    def test_result_is_json_serialisable(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        doc   = [_kserve_envelope() for _ in range(3)]
        jf    = _write_json(tmp_path, doc)
        for _, req in iter_requests(pbtxt, jf):
            json.dumps(req)  # must not raise

    def test_raises_on_empty_array(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        jf    = _write_json(tmp_path, [])
        with pytest.raises(ValueError, match="empty list"):
            list(iter_requests(pbtxt, jf))

    def test_raises_when_element_missing_inputs(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        doc   = [_kserve_envelope(), {"outputs": [{"name": "o", "data": [1]}]}]
        jf    = _write_json(tmp_path, doc)
        with pytest.raises(ValueError, match=r"element \[1\]"):
            list(iter_requests(pbtxt, jf))

    def test_raises_when_element_is_not_a_dict(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        doc   = [_kserve_envelope(), 42]
        jf    = _write_json(tmp_path, doc)
        with pytest.raises(ValueError, match=r"element \[1\]"):
            list(iter_requests(pbtxt, jf))

    def test_raises_when_element_inputs_is_empty(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt(tmp_path)
        doc   = [{"inputs": []}]
        jf    = _write_json(tmp_path, doc)
        with pytest.raises(ValueError, match=r"element \[0\]"):
            list(iter_requests(pbtxt, jf))


# ---------------------------------------------------------------------------
# iter_requests — multi-tensor CSV (column-per-tensor layout)
# ---------------------------------------------------------------------------

class TestIterRequestsCsvMultiTensor:
    """CSV with headers matching tensor names → column-per-tensor layout."""

    def test_each_tensor_gets_its_own_input(self, tmp_path: Path) -> None:
        """Output must have one input entry per model tensor."""
        pbtxt = _write_pbtxt_multi(tmp_path)
        csv   = _write_csv(tmp_path, "a,b\n1.0,2.0\n3.0,4.0\n")
        _, req = next(iter_requests(pbtxt, csv, batch_size=2))
        assert len(req["inputs"]) == 2
        names = [t["name"] for t in req["inputs"]]
        assert names == ["a", "b"]

    def test_tensor_order_follows_pbtxt(self, tmp_path: Path) -> None:
        """Tensor order must reflect pbtxt definition, not CSV column order."""
        pbtxt = _write_pbtxt_multi(tmp_path)
        # CSV has columns in reverse order b, a
        csv   = _write_csv(tmp_path, "b,a\n2.0,1.0\n")
        _, req = next(iter_requests(pbtxt, csv))
        assert req["inputs"][0]["name"] == "a"
        assert req["inputs"][1]["name"] == "b"

    def test_batch_size_1_produces_one_row_per_tensor(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt_multi(tmp_path)
        csv   = _write_csv(tmp_path, "a,b\n1.0,2.0\n3.0,4.0\n")
        batches = list(iter_requests(pbtxt, csv, batch_size=1))
        assert len(batches) == 2
        for _, req in batches:
            assert req["inputs"][0]["shape"] == [1, 1]
            assert req["inputs"][1]["shape"] == [1, 1]

    def test_batch_size_n_packs_n_rows_into_each_tensor(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt_multi(tmp_path)
        csv   = _write_csv(tmp_path, "a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n")
        _, req = next(iter_requests(pbtxt, csv, batch_size=3))
        assert req["inputs"][0]["shape"] == [3, 1]
        assert req["inputs"][1]["shape"] == [3, 1]

    def test_shape_is_n_by_1_per_tensor(self, tmp_path: Path) -> None:
        """Each tensor must have shape [batch_size, 1] — one scalar per observation."""
        pbtxt = _write_pbtxt_multi(tmp_path)
        csv   = _write_csv(tmp_path, "a,b\n10.0,20.0\n30.0,40.0\n50.0,60.0\n70.0,80.0\n")
        _, req = next(iter_requests(pbtxt, csv, batch_size=4))
        for tensor in req["inputs"]:
            assert tensor["shape"] == [4, 1]

    def test_data_values_correct_per_tensor(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt_multi(tmp_path)
        csv   = _write_csv(tmp_path, "a,b\n1.0,10.0\n2.0,20.0\n3.0,30.0\n")
        _, req = next(iter_requests(pbtxt, csv, batch_size=3))
        a_data = req["inputs"][0]["data"]
        b_data = req["inputs"][1]["data"]
        assert a_data == [[1.0], [2.0], [3.0]]
        assert b_data == [[10.0], [20.0], [30.0]]

    def test_each_request_contains_all_tensors(self, tmp_path: Path) -> None:
        """Every yielded request must include all model tensors."""
        pbtxt = _write_pbtxt_multi(tmp_path)
        csv   = _write_csv(tmp_path, "a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n")
        for _, req in iter_requests(pbtxt, csv, batch_size=1):
            assert len(req["inputs"]) == 2

    def test_batching_produces_correct_number_of_requests(self, tmp_path: Path) -> None:
        """5 rows batch_size=2 → 3 requests (sizes 2, 2, 1)."""
        pbtxt = _write_pbtxt_multi(tmp_path)
        rows  = "\n".join(f"{i}.0,{i*10}.0" for i in range(1, 6))
        csv   = _write_csv(tmp_path, f"a,b\n{rows}\n")
        batches = list(iter_requests(pbtxt, csv, batch_size=2))
        assert len(batches) == 3
        sizes = [req["inputs"][0]["shape"][0] for _, req in batches]
        assert sizes == [2, 2, 1]

    def test_no_row_duplication_across_batches(self, tmp_path: Path) -> None:
        """Total rows across all batches equals the CSV row count."""
        pbtxt = _write_pbtxt_multi(tmp_path)
        rows  = "\n".join(f"{i}.0,{i}.0" for i in range(7))
        csv   = _write_csv(tmp_path, f"a,b\n{rows}\n")
        batches = list(iter_requests(pbtxt, csv, batch_size=3))
        total = sum(req["inputs"][0]["shape"][0] for _, req in batches)
        assert total == 7

    def test_mixed_datatypes_cast_correctly(self, tmp_path: Path) -> None:
        """FP32, INT32, and BYTES columns must be cast to their respective types."""
        pbtxt = _write_pbtxt_multi_types(tmp_path)
        csv   = _write_csv(tmp_path, "amount,count,category\n9.5,3,fraud\n")
        _, req = next(iter_requests(pbtxt, csv))
        tensors = {t["name"]: t for t in req["inputs"]}
        assert tensors["amount"]["data"]   == [[9.5]]
        assert tensors["count"]["data"]    == [[3]]
        assert tensors["category"]["data"] == [["fraud"]]

    def test_result_is_json_serialisable(self, tmp_path: Path) -> None:
        pbtxt = _write_pbtxt_multi(tmp_path)
        csv   = _write_csv(tmp_path, "a,b\n1.0,2.0\n3.0,4.0\n")
        for _, req in iter_requests(pbtxt, csv, batch_size=2):
            json.dumps(req)  # must not raise

    def test_extra_csv_columns_are_ignored(self, tmp_path: Path) -> None:
        """Columns not present in pbtxt must be silently ignored."""
        pbtxt = _write_pbtxt_multi(tmp_path)
        csv   = _write_csv(tmp_path, "a,b,extra_col\n1.0,2.0,999.0\n")
        _, req = next(iter_requests(pbtxt, csv))
        names = [t["name"] for t in req["inputs"]]
        assert "extra_col" not in names
        assert len(req["inputs"]) == 2

    def test_partial_column_match_uses_placeholder_for_missing(
        self, tmp_path: Path
    ) -> None:
        """If only some tensor columns are present, missing tensors get placeholder."""
        pbtxt = _write_pbtxt_multi(tmp_path)
        csv   = _write_csv(tmp_path, "a\n5.0\n")   # only tensor 'a', no 'b'
        _, req = next(iter_requests(pbtxt, csv))
        assert len(req["inputs"]) == 2
        a_tensor = next(t for t in req["inputs"] if t["name"] == "a")
        b_tensor = next(t for t in req["inputs"] if t["name"] == "b")
        assert a_tensor["data"] == [[5.0]]
        # 'b' absent → placeholder value (0.0 for FP32)
        assert b_tensor["data"] == [0.0]


# ---------------------------------------------------------------------------
# generate_batched_requests
# ---------------------------------------------------------------------------

class TestGenerateBatchedRequests:

    def test_output_is_a_json_array(self, tmp_path: Path) -> None:
        pbtxt  = _write_pbtxt_multi(tmp_path)
        csv    = _write_csv(tmp_path, "a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n")
        out    = tmp_path / "out.json"
        generate_batched_requests(pbtxt, csv, out, batch_size=1)
        doc = json.loads(out.read_text())
        assert isinstance(doc, list)

    def test_array_length_equals_number_of_batches(self, tmp_path: Path) -> None:
        """3 rows, batch_size=1 → array of 3 requests."""
        pbtxt  = _write_pbtxt_multi(tmp_path)
        csv    = _write_csv(tmp_path, "a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n")
        out    = tmp_path / "out.json"
        n = generate_batched_requests(pbtxt, csv, out, batch_size=1)
        assert n == 3
        assert len(json.loads(out.read_text())) == 3

    def test_each_element_is_a_valid_kserve_v2_envelope(self, tmp_path: Path) -> None:
        pbtxt  = _write_pbtxt_multi(tmp_path)
        csv    = _write_csv(tmp_path, "a,b\n1.0,2.0\n3.0,4.0\n")
        out    = tmp_path / "out.json"
        generate_batched_requests(pbtxt, csv, out, batch_size=1)
        for elem in json.loads(out.read_text()):
            assert "inputs" in elem
            assert isinstance(elem["inputs"], list)
            assert len(elem["inputs"]) > 0

    def test_each_element_contains_all_model_tensors(self, tmp_path: Path) -> None:
        pbtxt  = _write_pbtxt_multi(tmp_path)
        csv    = _write_csv(tmp_path, "a,b\n1.0,2.0\n3.0,4.0\n")
        out    = tmp_path / "out.json"
        generate_batched_requests(pbtxt, csv, out, batch_size=1)
        for elem in json.loads(out.read_text()):
            names = [t["name"] for t in elem["inputs"]]
            assert set(names) == {"a", "b"}

    def test_batch_size_n_shapes_are_correct(self, tmp_path: Path) -> None:
        """batch_size=3: first element has shape [3,1], last has remainder."""
        pbtxt  = _write_pbtxt_multi(tmp_path)
        rows   = "\n".join(f"{i}.0,{i*2}.0" for i in range(1, 8))   # 7 rows
        csv    = _write_csv(tmp_path, f"a,b\n{rows}\n")
        out    = tmp_path / "out.json"
        generate_batched_requests(pbtxt, csv, out, batch_size=3)
        doc = json.loads(out.read_text())
        assert len(doc) == 3   # ceil(7/3)
        assert doc[0]["inputs"][0]["shape"] == [3, 1]
        assert doc[1]["inputs"][0]["shape"] == [3, 1]
        assert doc[2]["inputs"][0]["shape"] == [1, 1]

    def test_returns_request_count(self, tmp_path: Path) -> None:
        pbtxt  = _write_pbtxt_multi(tmp_path)
        rows   = "\n".join(f"{i}.0,{i}.0" for i in range(5))
        csv    = _write_csv(tmp_path, f"a,b\n{rows}\n")
        out    = tmp_path / "out.json"
        n = generate_batched_requests(pbtxt, csv, out, batch_size=2)
        assert n == 3   # ceil(5/2)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        pbtxt  = _write_pbtxt_multi(tmp_path)
        csv    = _write_csv(tmp_path, "a,b\n1.0,2.0\n")
        out    = tmp_path / "deep" / "nested" / "out.json"
        generate_batched_requests(pbtxt, csv, out)
        assert out.exists()

    def test_output_file_is_valid_json(self, tmp_path: Path) -> None:
        pbtxt  = _write_pbtxt_multi(tmp_path)
        csv    = _write_csv(tmp_path, "a,b\n1.0,2.0\n3.0,4.0\n")
        out    = tmp_path / "out.json"
        generate_batched_requests(pbtxt, csv, out)
        json.loads(out.read_text())  # must not raise

    def test_json_dataset_preserved_verbatim(self, tmp_path: Path) -> None:
        """For a JSON array dataset the elements pass through unchanged."""
        pbtxt = _write_pbtxt(tmp_path)
        doc   = [_kserve_envelope(n_rows=1) for _ in range(3)]
        jf    = _write_json(tmp_path, doc)
        out   = tmp_path / "out.json"
        n = generate_batched_requests(pbtxt, jf, out)
        assert n == 3
        result = json.loads(out.read_text())
        assert result == doc


# ---------------------------------------------------------------------------
# read_csv_headers
# ---------------------------------------------------------------------------

class TestReadCsvHeaders:
    def test_returns_headers_in_order(self, tmp_path: Path) -> None:
        p = tmp_path / "d.csv"
        p.write_text("age,income,is_male\n30,1000,1\n")
        assert read_csv_headers(p) == ["age", "income", "is_male"]

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.csv"
        p.write_text("")
        with pytest.raises(ValueError, match="no header row"):
            read_csv_headers(p)


# ---------------------------------------------------------------------------
# derive_input_name_mapping
# ---------------------------------------------------------------------------

def _items(*specs: tuple[str, str, int]) -> dict[str, dict]:
    """Build an inputSchema.items dict from (display_name, internal_name, columnIndex)."""
    return {
        display: {"name": internal, "type": "DOUBLE", "columnIndex": idx}
        for display, internal, idx in specs
    }


class TestDeriveInputNameMapping:
    def test_scalar_tensors_with_identical_names_map_by_identity(self) -> None:
        """Multi-scalar model whose feature names already equal the CSV headers."""
        headers = ["age", "income", "is_male"]
        items = _items(("age", "age", 0), ("income", "income", 1), ("is_male", "is_male", 2))
        assert derive_input_name_mapping(headers, items, num_input_tensors=3) == {
            "age": "Age", "income": "Income", "is_male": "Is Male",
        }

    def test_single_flattened_tensor_maps_by_column_index(self) -> None:
        headers = ["num_children", "income", "is_male"]
        items = _items(
            ("customer_data_input-0", "customer_data_input-0", 0),
            ("customer_data_input-1", "customer_data_input-1", 1),
            ("customer_data_input-2", "customer_data_input-2", 2),
        )
        assert derive_input_name_mapping(headers, items, num_input_tensors=1) == {
            "customer_data_input-0": "Num Children",
            "customer_data_input-1": "Income",
            "customer_data_input-2": "Is Male",
        }

    def test_column_index_respects_schema_order_not_header_order(self) -> None:
        """A flat tensor whose items arrive out of order still maps by columnIndex."""
        headers = ["a", "b", "c"]
        items = _items(("t-2", "t-2", 2), ("t-0", "t-0", 0), ("t-1", "t-1", 1))
        assert derive_input_name_mapping(headers, items, num_input_tensors=1) == {
            "t-0": "A", "t-1": "B", "t-2": "C",
        }

    def test_multiple_scalar_tensors_with_suffixed_names_skip(self) -> None:
        """Suffixed names on a >1-tensor model can't safely use columnIndex → skip."""
        headers = ["age", "income"]
        items = _items(("age-0", "age-0", 0), ("income-0", "income-0", 1))
        assert derive_input_name_mapping(headers, items, num_input_tensors=2) == {}

    def test_multiple_vector_tensors_skip(self) -> None:
        headers = ["a", "b", "c", "d"]
        items = _items(
            ("t1-0", "t1-0", 0), ("t1-1", "t1-1", 1),
            ("t2-0", "t2-0", 2), ("t2-1", "t2-1", 3),
        )
        assert derive_input_name_mapping(headers, items, num_input_tensors=2) == {}

    def test_count_mismatch_skips(self) -> None:
        headers = ["a", "b", "c"]
        items = _items(("t-0", "t-0", 0), ("t-1", "t-1", 1))
        assert derive_input_name_mapping(headers, items, num_input_tensors=1) == {}

    def test_duplicate_headers_skip(self) -> None:
        headers = ["a", "a", "b"]
        items = _items(("t-0", "t-0", 0), ("t-1", "t-1", 1), ("t-2", "t-2", 2))
        assert derive_input_name_mapping(headers, items, num_input_tensors=1) == {}

    def test_non_bijective_column_index_skips(self) -> None:
        headers = ["a", "b", "c"]
        items = _items(("t-0", "t-0", 0), ("t-1", "t-1", 1), ("t-2", "t-2", 1))
        assert derive_input_name_mapping(headers, items, num_input_tensors=1) == {}

    def test_missing_column_index_skips(self) -> None:
        headers = ["a", "b"]
        items = {
            "t-0": {"name": "t-0", "type": "DOUBLE"},          # no columnIndex
            "t-1": {"name": "t-1", "type": "DOUBLE", "columnIndex": 1},
        }
        assert derive_input_name_mapping(headers, items, num_input_tensors=1) == {}

    def test_no_headers_returns_empty(self) -> None:
        items = _items(("t-0", "t-0", 0))
        assert derive_input_name_mapping([], items, num_input_tensors=1) == {}

    def test_no_schema_items_returns_empty(self) -> None:
        assert derive_input_name_mapping(["a", "b"], {}, num_input_tensors=1) == {}


# ---------------------------------------------------------------------------
# normalize_feature_name
# ---------------------------------------------------------------------------

class TestNormalizeFeatureName:
    @pytest.mark.parametrize("raw,expected", [
        ("Bank_Name", "bank name"),
        ("bank_name", "bank name"),
        ("Bank Name", "bank name"),
        ("USER_MEAN_AMOUNT", "user mean amount"),
        ("user_mean_amount", "user mean amount"),
        ("  Multiple   Spaces_here ", "multiple spaces here"),
    ])
    def test_normalization(self, raw: str, expected: str) -> None:
        assert normalize_feature_name(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("bank_name", "Bank Name"),
        ("USER_MEAN_AMOUNT", "User Mean Amount"),
        ("num_children", "Num Children"),
        ("Transaction Amount", "Transaction Amount"),
    ])
    def test_humanize(self, raw: str, expected: str) -> None:
        assert humanize_feature_name(raw) == expected


class TestDeriveInputNameMappingNormalized:
    def test_matches_across_case_and_separators(self) -> None:
        """Schema names and headers that differ only by case/underscores match."""
        headers = ["bank name", "user_mean_amount"]
        items = _items(("Bank_Name", "Bank_Name", 0), ("USER_MEAN_AMOUNT", "USER_MEAN_AMOUNT", 1))
        assert derive_input_name_mapping(headers, items, num_input_tensors=2) == {
            "Bank_Name": "Bank Name",
            "USER_MEAN_AMOUNT": "User Mean Amount",
        }

    def test_display_names_are_title_cased(self) -> None:
        headers = ["transaction_amount"]
        items = _items(("transaction_amount", "transaction_amount", 0))
        assert derive_input_name_mapping(headers, items, num_input_tensors=1) == {
            "transaction_amount": "Transaction Amount",
        }

    def test_ambiguous_normalized_headers_are_skipped(self) -> None:
        """Two headers collapsing to the same normal form don't match by name."""
        headers = ["Bank_Name", "bank name"]      # both -> "bank name"
        items = _items(("BANK NAME", "BANK NAME", 0), ("other", "other", 1))
        # 'BANK NAME' is ambiguous → not matched; 'other' has no header → skipped.
        # num_input_tensors=2 blocks the columnIndex fallback.
        assert derive_input_name_mapping(headers, items, num_input_tensors=2) == {}
