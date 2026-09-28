"""Generic Triton/KServe v2 request generation helpers."""

import csv as csv_mod
import io
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

_TRITON_TO_KSERVE: dict[str, str] = {
    "TYPE_FP16": "FP16",
    "TYPE_FP32": "FP32",
    "TYPE_FP64": "FP64",
    "TYPE_INT8": "INT8",
    "TYPE_INT16": "INT16",
    "TYPE_INT32": "INT32",
    "TYPE_INT64": "INT64",
    "TYPE_UINT8": "UINT8",
    "TYPE_UINT16": "UINT16",
    "TYPE_UINT32": "UINT32",
    "TYPE_UINT64": "UINT64",
    "TYPE_BOOL": "BOOL",
    "TYPE_STRING": "BYTES",
    "TYPE_BYTES": "BYTES",
}

_PLACEHOLDER: dict[str, object] = {
    "FP16": 0.0,
    "FP32": 0.0,
    "FP64": 0.0,
    "INT8": 0,
    "INT16": 0,
    "INT32": 0,
    "INT64": 0,
    "UINT8": 0,
    "UINT16": 0,
    "UINT32": 0,
    "UINT64": 0,
    "BOOL": False,
    "BYTES": "",
}

_FLOAT_TYPES = {"FP16", "FP32", "FP64"}
_INT_TYPES = {"INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16", "UINT32", "UINT64"}


@dataclass
class TensorSpec:
    name: str
    datatype: str
    dims: list[int]
    max_batch_size: int = 0


@dataclass
class CsvSource:
    path: Path
    key_col: str | None
    key_val: str | None
    col_map: dict[str, str]
    flat_tensor: str | None = None



def _extract_bracket_block(text: str, keyword: str) -> str | None:
    """Return the inner content of the first ``keyword [ ... ]`` block in *text*."""
    pattern = re.compile(rf"\b{re.escape(keyword)}\s*\[")
    match = pattern.search(text)
    if not match:
        return None
    pos = match.end()
    depth = 1
    start = pos
    while pos < len(text) and depth > 0:
        if text[pos] == "[":
            depth += 1
        elif text[pos] == "]":
            depth -= 1
        pos += 1
    return text[start : pos - 1]



def _split_brace_blocks(text: str) -> list[str]:
    """Return the inner content of every top-level ``{ ... }`` block in *text*."""
    blocks: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                blocks.append(text[start:i])
                start = -1
    return blocks



def parse_pbtxt(path: Path) -> tuple[str, list[TensorSpec]]:
    """Return ``(model_name, [TensorSpec, ...])`` from a Triton config.pbtxt."""
    text = path.read_text()

    name_match = re.search(r'^name\s*:\s*"([^"]+)"', text, re.MULTILINE)
    model_name = name_match.group(1) if name_match else path.stem

    mbs_match = re.search(r"^max_batch_size\s*:\s*(\d+)", text, re.MULTILINE)
    max_batch_size = int(mbs_match.group(1)) if mbs_match else 0

    input_block = _extract_bracket_block(text, "input")
    if input_block is None:
        raise ValueError(f"No 'input [...]' block found in: {path}")

    specs: list[TensorSpec] = []
    for entry in _split_brace_blocks(input_block):
        name_match = re.search(r'name\s*:\s*"([^"]+)"', entry)
        if not name_match:
            continue
        tensor_name = name_match.group(1)

        dtype_match = re.search(r"data_type\s*:\s*(TYPE_\w+)", entry)
        if not dtype_match:
            raise ValueError(f"Tensor '{tensor_name}': missing data_type in {path}")
        kserve_type = _TRITON_TO_KSERVE.get(dtype_match.group(1))
        if kserve_type is None:
            raise ValueError(
                f"Tensor '{tensor_name}': unknown Triton type '{dtype_match.group(1)}'."
            )

        dims_match = re.search(r"dims\s*:\s*\[([^\]]*)\]", entry)
        if not dims_match:
            raise ValueError(f"Tensor '{tensor_name}': missing dims in {path}")
        dims = [int(d.strip()) for d in dims_match.group(1).split(",") if d.strip()]

        specs.append(
            TensorSpec(
                name=tensor_name,
                datatype=kserve_type,
                dims=dims,
                max_batch_size=max_batch_size,
            )
        )

    if not specs:
        raise ValueError(f"No input tensors parsed from: {path}")

    return model_name, specs



def parse_pbtxt_output_names(path: Path) -> list[str]:
    """Return the ordered output tensor names from a Triton ``config.pbtxt``.

    Reuses the same bracket/brace scanning as :func:`parse_pbtxt`.  Returns an
    empty list when the file has no ``output [...]`` block.
    """
    text = path.read_text()
    output_block = _extract_bracket_block(text, "output")
    if output_block is None:
        return []
    names: list[str] = []
    for entry in _split_brace_blocks(output_block):
        name_match = re.search(r'name\s*:\s*"([^"]+)"', entry)
        if name_match:
            names.append(name_match.group(1))
    return names


def _flat_row(row: dict, headers: list[str], transforms: dict[str, str]) -> list[float]:
    """Convert one CSV row to a flat float list (flat-tensor mode)."""
    flat: list[float] = []
    for col in headers:
        raw = row[col]
        if col in transforms:
            raw = _apply_transform(raw, transforms[col])
        flat.append(float(raw))
    return flat



def load_csv_source(
    source: CsvSource,
    row_index: int,
    transforms: dict[str, str] | None = None,
    all_rows: bool = False,
) -> dict[str, object]:
    """Read one (or all) rows from a CSV and return ``{tensor_name: value}``."""
    transforms = transforms or {}

    if not source.path.exists():
        raise FileNotFoundError(f"CSV file not found: {source.path}")

    reader = csv_mod.DictReader(io.StringIO(source.path.read_text()))
    rows = list(reader)
    if not rows:
        raise ValueError(f"CSV '{source.path.name}' contains no data rows.")

    headers = list(rows[0].keys())

    if source.flat_tensor is not None and all_rows:
        batch = [_flat_row(r, headers, transforms) for r in rows]
        return {source.flat_tensor: batch}

    if source.key_col is not None:
        if source.key_col not in headers:
            raise ValueError(
                f"CSV '{source.path.name}': key column '{source.key_col}' not found. "
                f"Available columns: {headers}"
            )
        matched = [r for r in rows if r[source.key_col] == source.key_val]
        if not matched:
            raise ValueError(
                f"CSV '{source.path.name}': no row where "
                f"'{source.key_col}' = '{source.key_val}'."
            )
        row = matched[0]
    else:
        if row_index >= len(rows):
            raise ValueError(
                f"CSV '{source.path.name}': --row {row_index} is out of range "
                f"(file has {len(rows)} data rows)."
            )
        row = rows[row_index]

    if source.flat_tensor is not None:
        return {source.flat_tensor: _flat_row(row, headers, transforms)}

    result: dict[str, str] = {}
    for csv_col, tensor_name in source.col_map.items():
        if csv_col not in headers:
            raise ValueError(
                f"CSV '{source.path.name}': column '{csv_col}' not found. "
                f"Available: {headers}"
            )
        raw = row[csv_col]
        if csv_col in transforms:
            raw = _apply_transform(raw, transforms[csv_col])
        result[tensor_name] = raw

    return result



def _apply_transform(raw: str, func: str) -> str:
    """Apply a named transform to a raw CSV string; return a string result."""
    value = raw.strip()
    if func == "yes_no_bin":
        return "1.0" if value.lower() in ("yes", "1", "true") else "0.0"
    if func == "log":
        return str(math.log(float(value)))
    if func == "hour":
        return str(float(value.split(":")[0]))
    if func == "day_of_week":
        dt = _parse_date(value)
        return str(float(dt.weekday()))
    if func == "is_weekend":
        dt = _parse_date(value)
        return "1.0" if dt.weekday() >= 5 else "0.0"
    if func == "is_late_night":
        hour = float(value.split(":")[0])
        return "1.0" if 0 <= hour < 2 else "0.0"
    if func == "is_early_morning":
        hour = float(value.split(":")[0])
        return "1.0" if 5 <= hour < 8 else "0.0"
    if func == "age_bin":
        days = float(value)
        if days <= 365:
            return "0.0"
        if days <= 730:
            return "1.0"
        if days <= 1095:
            return "2.0"
        if days <= 1825:
            return "3.0"
        return "4.0"
    raise ValueError(f"Unknown transform '{func}'.")



def _parse_date(s: str) -> datetime:
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date '{s}'. Expected DD/MM/YY or DD/MM/YYYY.")


def cast_value(raw: str, datatype: str) -> object:
    """Cast a raw string to the Python type matching a KServe v2 datatype."""
    if datatype in _FLOAT_TYPES:
        return float(raw)
    if datatype in _INT_TYPES:
        return int(float(raw))
    if datatype == "BOOL":
        return raw.strip().lower() in ("1", "true", "yes")
    return raw  # BYTES / STRING


def build_request(specs: list[TensorSpec], values: dict[str, object]) -> dict:
    """Assemble a KServe v2 inference request dict from tensor specs and values."""
    inputs = []
    for spec in specs:
        value = values.get(spec.name, _PLACEHOLDER.get(spec.datatype, 0))
        # Wrap scalars in a list; leave batches (list-of-lists) as-is
        if isinstance(value, list) and value and isinstance(value[0], list):
            data = value  # batch
            shape = [len(value)] + [len(value[0])]
        elif isinstance(value, list):
            data = value
            shape = [1, len(value)]
        else:
            data = [value]
            shape = [1, 1]

        inputs.append({
            "name": spec.name,
            "shape": shape,
            "datatype": spec.datatype,
            "data": data,
        })

    return {"inputs": inputs}


def parse_csv_arg(raw: str) -> "CsvSource":
    """Parse a ``FILE[:KEY=VAL]:COL=TENSOR,...`` argument into a CsvSource."""
    # Format: path[:key_col=key_val]:col1=tensor1,col2=tensor2
    parts = raw.split(":", 2)
    path = Path(parts[0])
    key_col: str | None = None
    key_val: str | None = None
    mapping_str: str = ""

    remaining = parts[1:]
    # A key filter looks like "COL=VAL" where VAL contains no commas (not a col map)
    for i, segment in enumerate(remaining):
        if "=" in segment and "," not in segment and not any(
            "=" in s for s in segment.split(",")[1:]
        ):
            # Treat as key=val filter only if it doesn't look like col=tensor pairs
            kv_parts = segment.split("=", 1)
            # Heuristic: if the right side looks like a tensor name (no spaces), it's a key filter
            # We accept it as a key filter when there is exactly one = and no commas
            key_col, key_val = kv_parts[0].strip(), kv_parts[1].strip()
        else:
            mapping_str = segment

    if not mapping_str and remaining:
        mapping_str = remaining[-1]

    col_map: dict[str, str] = {}
    for pair in mapping_str.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        csv_col, tensor_name = pair.split("=", 1)
        col_map[csv_col.strip()] = tensor_name.strip()

    return CsvSource(path=path, key_col=key_col, key_val=key_val, col_map=col_map)


def load_json_values(path: Path) -> dict[str, object]:
    """Load a ``{tensor_name: value}`` override file."""
    import json as _json
    text = path.read_text()
    data = _json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in '{path}', got {type(data).__name__}.")
    return data


def build_request_from_csv_file(
    pbtxt_path: Path,
    csv_path: Path,
    *,
    row_index: int = 0,
) -> tuple[str, dict]:
    """High-level helper: parse pbtxt, load one CSV row, return ``(model_name, request)``."""
    model_name, specs = parse_pbtxt(pbtxt_path)

    if len(specs) != 1:
        raise ValueError(
            f"build_request_from_csv_file requires a model with exactly one input tensor, "
            f"but '{model_name}' has {len(specs)}. Use build_request with explicit sources instead."
        )

    spec = specs[0]
    source = CsvSource(path=csv_path, key_col=None, key_val=None, col_map={}, flat_tensor=spec.name)
    loaded = load_csv_source(source, row_index)
    flat = loaded[spec.name]
    values = {spec.name: [cast_value(str(v), spec.datatype) for v in flat] if isinstance(flat, list) else cast_value(str(flat), spec.datatype)}
    return model_name, build_request(specs, values)


def iter_requests(
    pbtxt_path: Path,
    dataset_path: Path,
    *,
    batch_size: int = 1,
) -> Iterator[tuple[str, dict]]:
    """Yield ``(model_name, request)`` for each batch from *dataset_path*.

    Accepts three dataset formats:

    **JSON object** (``.json``, ``dict``) — a pre-built KServe v2 batch
    envelope::

        {
          "inputs": [{
            "name":     "<tensor_name>",
            "shape":    [N, F],
            "datatype": "<kserve_type>",
            "data":     [[...], ...]
          }]
        }

    The file is yielded as-is in a single iteration.  ``batch_size`` is
    ignored.  The model name is read from *pbtxt_path*; the tensor payload
    is taken verbatim.

    **JSON array** (``.json``, ``list``) — a sequence of individual KServe v2
    request envelopes, one per observation::

        [{"inputs": [...]}, {"inputs": [...]}, ...]

    Each element must itself be a valid KServe v2 envelope.  Elements are
    yielded one by one; ``batch_size`` is ignored (each element is already
    a self-contained request).

    **CSV** (``.csv``) — a dataset with one observation per row.

    Two CSV layouts are supported:

    *Multi-tensor* (column-per-tensor): the CSV header contains at least one
    tensor name from the model's ``config.pbtxt``.  Every matching column is
    mapped to its tensor; tensor order follows the pbtxt definition.  Each
    tensor receives ``shape=[N, 1]`` and ``data=[[v0], [v1], …]`` for a
    batch of N rows.  This supports models with any number of input tensors.

    *Flat-tensor* (all-columns-in-one-tensor): no CSV header column matches
    any tensor name.  All columns are packed into the single tensor as a flat
    row vector, producing ``shape=[N, cols]``.  This preserves backward
    compatibility for single-tensor models with arbitrary column names.

    In both CSV modes, rows are read eagerly, then sliced into batches of
    *batch_size*.  The final batch contains ``min(batch_size, remaining_rows)``
    rows — it is never padded or duplicated.

    In all cases each yielded value is a valid KServe v2 inference request
    dict ready for JSON serialisation.

    Args:
        pbtxt_path:   Path to the Triton ``config.pbtxt`` file.
        dataset_path: Path to a ``.json`` or ``.csv`` dataset file.
        batch_size:   Rows per yielded request for CSV datasets.  Must be
                      >= 1.  Ignored for JSON object and JSON array datasets.

    Yields:
        ``(model_name, kserve_v2_request_dict)`` — one per batch (CSV), once
        (JSON object), or one per element (JSON array).

    Raises:
        ValueError:        If ``batch_size < 1``; if the CSV is empty; if a
                           value cannot be cast to the tensor's datatype; or
                           if a JSON file is not a valid KServe v2 envelope.
        FileNotFoundError: If *pbtxt_path* or *dataset_path* does not exist.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    model_name, specs = parse_pbtxt(pbtxt_path)

    suffix = dataset_path.suffix.lower()

    # ------------------------------------------------------------------
    # JSON path — pre-built KServe v2 envelope(s), yielded as-is.
    # ------------------------------------------------------------------
    if suffix == ".json":
        try:
            doc = json.loads(dataset_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Dataset file is not valid JSON: {dataset_path}"
            ) from exc

        # --- JSON array: list of individual KServe v2 request envelopes ---
        if isinstance(doc, list):
            if not doc:
                raise ValueError(
                    f"JSON dataset '{dataset_path.name}' is an empty list. "
                    "Expected a non-empty array of KServe v2 envelopes."
                )
            for i, element in enumerate(doc):
                if not isinstance(element, dict) or not element.get("inputs"):
                    raise ValueError(
                        f"JSON dataset '{dataset_path.name}' element [{i}] "
                        "is not a valid KServe v2 envelope — "
                        "expected {\"inputs\": [{...}]}."
                    )
                yield model_name, element
            return

        # --- JSON object: single KServe v2 batch envelope ---
        inputs = doc.get("inputs")
        if not inputs or not isinstance(inputs, list):
            raise ValueError(
                f"JSON dataset '{dataset_path.name}' has no 'inputs' list. "
                "Expected a KServe v2 envelope "
                "({'inputs': [{'name': ..., 'shape': [...], 'datatype': ..., 'data': [...]}]})."
            )
        yield model_name, doc
        return

    # ------------------------------------------------------------------
    # CSV path — batch by row, driven by pbtxt tensor specs.
    # ------------------------------------------------------------------
    if suffix != ".csv":
        raise ValueError(
            f"Unsupported dataset file type: '{dataset_path.name}'. "
            "Expected a .json or .csv file."
        )

    reader = csv_mod.DictReader(io.StringIO(dataset_path.read_text()))
    csv_rows = list(reader)
    if not csv_rows:
        raise ValueError(f"CSV '{dataset_path.name}' contains no data rows.")

    csv_headers = list(csv_rows[0].keys())
    spec_names  = {s.name for s in specs}

    # Choose layout: multi-tensor when any CSV header matches a tensor name.
    use_multi_tensor = bool(spec_names & set(csv_headers))

    if use_multi_tensor:
        # Each CSV column matching a tensor name → its own tensor.
        # Missing columns get the type-appropriate placeholder value.
        for start in range(0, len(csv_rows), batch_size):
            chunk = csv_rows[start : start + batch_size]
            values: dict[str, object] = {}
            for spec in specs:
                if spec.name in csv_headers:
                    values[spec.name] = [
                        [cast_value(row[spec.name], spec.datatype)]
                        for row in chunk
                    ]
                # Absent columns fall through to build_request's placeholder.
            yield model_name, build_request(specs, values)
    else:
        # Flat-tensor fallback: all columns → single tensor as row vector.
        if len(specs) != 1:
            raise ValueError(
                f"CSV dataset '{dataset_path.name}' has no columns matching "
                f"any tensor name in '{model_name}'. For flat-tensor mode the "
                f"model must have exactly one input tensor, but '{model_name}' "
                f"has {len(specs)}. Rename CSV columns to match tensor names."
            )
        spec = specs[0]
        all_flat: list[list] = [
            [cast_value(v, spec.datatype) for v in row.values()]
            for row in csv_rows
        ]
        for start in range(0, len(all_flat), batch_size):
            chunk = all_flat[start : start + batch_size]
            yield model_name, build_request(specs, {spec.name: chunk})


def normalize_feature_name(name: str) -> str:
    """Normalise a feature name for tolerant comparison.

    Lowercases, treats underscores as spaces, collapses runs of whitespace, and
    trims.  Lets differently-cased/spaced spellings of the same feature match::

        Bank_Name  ->  "bank name"
        bank_name  ->  "bank name"
        Bank Name  ->  "bank name"
        USER_MEAN_AMOUNT / user_mean_amount  ->  "user mean amount"
    """
    return re.sub(r"\s+", " ", name.replace("_", " ")).strip().lower()


def humanize_feature_name(name: str) -> str:
    """Return a title-cased display label for a raw feature/column name.

    Normalises (see :func:`normalize_feature_name`) then capitalises each word::

        bank_name         ->  "Bank Name"
        USER_MEAN_AMOUNT  ->  "User Mean Amount"
        num_children      ->  "Num Children"
    """
    return " ".join(w.capitalize() for w in normalize_feature_name(name).split(" ") if w)


def read_csv_headers(path: Path) -> list[str]:
    """Return the ordered column headers of a CSV file.

    Reads only the header row.  Raises ``ValueError`` if the file has no rows.
    """
    reader = csv_mod.DictReader(io.StringIO(path.read_text()))
    if reader.fieldnames is None:
        raise ValueError(f"CSV '{path.name}' has no header row.")
    return list(reader.fieldnames)


def derive_input_name_mapping(
    csv_headers: list[str],
    input_schema_items: dict[str, dict],
    num_input_tensors: int,
) -> dict[str, str]:
    """Derive an ``{internal_name: display_name}`` input mapping from CSV headers.

    Given the CSV headers that produced a model's observations and TrustyAI's
    ``inputSchema.items`` (each value carries the feature's original ``name`` and
    its ``columnIndex`` in the flattened feature vector), return the input name
    mapping to apply — or ``{}`` when the layout is ambiguous and no safe mapping
    can be derived.

    Two rules, applied per schema item:

    Derived display names are title-cased via :func:`humanize_feature_name`
    (``bank_name`` → ``"Bank Name"``).  Explicit user-supplied mappings are never
    passed through here, so they keep their exact spelling.

    * **Name match** — if a schema item's internal name matches a CSV header
      after normalisation (see :func:`normalize_feature_name` — case-, underscore-
      and whitespace-insensitive), map it to that header's humanized label.  A
      schema name matches a header only when the request built that feature
      directly from that column (multi-tensor mode, where the tensor name *is* the
      header), so the mapping cannot mislabel.  Headers that normalise ambiguously
      (two distinct headers collapsing to the same normal form) are excluded from
      matching.  Covers multi-scalar models whose tensor names are meaningful.

    * **columnIndex positional derivation** — for any item left unmatched, map it
      to the humanized ``csv_headers[columnIndex]``.  Only safe for a model with
      **exactly one input tensor**: with more than one tensor the flattened
      ``columnIndex`` follows request (pbtxt) order, which need not match CSV
      header order, so a positional mapping could silently mislabel features.
      Covers the single flattened-tensor case (``<tensor>-0``…``<tensor>-N``).

    Returns ``{}`` (skip) when there are no headers or schema items.  Explicit
    user-supplied mappings and output mappings are handled by the caller — this
    function only ever derives *input* names.
    """
    if not csv_headers or not input_schema_items:
        return {}

    # Each schema item value carries the feature's ORIGINAL internal name and its
    # position in TrustyAI's flattened feature vector.
    items: list[tuple[str, object]] = []
    for meta in input_schema_items.values():
        internal = meta.get("name")
        if internal is None:
            return {}
        items.append((str(internal), meta.get("columnIndex")))

    header_set = set(csv_headers)

    # Map each normalised CSV header to its original spelling; a normal form
    # produced by two distinct headers is ambiguous and excluded from matching.
    norm_to_header: dict[str, str] = {}
    ambiguous: set[str] = set()
    for header in csv_headers:
        norm = normalize_feature_name(header)
        if norm in norm_to_header and norm_to_header[norm] != header:
            ambiguous.add(norm)
        norm_to_header.setdefault(norm, header)

    mapping: dict[str, str] = {}
    unmatched: list[tuple[str, object]] = []

    # Rule A — normalised-name match (maps to a title-cased header; safe for
    # every layout since a match means the feature came from that column).
    for internal, column_index in items:
        norm = normalize_feature_name(internal)
        if norm in norm_to_header and norm not in ambiguous:
            mapping[internal] = humanize_feature_name(norm_to_header[norm])
        else:
            unmatched.append((internal, column_index))

    if not unmatched:
        return mapping

    # Rule B — columnIndex derivation, guarded to single-tensor models.
    if num_input_tensors != 1:
        return mapping                                    # order not CSV-aligned
    if len(csv_headers) != len(header_set):
        return mapping                                    # duplicate headers
    if len(items) != len(csv_headers):
        return mapping                                    # feature-count mismatch
    indices = [column_index for _, column_index in items]
    if any(not isinstance(i, int) for i in indices):
        return mapping                                    # missing/invalid index
    int_indices = [i for i in indices if isinstance(i, int)]
    if sorted(int_indices) != list(range(len(csv_headers))):
        return mapping                                    # non-bijective / out of range

    for internal, column_index in unmatched:
        if isinstance(column_index, int):
            mapping[internal] = humanize_feature_name(csv_headers[column_index])
    return mapping


def generate_batched_requests(
    pbtxt_path: Path,
    dataset_path: Path,
    output_path: Path,
    *,
    batch_size: int = 1,
) -> int:
    """Write a JSON array of batched KServe v2 requests to *output_path*.

    Iterates over *dataset_path* in batches of *batch_size* using
    :func:`iter_requests` and collects every yielded request into a JSON
    array written to *output_path*::

        [
          {"inputs": [...]},   // batch 1
          {"inputs": [...]},   // batch 2
          ...
        ]

    Each element is a complete, valid KServe v2 inference request containing
    all model inputs defined in *pbtxt_path*.

    For a multi-tensor CSV dataset (column names matching tensor names), each
    element has one tensor per model input with ``shape=[N, 1]`` and N values
    from consecutive rows, where N == ``batch_size`` (or fewer for the final
    batch).

    For a JSON dataset the file content is yielded as-is (no re-batching);
    the output array contains exactly the same requests.

    Args:
        pbtxt_path:   Path to the Triton ``config.pbtxt`` file.
        dataset_path: Path to a ``.json`` or ``.csv`` dataset file.
        output_path:  Destination path for the output JSON array file.
                      Parent directories are created if necessary.
        batch_size:   Observations per request for CSV datasets.  Must be
                      >= 1.  Ignored for JSON datasets.

    Returns:
        Number of requests written to *output_path*.

    Raises:
        ValueError:        If ``batch_size < 1`` or the dataset is malformed.
        FileNotFoundError: If *pbtxt_path* or *dataset_path* does not exist.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    requests = [req for _, req in iter_requests(pbtxt_path, dataset_path, batch_size=batch_size)]
    output_path.write_text(json.dumps(requests, indent=2))
    return len(requests)
