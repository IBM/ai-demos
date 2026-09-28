"""Unit tests for rhoai.utils.yaml_io."""

from pathlib import Path

import pytest

from rhoai.utils.yaml_io import deep_merge, load, load_all


@pytest.fixture
def yaml_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.yaml"
    p.write_text("key: value\nnested:\n  a: 1\n")
    return p


@pytest.fixture
def multi_doc_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "multi.yaml"
    p.write_text("kind: Pod\n---\nkind: Service\n")
    return p


@pytest.fixture
def blank_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "blank.yaml"
    p.write_text("")
    return p


class TestLoad:
    def test_returns_dict(self, yaml_file: Path) -> None:
        result = load(yaml_file)
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_blank_file_returns_empty_dict(self, blank_yaml: Path) -> None:
        assert load(blank_yaml) == {}

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load(tmp_path / "nonexistent.yaml")


class TestLoadAll:
    def test_returns_list(self, multi_doc_yaml: Path) -> None:
        docs = load_all(multi_doc_yaml)
        assert len(docs) == 2
        assert docs[0] == {"kind": "Pod"}
        assert docs[1] == {"kind": "Service"}

    def test_skips_blank_documents(self, tmp_path: Path) -> None:
        p = tmp_path / "with_blank.yaml"
        p.write_text("kind: Pod\n---\n---\nkind: Service\n")
        docs = load_all(p)
        assert len(docs) == 2


class TestDeepMerge:
    def test_simple_override(self) -> None:
        result = deep_merge({"a": 1, "b": 2}, {"b": 99})
        assert result == {"a": 1, "b": 99}

    def test_nested_merge(self) -> None:
        base     = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"b": 99, "c": 3}}
        result   = deep_merge(base, override)
        assert result == {"outer": {"a": 1, "b": 99, "c": 3}}

    def test_non_dict_value_replaces(self) -> None:
        result = deep_merge({"x": {"y": 1}}, {"x": "scalar"})
        assert result == {"x": "scalar"}

    def test_neither_input_mutated(self) -> None:
        base     = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        deep_merge(base, override)
        assert base     == {"a": {"b": 1}}
        assert override == {"a": {"c": 2}}

    def test_new_keys_added(self) -> None:
        result = deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}
