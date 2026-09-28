"""Unit tests for rhoai.config.loader."""

from pathlib import Path

import pytest

from rhoai.config.loader import load_config


@pytest.fixture
def minimal_config(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text("platform:\n  namespace: my-namespace\n")
    return p


@pytest.fixture
def defaults_only_config(tmp_path: Path) -> Path:
    """A config file that specifies nothing — all defaults apply."""
    p = tmp_path / "empty.yaml"
    p.write_text("{}\n")
    return p


class TestLoadConfig:
    def test_returns_dict(self) -> None:
        config = load_config()
        assert isinstance(config, dict)

    def test_defaults_present(self) -> None:
        config = load_config()
        assert config["log_level"] == "INFO"
        assert config["operator"]["name"] == "rhods-operator"
        assert config["dsc"]["name"] == "default-dsc"

    def test_user_file_overrides_defaults(self, minimal_config: Path) -> None:
        config = load_config(minimal_config)
        assert config["platform"]["namespace"] == "my-namespace"

    def test_user_file_preserves_unset_defaults(self, minimal_config: Path) -> None:
        config = load_config(minimal_config)
        assert config["operator"]["name"] == "rhods-operator"

    def test_no_path_returns_defaults_only(self) -> None:
        config = load_config()
        assert config["platform"]["namespace"] == "redhat-ods-applications"

    def test_empty_config_file_returns_defaults(self, defaults_only_config: Path) -> None:
        config = load_config(defaults_only_config)
        assert config["operator"]["name"] == "rhods-operator"
