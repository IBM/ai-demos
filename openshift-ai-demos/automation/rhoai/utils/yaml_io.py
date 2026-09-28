"""YAML file helpers.

Loads YAML files and merges configuration dicts.
No knowledge of manifests or asset paths — those live in platform/manifests.py
and usecases/*/assets.py.
"""

from pathlib import Path
from typing import Any

import yaml


def load(path: Path) -> dict[str, Any]:
    """Load a single-document YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed content as a dict, or empty dict if the file is blank.
    """
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def load_all(path: Path) -> list[dict[str, Any]]:
    """Load a multi-document YAML file (--- separated).

    Args:
        path: Path to the YAML file.

    Returns:
        List of parsed documents, skipping blank documents.
    """
    with open(path) as fh:
        return [doc for doc in yaml.safe_load_all(fh) if doc]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base and return a new dict.

    Nested dicts are merged rather than replaced. Neither input is modified.

    Args:
        base:     Base values (e.g. from defaults.yaml).
        override: Values to apply on top (e.g. from user config file).
    """
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
