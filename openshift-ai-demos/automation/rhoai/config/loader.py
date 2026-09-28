"""Deployment configuration loader.

Loads the YAML config file and returns a plain dict merged on top of
the bundled defaults.  Pydantic can be added later if config validation
becomes complex enough to justify it.

Priority (highest → lowest):
    1. CLI flags          (applied by each command after load_config returns)
    2. User YAML file     (--config / path argument)
    3. Bundled defaults   (config/defaults.yaml)

Depends on: utils.yaml_io, utils.logger, stdlib
"""

from pathlib import Path
from typing import Any

import yaml

from rhoai.utils.logger import get_logger
from rhoai.utils.yaml_io import deep_merge, load

log = get_logger(__name__)

_DEFAULTS = Path(__file__).parent / "defaults.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load, merge, and return the deployment configuration.

    Args:
        path: Path to the user YAML config file.  If None, only the
              bundled defaults apply.

    Returns:
        Merged configuration as a plain dict.
    """
    config = load(_DEFAULTS)

    if path:
        log.debug("Loading config from %s", path)
        config = deep_merge(config, load(path))

    return config


def save_defaults(updates: dict[str, Any]) -> None:
    """Persist key-value updates into defaults.yaml so future bare runs use them.

    Only the keys present in *updates* are changed — every other value in
    defaults.yaml is left exactly as-is.  The file is written atomically.

    Args:
        updates: Flat or nested dict matching the structure of defaults.yaml.
                 Example: ``{"operator": {"channel": "stable-3.5",
                                          "version": "rhods-operator.v3.5.0"}}``
    """
    current = load(_DEFAULTS)
    merged  = deep_merge(current, updates)
    with open(_DEFAULTS, "w") as fh:
        yaml.dump(merged, fh, default_flow_style=False, sort_keys=False)
    log.debug("defaults.yaml updated: %s", updates)
