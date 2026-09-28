"""Use case registry.

Maps CLI names to use-case modules. The CLI calls registry.get(name)
and then calls deploy(), verify(), or cleanup() on the returned module.

To add a new use case:
    1. Create usecases/<name>/ with deploy.py, verify.py, cleanup.py.
    2. Add one entry to REGISTRY below.
    3. Done — no other file needs to change.

No lazy loading, no plugin discovery, no base classes.
"""

from types import ModuleType

from rhoai.usecases import fraud_detection

REGISTRY: dict[str, ModuleType] = {
    "fraud-detection": fraud_detection,
}


def get(name: str) -> ModuleType:
    """Return the module for the named use case.

    Args:
        name: Use case name as typed on the CLI, e.g. "fraud-detection".

    Returns:
        Module exposing deploy(config), verify(config), cleanup(config).

    Raises:
        RuntimeError: If the name is not registered.
    """
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise RuntimeError(f"Unknown use case '{name}'. Available: {available}")
    return REGISTRY[name]


def list_available() -> list[str]:
    """Return a sorted list of registered use case names."""
    return sorted(REGISTRY)
