"""Shared manifest path resolver for platform-level resources.

Maps logical names to file paths inside the openshift-ai-demos repo.
If the repo is reorganised, only this file needs to change.
Imports stdlib only.
"""

from pathlib import Path


def get_dsci(repo_root: str | Path) -> Path:
    return Path(repo_root) / "shared" / "dsci.yaml"


def get_dsc(repo_root: str | Path) -> Path:
    return Path(repo_root) / "shared" / "dsc.yaml"


def get_s3_secret(repo_root: str | Path) -> Path:
    return Path(repo_root) / "shared" / "s3-secret.yaml"


def get_operator_group(repo_root: str | Path) -> Path:
    return Path(repo_root) / "shared" / "operator-group.yaml"


def get_subscription(repo_root: str | Path) -> Path:
    return Path(repo_root) / "shared" / "subscription.yaml"


def get_triton_serving_runtime_template(repo_root: str | Path) -> Path:
    return (
        Path(repo_root)
        / "model-serving"
        / "predictive-models"
        / "triton"
        / "triton-ppc64le-runtime-template.yaml"
    )


# ---------------------------------------------------------------------------
# TrustyAI service manifests
# ---------------------------------------------------------------------------

def get_trustyai_monitoring_config(repo_root: str | Path) -> Path:
    """Return the path to the monitoring ConfigMaps manifest.

    Contains cluster-monitoring-config and user-workload-monitoring-config.

    Returns:
        Path to trustyai/service/monitoring-config.yaml
    """
    return Path(repo_root) / "trustyai" / "service" / "monitoring-config.yaml"


def get_trustyai_rbac(repo_root: str | Path) -> Path:
    """Return the path to the TrustyAI ServiceAccount + RoleBinding manifest.

    Returns:
        Path to trustyai/service/rbac.yaml
    """
    return Path(repo_root) / "trustyai" / "service" / "rbac.yaml"


def get_trustyai_service(repo_root: str | Path) -> Path:
    """Return the path to the TrustyAIService CR manifest.

    Returns:
        Path to trustyai/service/trustyai-service.yaml
    """
    return Path(repo_root) / "trustyai" / "service" / "trustyai-service.yaml"


def get_trustyai_logger_ca_bundle(repo_root: str | Path) -> Path:
    """Return the path to the kserve-logger-ca-bundle ConfigMap manifest.

    Returns:
        Path to trustyai/service/kserve-logger-ca-bundle.yaml
    """
    return Path(repo_root) / "trustyai" / "service" / "kserve-logger-ca-bundle.yaml"
