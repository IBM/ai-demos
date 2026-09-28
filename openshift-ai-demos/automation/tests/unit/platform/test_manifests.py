"""Unit tests for rhoai.platform.manifests."""

from pathlib import Path

import pytest

from rhoai.platform import manifests


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    return tmp_path


class TestSharedManifests:
    def test_get_dsci(self, root: Path) -> None:
        assert manifests.get_dsci(root) == root / "shared" / "dsci.yaml"

    def test_get_dsc(self, root: Path) -> None:
        assert manifests.get_dsc(root) == root / "shared" / "dsc.yaml"

    def test_get_s3_secret(self, root: Path) -> None:
        assert manifests.get_s3_secret(root) == root / "shared" / "s3-secret.yaml"

    def test_get_operator_group(self, root: Path) -> None:
        assert manifests.get_operator_group(root) == root / "shared" / "operator-group.yaml"

    def test_get_subscription(self, root: Path) -> None:
        assert manifests.get_subscription(root) == root / "shared" / "subscription.yaml"

    def test_get_triton_serving_runtime_template(self, root: Path) -> None:
        assert manifests.get_triton_serving_runtime_template(root) == (
            root
            / "model-serving"
            / "predictive-models"
            / "triton"
            / "triton-ppc64le-runtime-template.yaml"
        )


class TestTrustyAIManifests:
    def test_get_trustyai_monitoring_config(self, root: Path) -> None:
        assert manifests.get_trustyai_monitoring_config(root) == (
            root / "trustyai" / "service" / "monitoring-config.yaml"
        )

    def test_get_trustyai_rbac(self, root: Path) -> None:
        assert manifests.get_trustyai_rbac(root) == (
            root / "trustyai" / "service" / "rbac.yaml"
        )

    def test_get_trustyai_service(self, root: Path) -> None:
        assert manifests.get_trustyai_service(root) == (
            root / "trustyai" / "service" / "trustyai-service.yaml"
        )

    def test_get_trustyai_logger_ca_bundle(self, root: Path) -> None:
        assert manifests.get_trustyai_logger_ca_bundle(root) == (
            root / "trustyai" / "service" / "kserve-logger-ca-bundle.yaml"
        )
