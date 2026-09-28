"""Unit tests for rhoai.platform.prepare."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from rhoai.platform import prepare


@pytest.fixture(autouse=True)
def patch_resources(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setattr("rhoai.platform.prepare.resources", mock)
    return mock


# ---------------------------------------------------------------------------
# validate_login
# ---------------------------------------------------------------------------
class TestValidateLogin:
    def test_passes_when_cluster_reachable(self, patch_resources: MagicMock) -> None:
        patch_resources.get.return_value = {"metadata": {"name": "version"}}
        prepare.validate_login()  # should not raise

    def test_raises_when_cluster_unreachable(self, patch_resources: MagicMock) -> None:
        patch_resources.get.side_effect = OSError("no route to host")
        with pytest.raises(RuntimeError, match="oc login"):
            prepare.validate_login()


# ---------------------------------------------------------------------------
# validate_permissions
# ---------------------------------------------------------------------------
class TestValidatePermissions:
    def test_passes_when_allowed(self, patch_resources: MagicMock) -> None:
        patch_resources.create_dict.return_value = {"status": {"allowed": True}}
        prepare.validate_permissions("redhat-ods-operator")  # should not raise

    def test_raises_when_denied(self, patch_resources: MagicMock) -> None:
        patch_resources.create_dict.return_value = {"status": {"allowed": False}}
        with pytest.raises(RuntimeError, match="Insufficient permissions"):
            prepare.validate_permissions("redhat-ods-operator")

    def test_raises_when_apply_fails(self, patch_resources: MagicMock) -> None:
        patch_resources.create_dict.side_effect = RuntimeError("API error")
        with pytest.raises(RuntimeError, match="Permission check failed"):
            prepare.validate_permissions("redhat-ods-operator")


# ---------------------------------------------------------------------------
# validate_storage
# ---------------------------------------------------------------------------
class TestValidateStorage:
    def test_passes_when_specific_class_exists(self, patch_resources: MagicMock) -> None:
        patch_resources.exists.return_value = True
        prepare.validate_storage("gp3-csi")  # should not raise
        patch_resources.exists.assert_called_once_with("StorageClass", "gp3-csi")

    def test_raises_when_specific_class_missing(self, patch_resources: MagicMock) -> None:
        patch_resources.exists.return_value = False
        with pytest.raises(RuntimeError, match="gp3-csi"):
            prepare.validate_storage("gp3-csi")

    def test_passes_when_any_class_present(self, patch_resources: MagicMock) -> None:
        patch_resources.list_resources.return_value = [
            {"metadata": {"name": "standard"}}
        ]
        prepare.validate_storage("")  # empty → accept any

    def test_raises_when_no_classes_at_all(self, patch_resources: MagicMock) -> None:
        patch_resources.list_resources.return_value = []
        with pytest.raises(RuntimeError, match="No StorageClasses"):
            prepare.validate_storage("")


# ---------------------------------------------------------------------------
# validate_namespace
# ---------------------------------------------------------------------------
class TestValidateNamespace:
    def test_skips_creation_when_namespace_exists(self, patch_resources: MagicMock) -> None:
        patch_resources.exists.return_value = True
        prepare.validate_namespace("my-ns")
        patch_resources.apply_dict.assert_not_called()

    def test_creates_namespace_when_absent(self, patch_resources: MagicMock) -> None:
        patch_resources.exists.return_value = False
        prepare.validate_namespace("new-ns")
        call_kwargs = patch_resources.apply_dict.call_args[0][0]
        assert call_kwargs["kind"] == "Namespace"
        assert call_kwargs["metadata"]["name"] == "new-ns"


# ---------------------------------------------------------------------------
# get_cluster_info
# ---------------------------------------------------------------------------
def _make_node(name: str, roles: list[str], cpu: str, memory: str, gpu: int = 0) -> dict[str, Any]:
    labels = {f"node-role.kubernetes.io/{r}": "" for r in roles}
    capacity: dict[str, Any] = {"cpu": cpu, "memory": memory}
    if gpu:
        capacity["nvidia.com/gpu"] = str(gpu)
    return {"metadata": {"name": name, "labels": labels}, "status": {"capacity": capacity}}


def _make_pv(storage_class: str, capacity: str, phase: str) -> dict[str, Any]:
    return {
        "spec": {"storageClassName": storage_class, "capacity": {"storage": capacity}},
        "status": {"phase": phase},
    }


def _make_cv(version: str) -> dict[str, Any]:
    return {"status": {"desired": {"version": version}}}


class TestGetClusterInfo:
    def _setup(
        self,
        patch_resources: MagicMock,
        nodes: list[dict],
        pvs: list[dict],
        classes: list[str],
        version: str = "4.15.0",
    ) -> None:
        cv = _make_cv(version)
        sc_list = [{"metadata": {"name": c}} for c in classes]

        def list_resources_side_effect(kind: str, *args, **kwargs):
            if kind == "Node":
                return nodes
            if kind == "StorageClass":
                return sc_list
            if kind == "PersistentVolume":
                return pvs
            return []

        patch_resources.get.return_value = cv
        patch_resources.list_resources.side_effect = list_resources_side_effect

    def test_multi_node_topology(self, patch_resources: MagicMock) -> None:
        nodes = [
            _make_node("master-0", ["master"], "4", "8Gi"),
            _make_node("worker-0", ["worker"], "8", "32Gi"),
            _make_node("worker-1", ["worker"], "8", "32Gi"),
        ]
        self._setup(patch_resources, nodes, [], ["gp3-csi"])
        info = prepare.get_cluster_info()
        assert info["topology"] == "Multi-node"
        assert info["node_count"] == 3
        assert info["worker_count"] == 2

    def test_sno_topology_single_node(self, patch_resources: MagicMock) -> None:
        nodes = [_make_node("sno-0", ["master", "worker"], "16", "64Gi")]
        self._setup(patch_resources, nodes, [], [])
        info = prepare.get_cluster_info()
        assert info["topology"] == "SNO"
        assert info["node_count"] == 1
        assert info["worker_count"] == 1

    def test_worker_node_cpu_and_memory(self, patch_resources: MagicMock) -> None:
        nodes = [_make_node("worker-0", ["worker"], "8", "32Gi")]
        self._setup(patch_resources, nodes, [], [])
        info = prepare.get_cluster_info()
        node = info["worker_nodes"][0]
        assert node["name"] == "worker-0"
        assert node["cpu"] == "8 cores"
        assert node["memory"] == "32 GiB"

    def test_worker_node_gpu_detected(self, patch_resources: MagicMock) -> None:
        nodes = [_make_node("gpu-worker", ["worker"], "32", "128Gi", gpu=4)]
        self._setup(patch_resources, nodes, [], [])
        info = prepare.get_cluster_info()
        assert info["worker_nodes"][0]["gpu"] == 4

    def test_worker_node_no_gpu(self, patch_resources: MagicMock) -> None:
        nodes = [_make_node("worker-0", ["worker"], "8", "32Gi")]
        self._setup(patch_resources, nodes, [], [])
        info = prepare.get_cluster_info()
        assert info["worker_nodes"][0]["gpu"] == 0

    def test_pv_bound_counted_as_used(self, patch_resources: MagicMock) -> None:
        nodes = [_make_node("worker-0", ["worker"], "4", "8Gi")]
        pvs = [_make_pv("gp3-csi", "10Gi", "Bound")]
        self._setup(patch_resources, nodes, pvs, ["gp3-csi"])
        info = prepare.get_cluster_info()
        assert info["storage_summary"]["gp3-csi"]["used"] == "10 GiB"

    def test_pv_available_not_counted_as_used(self, patch_resources: MagicMock) -> None:
        nodes = [_make_node("worker-0", ["worker"], "4", "8Gi")]
        pvs = [_make_pv("gp3-csi", "10Gi", "Available")]
        self._setup(patch_resources, nodes, pvs, ["gp3-csi"])
        info = prepare.get_cluster_info()
        assert info["storage_summary"]["gp3-csi"]["used"] == "0 GiB"

    def test_openshift_version_returned(self, patch_resources: MagicMock) -> None:
        nodes = [_make_node("worker-0", ["worker"], "4", "8Gi")]
        self._setup(patch_resources, nodes, [], [], version="4.17.3")
        info = prepare.get_cluster_info()
        assert info["openshift_version"] == "4.17.3"

    def test_cpu_millicores_parsed_correctly(self, patch_resources: MagicMock) -> None:
        # "4000m" = 4 cores
        nodes = [_make_node("worker-0", ["worker"], "4000m", "8Gi")]
        self._setup(patch_resources, nodes, [], [])
        info = prepare.get_cluster_info()
        assert info["worker_nodes"][0]["cpu"] == "4 cores"


# ---------------------------------------------------------------------------
# init_platform / install_component / bootstrap_platform
# ---------------------------------------------------------------------------
def _fresh_prepare_module():
    """Reimport rhoai.platform.prepare so monkeypatched lazy imports take hold.

    Also ensures sibling platform modules (dsc, operators, manifests) are
    pre-loaded in sys.modules so monkeypatch.setattr("rhoai.platform.dsc", ...)
    works even for functions that import them lazily.
    """
    import importlib
    import sys

    import rhoai.platform.dsc  # noqa: F401 — side-effect: loads into sys.modules
    import rhoai.platform.manifests  # noqa: F401
    import rhoai.platform.operators  # noqa: F401
    sys.modules.pop("rhoai.platform.prepare", None)
    return importlib.import_module("rhoai.platform.prepare")


def _base_config() -> dict[str, Any]:
    return {
        "repo_root": "/repo",
        "operator": {
            "name": "rhods-operator",
            "namespace": "redhat-ods-operator",
            "channel": "stable",
        },
        "dsc":      {"name": "default-dsc", "dsci_name": "default-dsci"},
        "timeouts": {"operator_ready": 300, "dsc_ready": 600},
    }


class TestInitPlatform:
    def test_calls_prepare_then_operator_then_dsci(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """init_platform must call prepare_platform, then operator, then DSCI — in order.

        Must NOT touch the DataScienceCluster — that's install_component's job.
        """
        prep_mod = _fresh_prepare_module()

        prepare_mock   = MagicMock()
        operators_mock = MagicMock()
        operators_mock.is_installed.return_value = True
        dsc_mock       = MagicMock()
        manifests_mock = MagicMock()

        monkeypatch.setattr(prep_mod, "prepare_platform", prepare_mock)
        monkeypatch.setattr("rhoai.platform.operators", operators_mock)
        monkeypatch.setattr("rhoai.platform.dsc",       dsc_mock)
        monkeypatch.setattr("rhoai.platform.manifests", manifests_mock)

        prep_mod.init_platform(_base_config())

        prepare_mock.assert_called_once()
        operators_mock.is_installed.assert_called_once()
        dsc_mock.apply_dsci.assert_called_once()
        dsc_mock.wait_dsci_ready.assert_called_once()
        dsc_mock.apply_dsc.assert_not_called()

    def test_installs_operator_when_not_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prep_mod = _fresh_prepare_module()

        monkeypatch.setattr(prep_mod, "prepare_platform", MagicMock())
        operators_mock = MagicMock()
        operators_mock.is_installed.return_value = False
        monkeypatch.setattr("rhoai.platform.operators", operators_mock)
        monkeypatch.setattr("rhoai.platform.dsc",       MagicMock())
        monkeypatch.setattr("rhoai.platform.manifests", MagicMock())

        prep_mod.init_platform(_base_config())

        operators_mock.install.assert_called_once()
        operators_mock.wait_until_ready.assert_not_called()

    def test_waits_for_existing_operator_when_already_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prep_mod = _fresh_prepare_module()

        monkeypatch.setattr(prep_mod, "prepare_platform", MagicMock())
        operators_mock = MagicMock()
        operators_mock.is_installed.return_value = True
        monkeypatch.setattr("rhoai.platform.operators", operators_mock)
        monkeypatch.setattr("rhoai.platform.dsc",       MagicMock())
        monkeypatch.setattr("rhoai.platform.manifests", MagicMock())

        prep_mod.init_platform(_base_config())

        operators_mock.install.assert_not_called()
        operators_mock.wait_until_ready.assert_called_once()

    def test_passes_version_to_install_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When config['operator']['version'] is set, install() must receive it."""
        prep_mod = _fresh_prepare_module()

        monkeypatch.setattr(prep_mod, "prepare_platform", MagicMock())
        operators_mock = MagicMock()
        operators_mock.is_installed.return_value = False
        monkeypatch.setattr("rhoai.platform.operators", operators_mock)
        monkeypatch.setattr("rhoai.platform.dsc",       MagicMock())
        monkeypatch.setattr("rhoai.platform.manifests", MagicMock())

        config = _base_config()
        config["operator"]["version"] = "rhods-operator.v3.5.0"
        prep_mod.init_platform(config)

        operators_mock.install.assert_called_once()
        _, kwargs = operators_mock.install.call_args
        assert kwargs.get("version") == "rhods-operator.v3.5.0"

    def test_passes_empty_version_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When version is absent from config, install() receives version=''."""
        prep_mod = _fresh_prepare_module()

        monkeypatch.setattr(prep_mod, "prepare_platform", MagicMock())
        operators_mock = MagicMock()
        operators_mock.is_installed.return_value = False
        monkeypatch.setattr("rhoai.platform.operators", operators_mock)
        monkeypatch.setattr("rhoai.platform.dsc",       MagicMock())
        monkeypatch.setattr("rhoai.platform.manifests", MagicMock())

        prep_mod.init_platform(_base_config())

        operators_mock.install.assert_called_once()
        _, kwargs = operators_mock.install.call_args
        assert kwargs.get("version", "") == ""


class TestInstallComponent:
    def test_raises_when_operator_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prep_mod = _fresh_prepare_module()

        operators_mock = MagicMock()
        operators_mock.is_installed.return_value = False
        monkeypatch.setattr("rhoai.platform.operators", operators_mock)
        monkeypatch.setattr("rhoai.platform.dsc",       MagicMock())
        monkeypatch.setattr("rhoai.platform.manifests", MagicMock())
        monkeypatch.setattr("rhoai.ocp.resources",      MagicMock())

        with pytest.raises(RuntimeError, match="platform init"):
            prep_mod.install_component(_base_config(), ["kserve"])

    def test_raises_when_dsci_not_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prep_mod = _fresh_prepare_module()

        operators_mock = MagicMock()
        operators_mock.is_installed.return_value = True
        dsc_mock = MagicMock()
        err_msg = "DSCInitialization 'default-dsci' is not ready"
        dsc_mock.verify_dsci.side_effect = RuntimeError(err_msg)
        monkeypatch.setattr("rhoai.platform.operators", operators_mock)
        monkeypatch.setattr("rhoai.platform.dsc",       dsc_mock)
        monkeypatch.setattr("rhoai.platform.manifests", MagicMock())
        monkeypatch.setattr("rhoai.ocp.resources",      MagicMock())

        with pytest.raises(RuntimeError, match="platform init"):
            prep_mod.install_component(_base_config(), ["kserve"])

    def test_creates_dsc_from_base_manifest_when_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no DSC exists yet, install_component bootstraps it from the base
        manifest before patching — so component fields are always schema-valid.
        """
        prep_mod = _fresh_prepare_module()

        operators_mock = MagicMock()
        operators_mock.is_installed.return_value = True
        dsc_mock = MagicMock()
        manifests_mock = MagicMock()
        resources_mock = MagicMock()
        resources_mock.exists.return_value = False

        monkeypatch.setattr("rhoai.platform.operators", operators_mock)
        monkeypatch.setattr("rhoai.platform.dsc",       dsc_mock)
        monkeypatch.setattr("rhoai.platform.manifests", manifests_mock)
        monkeypatch.setattr("rhoai.ocp.resources",      resources_mock)
        monkeypatch.setattr(prep_mod, "resources",      resources_mock)

        prep_mod.install_component(_base_config(), ["kserve", "trustyai"])

        dsc_mock.apply_dsc.assert_called_once()
        dsc_mock.set_component_states.assert_called_once_with(
            "default-dsc", {"kserve": "Managed", "trustyai": "Managed"}
        )
        dsc_mock.wait_until_ready.assert_called_once()

    def test_does_not_recreate_dsc_when_already_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When a DSC already exists, install_component must only patch — it must
        never re-apply the full manifest, which would reset unrelated components.
        """
        prep_mod = _fresh_prepare_module()

        operators_mock = MagicMock()
        operators_mock.is_installed.return_value = True
        dsc_mock = MagicMock()
        resources_mock = MagicMock()
        resources_mock.exists.return_value = True

        monkeypatch.setattr("rhoai.platform.operators", operators_mock)
        monkeypatch.setattr("rhoai.platform.dsc",       dsc_mock)
        monkeypatch.setattr("rhoai.platform.manifests", MagicMock())
        monkeypatch.setattr("rhoai.ocp.resources",      resources_mock)
        monkeypatch.setattr(prep_mod, "resources",      resources_mock)

        prep_mod.install_component(_base_config(), ["trustyai"])

        dsc_mock.apply_dsc.assert_not_called()
        dsc_mock.set_component_states.assert_called_once_with(
            "default-dsc", {"trustyai": "Managed"}
        )


    def test_raises_for_unknown_component(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """install_component must reject unknown component names immediately,
        before any cluster calls, with a clear error listing valid names.
        """
        import typer as _typer

        prep_mod = _fresh_prepare_module()

        with pytest.raises(_typer.BadParameter, match="unknown component"):
            prep_mod.install_component(_base_config(), ["kserv"])

    def test_raises_for_multiple_unknown_components(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import typer as _typer

        prep_mod = _fresh_prepare_module()

        with pytest.raises(_typer.BadParameter, match="unknown component"):
            prep_mod.install_component(_base_config(), ["kserv", "notacomponent"])

    def test_valid_component_names_pass_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All names in VALID_COMPONENTS must be accepted without error."""
        from rhoai.platform.prepare import VALID_COMPONENTS

        prep_mod = _fresh_prepare_module()

        operators_mock = MagicMock()
        operators_mock.is_installed.return_value = True
        dsc_mock = MagicMock()
        resources_mock = MagicMock()
        resources_mock.exists.return_value = True

        monkeypatch.setattr("rhoai.platform.operators", operators_mock)
        monkeypatch.setattr("rhoai.platform.dsc",       dsc_mock)
        monkeypatch.setattr("rhoai.platform.manifests", MagicMock())
        monkeypatch.setattr("rhoai.ocp.resources",      resources_mock)
        monkeypatch.setattr(prep_mod, "resources",      resources_mock)

        # Every valid name must not raise BadParameter
        for name in VALID_COMPONENTS:
            prep_mod.install_component(_base_config(), [name])


class TestPlatformNeedsReconciliation:
    """Unit tests for the read-only platform_needs_reconciliation() helper."""

    def _mock_platform(
        self,
        monkeypatch: pytest.MonkeyPatch,
        operator_ready: bool = True,
        dsci_ready: bool = True,
        dsc_ready: bool = True,
        component_states: dict | None = None,
    ) -> None:
        prep_mod = _fresh_prepare_module()
        operators_mock = MagicMock()
        operators_mock.is_installed.return_value = operator_ready
        dsc_mock = MagicMock()
        dsc_mock.is_dsci_ready.return_value = dsci_ready
        dsc_mock.is_dsc_ready.return_value  = dsc_ready
        dsc_mock.get_component_states.return_value = component_states or {}
        monkeypatch.setattr("rhoai.platform.operators", operators_mock)
        monkeypatch.setattr("rhoai.platform.dsc",       dsc_mock)
        return prep_mod

    def test_returns_false_when_platform_fully_ready_no_components(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No specific components requested and all resources healthy → no reconciliation."""
        prep_mod = self._mock_platform(monkeypatch)
        assert prep_mod.platform_needs_reconciliation(_base_config()) is False

    def test_returns_false_when_all_components_already_managed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prep_mod = self._mock_platform(
            monkeypatch,
            component_states={"kserve": "Managed", "trustyai": "Managed"},
        )
        config = _base_config()
        config["components"] = ["kserve", "trustyai"]
        assert prep_mod.platform_needs_reconciliation(config) is False

    def test_returns_true_when_operator_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prep_mod = self._mock_platform(monkeypatch, operator_ready=False)
        assert prep_mod.platform_needs_reconciliation(_base_config()) is True

    def test_returns_true_when_dsci_not_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prep_mod = self._mock_platform(monkeypatch, dsci_ready=False)
        assert prep_mod.platform_needs_reconciliation(_base_config()) is True

    def test_returns_true_when_dsc_not_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prep_mod = self._mock_platform(monkeypatch, dsc_ready=False)
        assert prep_mod.platform_needs_reconciliation(_base_config()) is True

    def test_returns_true_when_component_not_yet_managed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One component Removed → reconciliation needed."""
        prep_mod = self._mock_platform(
            monkeypatch,
            component_states={"kserve": "Managed", "trustyai": "Removed"},
        )
        config = _base_config()
        config["components"] = ["kserve", "trustyai"]
        assert prep_mod.platform_needs_reconciliation(config) is True

    def test_returns_true_when_component_missing_from_dsc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Component not present in DSC at all → reconciliation needed."""
        prep_mod = self._mock_platform(
            monkeypatch,
            component_states={"kserve": "Managed"},  # trustyai absent
        )
        config = _base_config()
        config["components"] = ["kserve", "trustyai"]
        assert prep_mod.platform_needs_reconciliation(config) is True

    def test_returns_true_when_get_component_states_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the DSC cannot be read, conservatively require reconciliation."""
        prep_mod = _fresh_prepare_module()
        operators_mock = MagicMock()
        operators_mock.is_installed.return_value = True
        dsc_mock = MagicMock()
        dsc_mock.is_dsci_ready.return_value = True
        dsc_mock.is_dsc_ready.return_value  = True
        dsc_mock.get_component_states.side_effect = RuntimeError("API error")
        monkeypatch.setattr("rhoai.platform.operators", operators_mock)
        monkeypatch.setattr("rhoai.platform.dsc",       dsc_mock)
        config = _base_config()
        config["components"] = ["kserve"]
        assert prep_mod.platform_needs_reconciliation(config) is True


class TestBootstrapPlatform:
    def _force_reconciliation(self, monkeypatch: pytest.MonkeyPatch, prep_mod: Any) -> None:
        """Pass _needs_reconciliation=True so bootstrap proceeds without an internal check."""
        # No need to patch platform_needs_reconciliation — pass the value directly.

    def test_applies_full_dsc_manifest_when_components_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When config has no components list, apply the full base DSC manifest."""
        prep_mod = _fresh_prepare_module()

        init_mock      = MagicMock()
        dsc_mock       = MagicMock()
        manifests_mock = MagicMock()

        monkeypatch.setattr(prep_mod, "init_platform", init_mock)
        monkeypatch.setattr("rhoai.platform.dsc",       dsc_mock)
        monkeypatch.setattr("rhoai.platform.manifests", manifests_mock)

        config = _base_config()  # no "components" key → defaults to empty
        prep_mod.bootstrap_platform(config, _needs_reconciliation=True)

        init_mock.assert_called_once_with(config)
        dsc_mock.apply_dsc.assert_called_once()
        dsc_mock.wait_until_ready.assert_called_once()

    def test_uses_install_component_when_components_set_in_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When config['components'] is a non-empty list, delegate to install_component
        instead of applying the full manifest — enabling only the listed components."""
        prep_mod = _fresh_prepare_module()

        init_mock            = MagicMock()
        install_mock         = MagicMock()
        dsc_mock             = MagicMock()

        monkeypatch.setattr(prep_mod, "init_platform",      init_mock)
        monkeypatch.setattr(prep_mod, "install_component",  install_mock)
        monkeypatch.setattr("rhoai.platform.dsc",           dsc_mock)
        monkeypatch.setattr("rhoai.platform.manifests",     MagicMock())

        config = _base_config()
        config["components"] = ["kserve", "trustyai"]
        prep_mod.bootstrap_platform(config, _needs_reconciliation=True)

        init_mock.assert_called_once_with(config)
        install_mock.assert_called_once_with(config, ["kserve", "trustyai"])
        # Full manifest must NOT be applied when components are explicit
        dsc_mock.apply_dsc.assert_not_called()

    def test_skips_reconciliation_when_platform_already_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fast-path: when _needs_reconciliation=False, nothing is written."""
        prep_mod = _fresh_prepare_module()

        init_mock = MagicMock()
        dsc_mock  = MagicMock()
        monkeypatch.setattr(prep_mod, "init_platform", init_mock)
        monkeypatch.setattr("rhoai.platform.dsc",      dsc_mock)

        prep_mod.bootstrap_platform(_base_config(), _needs_reconciliation=False)

        init_mock.assert_not_called()
        dsc_mock.apply_dsc.assert_not_called()
        dsc_mock.set_component_states.assert_not_called()
        dsc_mock.wait_until_ready.assert_not_called()

    def test_skips_reconciliation_via_internal_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no pre-computed value is passed, the internal check runs."""
        prep_mod = _fresh_prepare_module()
        monkeypatch.setattr(prep_mod, "platform_needs_reconciliation", lambda _: False)

        init_mock = MagicMock()
        monkeypatch.setattr(prep_mod, "init_platform", init_mock)

        prep_mod.bootstrap_platform(_base_config())  # no _needs_reconciliation arg

        init_mock.assert_not_called()

    def test_deploy_platform_alias_still_works(self) -> None:
        """Backward-compatible alias for any external callers."""
        assert prepare.deploy_platform is prepare.bootstrap_platform


# ---------------------------------------------------------------------------
# _parse_cpu / _parse_memory_ki helpers
# ---------------------------------------------------------------------------
class TestParseCpu:
    def test_integer_cores(self) -> None:
        assert prepare._parse_cpu("4") == 4000

    def test_millicores(self) -> None:
        assert prepare._parse_cpu("4000m") == 4000

    def test_single_core(self) -> None:
        assert prepare._parse_cpu("1") == 1000


class TestParseMemoryKi:
    def test_kibibytes(self) -> None:
        assert prepare._parse_memory_ki("1024Ki") == 1024

    def test_mebibytes(self) -> None:
        assert prepare._parse_memory_ki("1Mi") == 1024

    def test_gibibytes(self) -> None:
        assert prepare._parse_memory_ki("1Gi") == 1024 * 1024

    def test_tebibytes(self) -> None:
        assert prepare._parse_memory_ki("1Ti") == 1024 * 1024 * 1024

    def test_plain_bytes(self) -> None:
        # 2048 bytes = 2 KiB
        assert prepare._parse_memory_ki("2048") == 2
