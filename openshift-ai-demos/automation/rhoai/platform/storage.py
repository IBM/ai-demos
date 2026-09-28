"""Storage configuration — S3 credentials, Secrets, PVCs, and file staging."""

import base64
import subprocess
import time
from pathlib import Path
from typing import Any

from rhoai.ocp import resources, wait
from rhoai.utils.logger import get_logger

log = get_logger(__name__)

# Prefix for the short-lived staging Pod created when copying files onto a PVC.
_STAGING_POD_PREFIX = "pvc-stage"

# Kubernetes Pod name limit.
_K8S_NAME_MAX = 63

# Seconds to poll while waiting for the staging Pod to become Running.
_STAGING_POLL_INTERVAL = 5


def apply_s3_secret(manifest_path: Path, namespace: str) -> None:
    """Apply the S3 credentials Secret manifest. Idempotent."""
    log.info("Applying S3 credentials from %s", manifest_path.name)
    resources.apply_manifest(manifest_path, namespace)


def create_secret(
    name: str, namespace: str, data: dict[str, str], secret_type: str = "Opaque"
) -> None:
    """Create or update a Secret. Values are plain strings; base64 is applied here."""
    log.info("Creating Secret '%s' in '%s'", name, namespace)
    resources.apply_dict({
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": name, "namespace": namespace},
        "type": secret_type,
        "data": {k: base64.b64encode(v.encode()).decode() for k, v in data.items()},
    }, namespace)


def create_pvc(
    name: str,
    namespace: str,
    size: str,
    storage_class: str = "",
    access_mode: str = "ReadWriteOnce",
) -> bool:
    """Create a PVC if it does not already exist.

    Returns:
        True  — PVC was just created.
        False — PVC already existed; nothing was changed.

    Raises:
        RuntimeError: If storage_class is specified but does not exist.
    """
    if resources.exists("PersistentVolumeClaim", name, namespace):
        log.info("PVC '%s' already exists — skipping", name)
        return False
    if storage_class and not resources.exists("StorageClass", storage_class):
        raise RuntimeError(f"StorageClass '{storage_class}' not found.")
    log.info("Creating PVC '%s' (%s, %s)", name, size, storage_class or "default class")
    spec: dict[str, Any] = {
        "accessModes": [access_mode],
        "resources": {"requests": {"storage": size}},
    }
    if storage_class:
        spec["storageClassName"] = storage_class
    resources.apply_dict({
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": name, "namespace": namespace},
        "spec": spec,
    }, namespace)
    return True


def verify(s3_secret_name: str, namespace: str) -> None:
    """Assert the S3 credentials Secret exists. Raises RuntimeError if absent."""
    log.info("Verifying storage Secret '%s'", s3_secret_name)
    if not resources.exists("Secret", s3_secret_name, namespace):
        raise RuntimeError(
            f"S3 credentials Secret '{s3_secret_name}' not found in namespace '{namespace}'. "
            "Run 'rhoai platform deploy' to configure storage."
        )
    log.info("Storage Secret '%s' confirmed", s3_secret_name)


def validate_local_artifacts(*paths: Path) -> None:
    """Raise FileNotFoundError if any of the given paths does not exist.

    Called before any cluster operation so the user gets an immediate, clear
    error when a local file is missing rather than a mid-deploy failure.

    Args:
        *paths: One or more local filesystem paths to validate.

    Raises:
        FileNotFoundError: On the first missing path, with its absolute value.
    """
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"Artifact not found: {p.resolve()}\n"
                "Check that the path in your config file points to an existing "
                "file on this machine."
            )


def copy_files_to_pvc(
    pvc_name: str,
    namespace: str,
    files: dict[str, Path],
    timeout: int = 120,
) -> None:
    """Copy local files onto a PVC at specified destination paths.

    A short-lived Pod is created to mount the PVC, ``oc cp`` is used to copy
    each file, and the Pod is deleted on completion (even on failure).

    This function is format-agnostic and runtime-agnostic.  The caller is
    responsible for deciding which files to copy and where to put them.
    Use it as the building block for any higher-level staging helper
    (e.g. ``stage_triton_model`` in a use-case assets module).

    Args:
        pvc_name:  Name of the PVC to write into.  Must already exist and be Bound.
        namespace: Namespace of the PVC and staging Pod.
        files:     Mapping of ``{dest_path_on_pvc: local_Path}``.
                   Destination paths are absolute paths inside the PVC mount
                   (e.g. ``"/models/resnet/1/model.onnx"``).
        timeout:   Maximum seconds to wait for the staging Pod to reach Running.

    Raises:
        RuntimeError: If the staging Pod does not reach Running within *timeout*,
                      or if any ``oc`` command fails.
    """
    pod_name = f"{_STAGING_POD_PREFIX}-{pvc_name}"[:_K8S_NAME_MAX]
    log.info("Staging %d file(s) onto PVC '%s'", len(files), pvc_name)

    # Remove any stale Pod from a previous interrupted run.
    resources.delete_manifest("Pod", pod_name, namespace)

    # Create a minimal Pod that mounts the PVC so oc cp can write into it.
    resources.apply_dict({
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": pod_name, "namespace": namespace},
        "spec": {
            "restartPolicy": "Never",
            "containers": [{
                "name": "staging",
                "image": "quay.io/powercloud/busybox:ubi",
                "command": ["sh", "-c", "sleep infinity"],
                "volumeMounts": [{"name": "model-vol", "mountPath": "/mnt/pvc"}],
            }],
            "volumes": [{
                "name": "model-vol",
                "persistentVolumeClaim": {"claimName": pvc_name},
            }],
        },
    }, namespace)

    # Wait for the Pod to be Running before oc cp will accept connections.
    _wait_for_pod_running(pod_name, namespace, timeout)

    try:
        for dest, local in files.items():
            _oc_cp(local, namespace, pod_name, dest)
    finally:
        # Always clean up the staging Pod, even on failure.
        resources.delete_manifest("Pod", pod_name, namespace)
        log.debug("Staging Pod '%s' deleted", pod_name)


def _wait_for_pod_running(pod_name: str, namespace: str, timeout: int) -> None:
    """Block until the named Pod is Running. Raises RuntimeError on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            pod_status = resources.status("Pod", pod_name, namespace)
            if pod_status.get("phase") == "Running":
                log.debug("Staging Pod '%s' is Running", pod_name)
                return
        except Exception:  # noqa: BLE001
            pass
        log.debug("Waiting for staging Pod '%s' to reach Running", pod_name)
        time.sleep(_STAGING_POLL_INTERVAL)
    raise RuntimeError(
        f"Staging Pod '{pod_name}' did not reach Running within {timeout}s. "
        "Check cluster events: oc describe pod -n "
        f"{namespace} {pod_name}"
    )


def _oc_cp(local_path: Path, namespace: str, pod_name: str, pod_dest: str) -> None:
    """Copy a local file into a running Pod via ``oc cp``.

    Runs ``oc cp <local_path> <namespace>/<pod_name>:<pod_dest>``.
    The destination directory is created with ``oc exec mkdir -p`` first
    because ``oc cp`` does not create missing parent directories.

    Raises:
        RuntimeError: If either ``oc exec`` or ``oc cp`` exits non-zero.
    """
    dest_dir = pod_dest.rsplit("/", 1)[0]

    mkdir_result = subprocess.run(
        ["oc", "exec", "-n", namespace, pod_name, "--", "mkdir", "-p", dest_dir],
        capture_output=True,
        text=True,
    )
    if mkdir_result.returncode != 0:
        raise RuntimeError(
            f"oc exec mkdir failed in Pod '{pod_name}': {mkdir_result.stderr.strip()}"
        )

    cp_result = subprocess.run(
        ["oc", "cp", str(local_path), f"{namespace}/{pod_name}:{pod_dest}"],
        capture_output=True,
        text=True,
    )
    if cp_result.returncode != 0:
        raise RuntimeError(
            f"oc cp failed copying '{local_path.name}' to Pod '{pod_name}': "
            f"{cp_result.stderr.strip()}"
        )
    log.debug("Copied '%s' → %s:%s", local_path.name, pod_name, pod_dest)


def delete_staging_pod(pvc_name: str, namespace: str) -> None:
    """Delete the staging Pod for a PVC if it exists.

    The staging Pod is created by ``copy_files_to_pvc`` and is normally
    deleted in its ``finally`` block.  This function cleans up any Pod that
    was left behind by an interrupted or failed deploy run.

    The Pod name is derived from the PVC name using the same formula as
    ``copy_files_to_pvc``, so it is always deterministic and idempotent.

    Args:
        pvc_name:  Name of the PVC whose staging Pod should be removed.
        namespace: Namespace where the Pod lives.
    """
    pod_name = f"{_STAGING_POD_PREFIX}-{pvc_name}"[:_K8S_NAME_MAX]
    log.info("Removing staging Pod '%s' (if present)", pod_name)
    resources.delete_manifest("Pod", pod_name, namespace)
    wait.wait_until_deleted("Pod", pod_name, namespace)


def delete_pvc(name: str, namespace: str) -> None:
    """Delete a PVC and wait for removal."""
    log.info("Deleting PVC '%s'", name)
    resources.delete_manifest("PersistentVolumeClaim", name, namespace)
    wait.wait_until_deleted("PersistentVolumeClaim", name, namespace)
