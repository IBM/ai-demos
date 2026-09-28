# rhoai — Module API Reference

Per-module function reference for the `platform/` and `ocp/` layers, plus the
internals of the `fraud-detection` use case. This is the "what does each module
expose" companion to the higher-level docs.

**Looking for something else?**

| You want… | See |
|---|---|
| Architecture, layering rules, conventions, adding a use case | [`CLAUDE.md`](../../../CLAUDE.md) |
| Install the CLI, quick start, troubleshooting | [`automation/README.md`](../README.md) |
| `rhoai platform` command reference | [`rhoai/platform/README.md`](../rhoai/platform/README.md) |
| `rhoai usecase` commands + shared config concepts | [`rhoai/usecases/README.md`](../rhoai/usecases/README.md) |
| Deploying the fraud-detection use case | [`rhoai/usecases/fraud_detection/README.md`](../rhoai/usecases/fraud_detection/README.md) |
| Default config values | [`rhoai/config/defaults.yaml`](../rhoai/config/defaults.yaml) |

Imports flow one direction — `cli → platform → ocp → utils`, with `usecases`
alongside (`cli → usecases → platform`). `ocp/resources.py` is the only file that
uses the `kubernetes` Python client; other modules reach the cluster through it
(a few also shell out to the `oc` CLI — e.g. `storage`, `trustyai`, `prepare`).
See [`CLAUDE.md`](../../../CLAUDE.md) for the full invariants.

---

## Contents

1. [Platform Capabilities](#1-platform-capabilities)
2. [OpenShift Operations](#2-openshift-operations)
3. [Use Case Internals](#3-use-case-internals)

---

## 1. Platform Capabilities

Platform modules live in `rhoai/platform/` and provide reusable building blocks.
Use cases and the CLI call these modules. Each platform module imports from
`rhoai.ocp.resources` and `rhoai.ocp.wait` to reach the cluster.

Two additional modules back model serving and TrustyAI monitoring:
`request_generator.py` (Triton `config.pbtxt` + dataset → KServe v2 requests) and
`trustyai_client.py` (stateless TrustyAI REST client — metrics, name mapping).

### `prepare.py`

Cluster validation and platform bootstrap.

| Function | Description |
|---|---|
| `validate_login()` | Confirm the cluster is reachable via the current kubeconfig. |
| `validate_permissions(namespace)` | Confirm `create` access on `Subscriptions` in the operator namespace. |
| `validate_storage(class_name)` | Confirm a named `StorageClass` exists, or that at least one exists when no name is given. |
| `validate_namespace(namespace)` | Ensure the namespace exists, creating it if absent. |
| `platform_needs_reconciliation(config)` | Return `True` if the platform state does not match the desired config. |
| `bootstrap_platform(config)` | Full bootstrap: validate cluster → operator install/wait → DSC/DSCI apply and wait. Shared by the CLI and all use-case deploy functions. |
| `get_cluster_info()` | Return OpenShift version, topology (SNO vs Multi-node), per-worker-node CPU/memory/GPU, and per-StorageClass PV usage. |

### `operators.py`

RHOAI operator lifecycle via OLM.

| Function | Description |
|---|---|
| `is_installed(name, namespace)` | Return `True` if the operator CSV is in `Succeeded` phase. |
| `install(name, namespace, channel, repo_root, timeout, version="", source="redhat-operators", source_namespace="openshift-marketplace")` | Apply `OperatorGroup` + `Subscription` and wait for CSV to succeed. Pin a CSV with `version`. |
| `wait_until_ready(name, namespace, timeout)` | Block until the CSV reaches `Succeeded`. |
| `resolve_csv_name(package_name, namespace)` | Discover the versioned CSV name (e.g. `rhods-operator.3.5.0`) from the package name prefix. |
| `verify(name, namespace)` | Assert the CSV is `Succeeded`; raise `RuntimeError` otherwise. |
| `get_csv_info(name, namespace)` | Return display info: name, version, phase. |

### `dsc.py`

`DataScienceCluster` and `DSCInitialization` lifecycle.

| Function | Description |
|---|---|
| `apply_dsci(path)` | Apply the `DSCInitialization` manifest. Idempotent. |
| `apply_dsc(path)` | Apply the `DataScienceCluster` manifest. Idempotent. |
| `wait_until_ready(name, timeout)` | Block until the DSC reaches `Ready` phase. |
| `verify_dsci(name)` | Assert the DSCI is `Ready`; raise `RuntimeError` otherwise. |
| `verify(name)` | Assert the DSC is `Ready`; raise `RuntimeError` otherwise. |
| `delete_dsc(name)` | Delete the DSC and wait for removal. |
| `delete_dsci(name)` | Delete the DSCI and wait for removal. |
| `get_component_states(name)` | Return a `{component: managementState}` dict from the DSC spec. |

### `inference.py`

KServe `InferenceService` and `ServingRuntime` management.

| Function | Description |
|---|---|
| `apply_serving_runtime_from_template(template_path, platform_namespace, deploy_namespace, model_name, runtime_name)` | Process an OpenShift Template (`oc process`) and apply the resulting `ServingRuntime` into the deploy namespace. |
| `wait_until_ready(name, namespace, timeout)` | Block until the `InferenceService` is `Ready`. |
| `verify(namespace, name=None)` | Assert one specific (or all) `InferenceService`(s) in the namespace are `Ready`. |
| `get_inference_url(name, namespace)` | Return the public inference URL from `.status.url` or the OpenShift `Route`. |
| `delete_inference_service(name, namespace)` | Send the delete request for an `InferenceService`. Does not wait (a separate poll handles removal). |
| `delete_serving_runtime(name, namespace)` | Send the delete request for a `ServingRuntime`. Does not wait (a separate poll handles removal). |

### `trustyai.py`

TrustyAI Service lifecycle for ML predictive model monitoring (bias detection,
data drift).

| Function | Description |
|---|---|
| `enable_user_workload_monitoring(manifest_path)` | Apply the cluster-monitoring and user-workload-monitoring `ConfigMap`s. Reads the namespace from each document in the multi-doc YAML. |
| `patch_inferenceservice_config(namespace)` | Set `opendatahub.io/managed: "false"` on `inferenceservice-config` and merge the CA-bundle settings into the logger config. Required in `RawDeployment` mode so TrustyAI can inject its payload-logging sidecar without RHOAI reverting the ConfigMap. |
| `apply_trustyai_service(path, namespace)` | Apply the `TrustyAIService` CR manifest. |
| `wait_until_ready(name, namespace, timeout)` | Block until TrustyAI is ready — polls the backing `Deployment` (the CR exposes no Ready condition). |
| `verify(name, namespace)` | Assert TrustyAI is ready via its backing `Deployment`; raise `RuntimeError` otherwise. |
| `delete_trustyai_service(name, namespace)` | Delete the `TrustyAIService` CR and wait for removal. |

### `storage.py`

S3 credentials and PVC management.

| Function | Description |
|---|---|
| `apply_s3_secret(path, namespace)` | Apply an S3 credentials `Secret` manifest. |
| `create_secret(name, namespace, data, secret_type="Opaque")` | Create or update a `Secret` from a plain-string dict (base64 encoding applied internally). |
| `create_pvc(name, namespace, size, ...)` | Create a `PersistentVolumeClaim`. Skips if already exists. |
| `verify(secret_name, namespace)` | Assert the S3 credentials `Secret` exists. |
| `delete_pvc(name, namespace)` | Delete a PVC and wait for removal. |

### `verify.py`

Core platform health checks.

| Function | Description |
|---|---|
| `verify_platform(config)` | Run three checks — operator CSV, DSCI, DSC — and return a list of `CheckResult` objects. Does not raise; each failure is captured in its `CheckResult` for the caller to inspect. |

---

## 2. OpenShift Operations

`rhoai/ocp/resources.py` is the **only file that uses the `kubernetes` Python
client**. Other modules reach the cluster through it; a few also shell out to the
`oc` CLI directly (`storage`, `trustyai`, `prepare`, and `resources.process_template`).

### `resources.py`

Wraps the `kubernetes` Python client's dynamic API with a thin, consistent
interface. Every resource kind that cannot be discovered by kind name alone is
pinned in `_API_HINTS` to avoid ambiguity with OpenShift's CRD overlaps.

| Function | Description |
|---|---|
| `apply_manifest(path, namespace)` | Load a YAML file and apply it via server-side apply. Idempotent. |
| `apply_dict(manifest, namespace)` | Apply a manifest dict via server-side apply. |
| `process_template(path, platform_namespace, deploy_namespace, params=None)` | Run `oc process -f <path>` (optionally with `params`), then apply the rendered resources into the deploy namespace. |
| `delete_manifest(kind, name, namespace)` | Delete a resource by kind and name. Silent if already absent. |
| `exists(kind, name, namespace)` | Return `True` if the resource exists. |
| `get(kind, name, namespace)` | Return the full resource object dict. Raises `NotFoundError` if absent. |
| `patch(kind, name, patch_body, namespace, strategy)` | Patch a resource (merge, json, or strategic). |
| `status(kind, name, namespace)` | Return the `.status` sub-object, or `{}` if absent. |
| `is_ready(kind, name, namespace)` | Return `True` if the resource meets its readiness condition. Dispatch varies by kind. |
| `list_resources(kind, namespace, label_selector)` | Return all matching resources as a list of dicts. |

### `wait.py`

Polling helpers that call back into `resources.py`.

| Function | Description |
|---|---|
| `wait_until_ready(kind, name, namespace, timeout)` | Poll `is_ready()` until `True` or timeout. |
| `wait_until_deleted(kind, name, namespace, timeout)` | Poll `exists()` until `False` or timeout. |
| `wait_until(condition, description, timeout)` | General-purpose poller for custom conditions. |

---

## 3. Use Case Internals

Each use case lives in `rhoai/usecases/<name>/` and exposes exactly three public
functions: `deploy(config)`, `verify(config)`, and `cleanup(config)`. This section
documents how the `fraud-detection` use case is wired internally; for how to
*deploy* it, see its
[user guide](../rhoai/usecases/fraud_detection/README.md). For the template to add
a new use case, see [`CLAUDE.md`](../../../CLAUDE.md) §"Adding a use case".

### `assets.py`

The only file in a use case that knows where that use case's manifests live on
disk. All paths are resolved relative to `config["repo_root"]`.

Shared platform manifests (DSC, DSCI, S3 secret, Triton template) are delegated
to `platform/manifests.py` — `assets.py` only resolves use-case-specific paths.

### `deploy.py`

Orchestrates the full deployment in a documented, numbered sequence:

1–3. Call `prepare.bootstrap_platform(config)` — validates the cluster,
     installs the operator, and applies DSC/DSCI.
4.   Apply S3 credentials (skipped for `pvc://` URIs).
5.   For each model in `deployment.models`: apply Triton `ServingRuntime`
     (via OpenShift Template), apply `InferenceService`, wait for `Ready`,
     run smoke-test inference.
6.   If any model has `bias_monitoring` configured: enable user-workload
     monitoring, apply RBAC, apply `TrustyAIService`, wait for `Ready`.
7.   For each model with `bias_monitoring`: send observations, wait for
     ingestion, apply name mapping (optional), schedule SPD and identity
     monitors (optional).
8.   Print deployment summary.

`deploy.py` orchestrates through `platform/` modules and additionally uses
`ocp/resources` directly for a few low-level applies/existence checks (e.g.
`apply_dict`, `exists`).

#### Inference input modes

Each model entry declares its inference input in exactly one of two mutually
exclusive modes, validated up front by `assets.validate_model_config` (both
`deploy` and `verify` run it before touching the cluster):

| | JSON mode | Dataset mode |
|---|---|---|
| Key | `inference_request` | `inference_dataset` |
| Smoke test | the file, used as-is (JSON) or converted (CSV) | first request from `iter_requests(..., batch_size=1)` — the first row/element, never the full batch |
| Observations | declared separately under `bias_monitoring.observations.path`/`.files` | derived from the dataset, batched by `bias_monitoring.observations.batch_size` (default 1) |
| Extra requirement | — | a Triton `config.pbtxt` via `inference_config_path`/`config_path`, or an ONNX `model_path` the framework generates one from |

Smoke test and observations share one generator (`request_generator.iter_requests`),
so in dataset mode the dataset is the single source of truth. Rejected as
ambiguous: both keys set; neither set; `inference_dataset` +
`observations.path`/`.files`; `inference_request` + `observations.batch_size`;
a dataset with no pbtxt and no `model_path` to generate one from.

When an ONNX `model_path` is supplied without a `config_path`, `deploy.py`
generates a Triton `config.pbtxt` via `platform/config_generator.py` (writing it
to `automation/rhoai/usecases/fraud_detection/inputs/<name>/config.pbtxt`, relative
to `repo_root`) and records the path on the model dict so staging, request
generation, and name mapping all resolve the same file.

### `verify.py`

Runs platform checks (`verify_platform`) then for each model:
confirms `InferenceService` is `Ready`, runs live inference, and (when
`bias_monitoring` is configured) checks TrustyAI observation count > 0.

### `cleanup.py`

Deletes use-case resources in reverse deploy order: first each model's
`InferenceService` and `ServingRuntime`, then (if it was deployed) the TrustyAI
Service and its supporting resources. Does not touch platform resources
(DSC, DSCI) — that is the CLI's responsibility via `--delete-platform`.

---

*Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).*
