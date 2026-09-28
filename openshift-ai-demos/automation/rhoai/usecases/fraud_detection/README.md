# Fraud Detection

Deploy one or more fraud detection models on Red Hat OpenShift AI using the Triton serving runtime, with optional TrustyAI bias monitoring. This guide covers the required assets, configuration, deployment, verification, and cleanup steps. 

For the generic use case lifecycle and shared concepts, see the [use-case index](../README.md).

## Contents

- [When should I use this?](#when-should-i-use-this)
- [Required assets](#required-assets)
- [Architecture](#architecture)
- [Configuration reference](#configuration-reference)
  - [Model configuration](#model-configuration)
  - [Providing the model](#providing-the-model)
  - [Inference inputs](#inference-inputs)
  - [TrustyAI configuration](#trustyai-configuration)
- [Deployment walkthrough](#deployment-walkthrough)
- [Verification](#verification)
- [Cleanup](#cleanup)
- [Troubleshooting](#troubleshooting)

---

## When should I use this?

Use this use case to deploy one or more predictive fraud detection models on OpenShift AI and, optionally, demonstrate TrustyAI fairness monitoring. For example, you can deploy a biased and an unbiased model side by side and compare their Statistical Parity Difference (SPD) over time.

**Prerequisites**

- For framework installation, cluster access, and platform requirements, see the [Getting Started](../../../README.md#getting-started) guide.
- **Compute:** each model deployment requests **2 CPU / 8 GiB** (limits also 2
  CPU / 8 GiB) on a worker node. Confirm capacity with `rhoai platform inspect`.

---

## Required assets

This use case requires external assets that are not included in the framework. Assemble them before deploying.

| Asset | What it is | Where it's used |
|---|---|---|
| Model file (e.g. `model.onnx`) | The trained fraud-detection model, in a Triton-supported format | Staged to a PVC via `model_path` or a pre-loaded `model_uri` |
| Triton `config.pbtxt` | The model's Triton configuration | `config_path` — **optional for ONNX models**; auto-generated from the model's I/O signature when omitted |
| Inference input | Either a pre-built KServe v2 request **or** a raw CSV/JSON dataset | `inference_request` **or** `inference_dataset`  |

All asset paths (`model_path`, `config_path`, `inference_request`,
`inference_dataset`) are **absolute** local paths. Set
`repo_root` to the **absolute** path of your `openshift-ai-demos` directory; it
is used to locate the framework's own manifests and generated files, not your
asset paths.

---


## Architecture

Each model in the config becomes an independent deployment: its own Triton
`ServingRuntime` and a KServe `InferenceService` in front of it. Multiple models
deploy in sequence, which is what enables the biased/unbiased side-by-side demo.

```
config.yaml (deployment.models[])
      │
      ▼
 per model:  ServingRuntime (triton-<name>)  ──►  InferenceService (<name>)  ──►  smoke test
      │
      └─ optional: TrustyAIService (one per namespace) ──► observations ──► SPD / identity monitors
```

Models are served from a PVC. KServe mounts the PVC at `/mnt/models` and Triton
loads the model from the standard repository layout:

```
<claim> (PVC root)
└── <name>/           ← the InferenceService / model name
    ├── config.pbtxt  ← Triton model configuration
    └── 1/
        └── model.onnx
```

---

## Configuration reference

Start with `config-fraud-detection.yaml`. If you want to enable TrustyAI bias monitoring, see `config-fraud-detection-trustyai.yaml`.

| Config file | When to use |
|---|---|
| [`config-fraud-detection.yaml`](../../../config-fraud-detection.yaml) | Deploy the Fraud Detection use case without TrustyAI |
| [`config-fraud-detection-trustyai.yaml`](../../../config-fraud-detection-trustyai.yaml) | Deploy the Fraud Detection use case with TrustyAI bias monitoring (biased and unbiased models side by side). |


```yaml
# config-fraud-detection.yaml
repo_root: /absolute/path/to/openshift-ai-demos   # must be absolute — ~ is not expanded

operator:
  channel: stable-3.5

components:
  - dashboard
  - workbenches
  - kserve

deployment:
  namespace: test-fraud
  models:
    - name: upi-fraud-detection
      model_path:  /absolute/path/to/model.onnx
      config_path: /absolute/path/to/config.pbtxt
      pvc_name:    my-fraud-model-pvc   
      pvc_size:    1Gi
      inference_request: /absolute/path/to/inference request
```

### Model configuration

Each entry under `deployment.models` defines one independent model deployment:

| Field | Required | Description |
|---|---|---|
| `name` | ✔ | Kubernetes resource name for the `InferenceService` and `ServingRuntime` |
| `model_uri` | ✦ | Reference a model **already staged** elsewhere: `pvc://<claim>/<path>`, `hf://…`, `oci://…`, or a plain S3 path (any other string, which applies the S3 secret). |
| `model_path` | ✦ | Local model file; the framework creates the PVC and stages the Triton layout for you |
| `config_path` | | Local Triton `config.pbtxt`. **Optional for ONNX models** — when omitted, one is generated from the model's I/O signature (see [Generated config.pbtxt](#generated-configpbtxt)) |
| `max_batch_size` | | Config generation only — enable batching with this max batch size (default `0` = shapes preserved verbatim). Requires a dynamic leading dim on every tensor |
| `dynamic_batching` | | Config generation only — emit a `dynamic_batching {}` block when batching is enabled (default `true`) |
| `pvc_name` | | Local-staging only — PVC to create/reuse (defaults to `<name>-pvc`) |
| `pvc_size` | | Local-staging only — PVC size (defaults to `1Gi`) |
| `inference_request` | ✱ |  Absolute path to a pre-built KServe v2 JSON payload used for post-deploy validation |
| `inference_dataset` | ✱ |  Absolute path to a raw CSV/JSON dataset the framework generates requests from |
| `bias_monitoring` | | TrustyAI bias monitoring config — see [TrustyAI configuration](#trustyai-configuration) |

> **✦ Model source** — supply **either** `model_uri` **or** `model_path`
> (with an optional `config_path`).
>
> **✱ Inference input** — supply **exactly one** of `inference_request` or
> `inference_dataset`.

Multiple entries deploy multiple models in sequence, each with its own
`ServingRuntime` and `InferenceService`. Use this to run biased and unbiased
models side by side for a comparative fairness demonstration.


### Providing the model

Provide the model in one of the following ways:

**Local files (recommended):**  
Specify a local model file with `model_path`. Optionally provide its Triton
`config.pbtxt` with `config_path`; for ONNX models you can omit it and the
framework generates one (see [Generated config.pbtxt](#generated-configpbtxt)).
The framework validates the files, creates (or reuses) the PVC, and stages the
Triton model repository layout automatically.

```yaml
deployment:
  models:
    - name: upi-fraud-detection
      model_path:  /abs/path/to/model.onnx
      config_path: /abs/path/to/config.pbtxt   # optional for ONNX — generated when omitted
      pvc_size:    5Gi                          # optional
```

**Existing PVC:**  
If the model is already staged on a PVC, reference it with
`model_uri: pvc://<claim>/<path>`. The framework uses the existing model
repository without copying any files.

> Ensure the PVC exists, is Bound, and contains the expected Triton model
repository before deploying.
```
oc get pvc -n <namespace>
```

#### Generated config.pbtxt

When you supply an ONNX `model_path` without a `config_path`, the framework
inspects the model's input/output signature (via ONNX Runtime) and generates a
Triton `config.pbtxt` for you. It records the model `name`, the
`onnxruntime_onnx` platform, and the input/output tensors (names, datatypes,
and shapes, preserving dynamic dimensions as `-1`).

The generated file is written to a persistent, discoverable location so you can
inspect exactly what was staged:

```
automation/rhoai/usecases/fraud_detection/inputs/<name>/config.pbtxt
```

(relative to `repo_root`, i.e. your `openshift-ai-demos` directory.)

Its path is printed in the deploy summary under the model, annotated
`(generated)`. It is also reused as the request schema in
[dataset mode](#dataset-mode) and as the source of output names for
[name mapping](#name-mapping).

Batching is **opt-in** because a dynamic leading dimension is ambiguous — it may
be a batch dim or a genuine variable-length dim. By default (`max_batch_size: 0`)
the model's shapes are preserved verbatim. Set `max_batch_size` > 0 to enable
batching (this requires a dynamic leading dim on every tensor; otherwise it
falls back to `0`), and set `dynamic_batching: false` to suppress the
`dynamic_batching {}` block.

```yaml
deployment:
  models:
    - name: upi-fraud-detection
      model_path:       /abs/path/to/model.onnx   # no config_path → generated
      max_batch_size:   8                          # optional (default 0 = no batching)
      dynamic_batching: true                       # optional (default true)
```

> Installing the framework with the `onnx` extra (`pip install -e ".[onnx]"`)
> pulls in ONNX Runtime, which is required for config generation.
>
> A standalone version of this generator is also available at
> [`tools/generate_config_pbtxt.py`](../../../../../tools/README_generate_config_pbtxt.md)
> for generating a `config.pbtxt` outside of a deployment.

### Inference inputs

Each model must specify exactly one inference input:

| Mode | Key | Used for |
|---|---|---|
| **JSON request** | `inference_request` | Uses a pre-built KServe request for validation |
| **Dataset** | `inference_dataset` | Generates validation requests and TrustyAI observations from the dataset |

JSON request uses a pre-built KServe v2 request payload for deployment validation. If TrustyAI is enabled, observations are provided separately through `bias_monitoring.observations`.

#### Dataset mode

Use `inference_dataset` when your input is a CSV dataset. The dataset
becomes the single source of truth:

- The framework generates the deployment validation request from the dataset.
- If TrustyAI is enabled, observations are generated from the same dataset.
- `bias_monitoring.observations.batch_size` controls how many observations are
  included in each generated request (default: `1`).

Dataset mode needs a Triton `config.pbtxt` to define the request schema. Supply
one via `config_path` (or `inference_config_path`), **or** provide an ONNX
`model_path` and let the framework [generate one](#generated-configpbtxt).

```yaml
# Dataset mode (excerpt) — see config-fraud-detection-trustyai.yaml
deployment:
  models:
    - name: upi-fraud-detection
      model_path:  /absolute/path/model.onnx
      config_path: /absolute/path/config.pbtxt      # optional — generated from model_path when omitted
      inference_dataset:/absolute/path/to/csv/batch_01.csv
      bias_monitoring:
        observations:
          batch_size: 250                        # observations per generated request
```

**Configuration rules**

- Use either `inference_request` or `inference_dataset`, but not both.
- Dataset mode needs a Triton `config.pbtxt` — supplied via `config_path`, or generated from an ONNX `model_path`.
- Dataset mode automatically generates TrustyAI observations, so do not set `bias_monitoring.observations.path` or `bias_monitoring.observations.files`.
- `observations.batch_size` is only used with dataset mode.

### TrustyAI configuration

TrustyAI bias monitoring is optional. Enable it by adding a
`bias_monitoring` block to each model you want to monitor.

To enable bias monitoring:

- Add `trustyai` to the `components` list.
- Add a `bias_monitoring` block to each model you want monitored.
- Configure `trustyai_service_name` and `trustyai_service_account` under
  `deployment:`. One TrustyAI service is shared by all monitored models in the
  namespace.

```yaml
# config-fraud-detection-trustyai.yaml (excerpt) — see the file for the full example
deployment:
  namespace: test-fraud
  trustyai_service_name:    trustyai-service   # default
  trustyai_service_account: trustyai-user      # default

  models:
    - name: upi-fraud-detection
      model_path:  /absolute/path/model.onnx
      config_path: /absolute/path/config.pbtxt   
      inference_dataset: /absolute/path/to/csv/batch_01.csv 
      bias_monitoring:
        observations:
          batch_size: 250   
        name_mapping:
          # inputs/outputs are optional — see "Name mapping" below.
          inputs:  
            customer_data_input-3: "Is Male-Identifying?"
          outputs: 
            predict: "Will Default?" 
        spd_monitors:
          - protected_attribute: "Is Male-Identifying?"
            privileged_value:    1.0
            unprivileged_value:  0.0
            outcome_name:        "Will Default?"
            favorable_outcome:   0.0
            batch_size:          5000
```

| Key | Purpose |
|---|---|
| `observations` | Baseline data sent to TrustyAI during deployment. In JSON mode, specify path or files. In dataset mode, observations are generated automatically from `inference_dataset`; configure `observations.batch_size` instead |
| `name_mapping` | Maps opaque tensor names to human-readable labels so monitors can reference meaningful names. Both `inputs` and `outputs` are optional — see [Name mapping](#name-mapping) |
| `spd_monitors` | Schedules recurring Statistical Parity Difference (SPD) monitors |
| `identity_monitors` | Optional. Tracks the average value of a named column over time |
| `trustyai_service_name` | `TrustyAIService` resource name (default `trustyai-service`). |
| `trustyai_service_account` | ServiceAccount created for TrustyAI RBAC (default `trustyai-user`). |

#### Name mapping

TrustyAI monitors reference features and outcomes by name, but models often
expose opaque tensor names (e.g. `customer_data_input-3`, `predict`). A
`name_mapping` block relabels them. Both halves are **optional** because the
framework derives sensible defaults:

- **Inputs** — when `inference_dataset` is a **`.csv`**, input names are derived
  automatically from the dataset's column headers. A single flat tensor is
  mapped by column index; a multi-tensor model reuses the header names. Set
  `name_mapping.inputs` only to override these derived names.
- **Outputs** — output names default to the **output names in the model's
  `config.pbtxt`** (whether you supplied it or the framework generated it),
  falling back to the names in the first inference response. Set
  `name_mapping.outputs` only to relabel them.

Provide only the entries you want to change:

```yaml
        name_mapping:
          inputs:  { customer_data_input-3: "Is Male-Identifying?" }   # override one column
          outputs: { predict: "Will Default?" }                        # relabel the outcome
```

The names you map here are the ones referenced by `spd_monitors`
(`protected_attribute`, `outcome_name`) and `identity_monitors` (`column_name`).

**Observing fairness metrics:**  
After deployment, the summary includes a link to the OpenShift metrics browser. Query `trustyai_spd` to view Statistical Parity Difference metrics. If you've configured `identity monitors`, `trustyai_identity` metrics are also available.

---

## Deployment walkthrough

```bash
rhoai usecase deploy fraud-detection -c config-fraud-detection.yaml
```

Checks the RHOAI platform (bootstrapping it if needed), deploys the Triton
`ServingRuntime` and `InferenceService` for each configured model, then
smoke-tests each model.

**Sample output:**
```

Deploying  : fraud-detection
Namespace  : test-fraud


Checking RHOAI platform...

  ✔  Operator ready
  ✔  DSCInitialization ready
  ✔  DataScienceCluster ready
  ✔  Components enabled: Dashboard, Workbenches, KServe

✔  Platform ready  (3s)

Deploying 'upi-fraud-detection'...

✔  Validating local model artifacts  (0s)
✔  Ensuring PVC 'my-fraud-model-pvc'  (0s)
✔  Staging model repository on PVC  (26s)
✔  Ensuring serving runtime  (6s)
✔  Ensuring inference service  (21s)
✔  Smoke-testing model endpoint  (1s)

✔  'upi-fraud-detection' ready  (56s)

Deployment complete.  Total: 59s

  Use case   : fraud-detection
  Namespace  : test-fraud

Models

✔  upi-fraud-detection
  Endpoint     : https://upi-fraud-detection-test-fraud.apps.rdr-varad-421.ocp-rhoai.com
  Smoke test   : Passed

  Inference request
    {
      "inputs": [
         . . .
      ]
    }
  Inference response
    {
      "model_name": "upi-fraud-detection",
      "model_version": "1",
      "outputs": [
            . . .
      ]
    }

Next steps

  Invoke the model

    # upi-fraud-detection
    curl -sk -X POST \
      https://upi-fraud-detection-test-fraud.apps.rdr-varad-421.ocp-rhoai.com/v2/models/upi-
fraud-detection/infer \
      -H 'Content-Type: application/json' \
      -d
@/Users/varad/Desktop/repos/ai-demos/openshift-ai-demos/automation/rhoai/usecases/fraud_dete
ction/inputs/upi-fraud-detection.json

  Clean up deployment

    rhoai usecase cleanup fraud-detection \
      -c openshift-ai-demos/automation/config-fraud-detection.yaml

```

When the framework generates artifacts for a model — a `config.pbtxt` from an
ONNX `model_path`, or a request JSON in [dataset mode](#dataset-mode) — their
paths are listed under the model in the summary, annotated `(generated)`, so you
can find and inspect them:

```
✔  upi-fraud-detection
  Endpoint     : https://upi-fraud-detection-test-fraud.apps.rdr-varad-421.ocp-rhoai.com
  Smoke test   : Passed
  Config       : .../inputs/upi-fraud-detection/config.pbtxt          (generated)
  Request      : .../inputs/upi-fraud-detection_generated_request.json  (generated)
```

If the endpoint is unreachable from the machine running the CLI (for example, a
workstation that cannot resolve the cluster's `.apps` route), inference
validation is marked **Unavailable**, the deploy is still considered successful,
and a `Follow-up actions` block prints the exact `verify` command to rerun once
the endpoint is reachable. See [Troubleshooting](#troubleshooting).

### With TrustyAI

When any model has `bias_monitoring` configured, the deploy sequence is extended
after all models are running:

1. Deploy each model (`ServingRuntime` + `InferenceService`)
2. Enable user-workload monitoring
3. Deploy `TrustyAIService`, wait for `Ready`
4. Send observations (from the dataset or the declared files), wait for ingestion
5. Apply name mapping (if configured)
6. Schedule SPD and identity monitors (if configured)

```bash
rhoai usecase deploy fraud-detection -c config-fraud-detection-trustyai.yaml
```

---

## Verification

```bash
rhoai usecase verify fraud-detection -c config-fraud-detection.yaml
```

Checks the platform, confirms each `InferenceService` is Ready, and runs a live
inference request for each model. With a TrustyAI config, `verify` additionally
confirms that TrustyAI has observations recorded (count > 0) for each monitored
model.

**Sample output:**
```
Verifying  : fraud-detection
Namespace  : test-fraud

✔  Checking RHOAI platform  (4s)

Verifying 'upi-fraud-detection'...

✔  Checking inference service  (0s)
✔  Checking model inference  (1s)

✔  'upi-fraud-detection' healthy  (1s)

Verification complete.  Total: 5s

  Use case   : fraud-detection
  Namespace  : test-fraud

Models

✔  upi-fraud-detection
  Endpoint     : https://upi-fraud-detection-test-fraud.apps.rdr-varad-421.ocp-rhoai.com
  Smoke test   : Passed

  Inference request { data }
  Inference response { data }

Next steps

  Invoke the model

    # upi-fraud-detection
    curl -sk -X POST \
      https://upi-fraud-detection-test-fraud.apps.<cluster>/v2/models/upi-fraud-detection/infer \
      -H 'Content-Type: application/json' \
      -d @openshift-ai-demos/automation/rhoai/usecases/fraud_detection/inputs/upi-fraud-detection.json

  Clean up deployment

    rhoai usecase cleanup fraud-detection \
      -c openshift-ai-demos/automation/config-fraud-detection.yaml
```

---

## Cleanup

```bash
# Remove use-case resources only
rhoai usecase cleanup fraud-detection -c config-fraud-detection.yaml

# Remove use-case resources AND platform (DSC + DSCI)
rhoai usecase cleanup fraud-detection -c config-fraud-detection.yaml --delete-platform
```

Removes each model's `InferenceService`, then its `ServingRuntime` (reversing the
creation order, in which the runtime is applied before the service). Pass
`--delete-platform` to also remove the DataScienceCluster and
DSCInitialization. When TrustyAI was deployed, cleanup adds a dedicated removal
phase (TrustyAI service, KServe logger restore, CA bundle, RBAC) after the models
are gone.

**Sample output:**
```
Removing   : fraud-detection
Namespace  : test-fraud


Removing model serving...

✔  Removing inference service 'upi-fraud-detection'  (1s)
✔  Waiting for inference service pods to terminate  (0s)
✔  Removing serving runtime 'triton-upi-fraud-detection'  (0s)
✔  Waiting for serving runtime pods to terminate  (0s)

✔  Model serving removed  (1s)

Removing staged models...

✔  Removing staging Pod for 'my-fraud-model-pvc'  (0s)
✔  Removing PVC 'my-fraud-model-pvc'  (5s)

✔  Staged models removed  (6s)

Cleanup complete.  Total: 8s

  Use case   : fraud-detection
  Namespace  : test-fraud

Removed

  ✔  upi-fraud-detection
  ✔  my-fraud-model-pvc (Pod + PVC)
```

---

## Troubleshooting

Common issues specific to this use case. For platform-wide problems (operator
install, timeouts, `rhoai: command not found`), see the
[Troubleshooting / FAQ](../../../README.md#troubleshooting--faq) section of the
automation README.

| Symptom | Cause / fix |
|---|---|
| Validation shows **Unavailable** after deploy | The endpoint isn't reachable from your machine. Add the ingress IP to `/etc/hosts` and rerun `rhoai usecase verify fraud-detection -c <config>`  |
| `InferenceService` stays `Unknown` | `model_uri` PVC not `Bound`, path wrong, or `kserve` not enabled. Check `oc get pvc -n <namespace>` and `rhoai platform status`. |
| Config rejected before any cluster call | An invalid input-mode combination — see the rejected combinations under [Inference inputs](#inference-inputs). |
| TrustyAI pod stays `Pending` | `trustyai` component not enabled, or no matching storage class. Run `rhoai platform enable trustyai`. |
| Ingestion never reaches the expected count | Raise `timeouts.ingestion_ready` on slow clusters |
