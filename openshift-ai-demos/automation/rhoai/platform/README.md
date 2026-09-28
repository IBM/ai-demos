# rhoai platform — CLI Reference

The `rhoai platform` command group manages the full lifecycle of a Red Hat OpenShift AI
(RHOAI) installation: operator install, DSCInitialization, component management,
status reporting, and clean uninstall.

Prerequisites and installation: see the [rhoai-automation README](../../README.md#getting-started).

---

## Output verbosity

Commands run in quiet mode by default. Pass `--log-level DEBUG`to see the full log stream:

```bash
# Normal output — structured summary only
rhoai platform init --channel stable-3.x

# Full log stream — useful when debugging failures
rhoai --log-level DEBUG platform init --channel stable-3.x
```

> **Flag position matters:** `--log-level` must come _before_ `platform`, not after it.

See [README.md Command reference](../../README.md#command-reference) for details.

---

## Configuration

Key defaults (full list in [`rhoai/config/defaults.yaml`](../config/defaults.yaml)):

```yaml
operator:
  name: rhods-operator
  namespace: redhat-ods-operator
  channel: stable
  source: redhat-operators
  source_namespace: openshift-marketplace

platform:
  namespace: redhat-ods-applications

dsc:
  name: default-dsc
  dsci_name: default-dsci

components: []   # used by 'setup'
```

Pass `--config /path/to/my-config.yaml` to any command to override defaults. See
[README.md Documentation](../../README.md#documentation) for how values are merged.

---

## `rhoai platform init`

Validates cluster prerequisites, installs the RHOAI operator via OLM, and initialises
the DSCInitialization (DSCI). Does **not** enable any DSC components, run
`enable` or `setup` afterwards.

**Sequence:**
1. Validate cluster login
2. Validate / confirm storage classes
3. Create operator namespace (`redhat-ods-operator`) if absent
4. Create workload namespace (`redhat-ods-applications`) if absent
5. Apply OperatorGroup + Subscription → wait for CSV `Succeeded`
6. Approve InstallPlan automatically if `--version` pin was requested (Manual approval)
7. Apply DSCI manifest → wait for `Ready`

### Flags

| Flag | Default | Description |
|---|---|---|
| `--channel` | `stable` | OLM channel. See available channels below. |
| `--version` | _(none)_ | Pin to a specific CSV. Accepts bare semver or full CSV name. |
| `--source` | `redhat-operators` | CatalogSource name. |
| `--config`, `-c` | _(none)_ | Path to a config YAML file. |

### Already-installed operator

When `init` is run on a cluster that already has the operator installed (a re-run, a
re-init after a previous partial failure, or a second `init` with a different channel
flag), it **does not reinstall or modify the Subscription**. Instead it:

1. Detects that the CSV is already in `Succeeded` phase.
2. Waits for the existing CSV to be ready (returns immediately, since it is already
   `Succeeded`) instead of creating or modifying the Subscription.
3. Continues to apply / wait for DSCI as normal.

The output is the same as a fresh install, no separate "skipping" message is printed:

```
Initializing RHOAI platform...

  ✔  RHOAI platform initialized
     Operator  : rhods-operator.3.4.2  3.4.2
     Platform  : Ready
```

This means `init` is **safe to re-run**, it will never accidentally upgrade or
downgrade an already-running operator. 

> **Version-agnostic check:** the "already installed" detection is based solely on the
> CSV being in `Succeeded` phase — it does **not** compare the installed version against
> the `--version` or `--channel` flags you pass. 

### Discovering channels and versions

```bash
# List all available channels for rhods-operator
oc get packagemanifest rhods-operator \
  -o jsonpath='{.status.channels[*].name}'

# Show which CSV a specific channel currently resolves to
oc get packagemanifest rhods-operator \
  -o jsonpath='{range .status.channels[?(@.name=="stable-3.x")]}{.currentCSV}{"\n"}{end}'
```

**Notable channels:**

| Channel | Description |
|---|---|
| `stable` | Latest stable release (long-term support line) |
| `stable-<N>.x` | Floating latest within major version `N` (e.g. `stable-3.x`) |
| `stable-<N>.<M>` | Pinned to a specific minor line (e.g. `stable-3.5`) |
| `fast-<N>.x` | Fast lane for major version `N`, promoted ahead of stable |
| `beta` | Pre-GA builds — **see note below** |
| `alpha` | Alpha builds |

> **Channel names are not validated by the CLI.** They are passed directly to OLM,
> which rejects unknown channel names with a clear `ConstraintsNotSatisfiable` error.
> Use the `oc get packagemanifest` commands above to discover valid channel names for
> your cluster's catalog before running `init`.

### Examples

```bash
# Latest 3.x from the standard catalog
rhoai platform init --channel stable-3.x

# Pinned to a specific patch (bare semver — normalised automatically)
rhoai platform init --channel stable-3.4 --version 3.4.2

# Full CSV name also accepted
rhoai platform init --channel stable-3.4 --version rhods-operator.3.4.2

# With an explicit config file
rhoai platform init --config /path/to/my-config.yaml --channel stable-3.x
```

> **Version normalisation:** bare semver `3.4.2` is automatically expanded to
> `rhods-operator.3.4.2`. The `v` prefix (e.g. `v3.4.2`) is **not** used in RHOAI
> CSV names.

> **Manual approval:** when `--version` is supplied the Subscription is created with
> `installPlanApproval: Manual` and the pending InstallPlan is automatically approved
> by the CLI — OLM will not auto-upgrade beyond the pinned CSV.

---

### Pre-GA install (`--source` / `beta` channel)

> ⚠️ **Work in progress** — automated pre-GA install support is not yet complete.
> The steps below describe what the CLI will handle end-to-end once implemented.
> For now, perform these steps manually before running `rhoai platform init`.

Installing a pre-GA build requires three cluster-side prerequisites that are not yet
automated:

#### 1. Pull secret for `quay.io/rhoai`

```bash
oc registry login \
  --registry=quay.io/rhoai \
  --auth-basic='<robot-account>:<token>' \
  --to=/tmp/rhoai-pull.json

oc set data secret/pull-secret \
  -n openshift-config \
  --from-file=.dockerconfigjson=/tmp/rhoai-pull.json
```

#### 2. ImageContentSourcePolicy (ICSP)

Apply the ICSP manifest provided with the EA release to redirect pulls from the
public registry to the internal mirror.

#### 3. CatalogSource

```bash
cat <<EOF | oc apply -f -
apiVersion: operators.coreos.com/v1alpha1
kind: CatalogSource
metadata:
  name: cs-rhoai-fbc-fragment
  namespace: openshift-marketplace
spec:
  sourceType: grpc
  image: quay.io/rhoai/rhoai-fbc-fragment:<ea-tag>
  displayName: RHOAI Pre-GA
  publisher: Red Hat
EOF

oc get catalogsource cs-rhoai-fbc-fragment -n openshift-marketplace -w
```

Once all three prerequisites are in place:

```bash
rhoai platform init --channel beta --source cs-rhoai-fbc-fragment
```

---

## `rhoai platform enable`

Enables one or more DSC components on an already-initialised platform. Requires
`init` to have run first.

**Behaviour:**
- Additive and idempotent — only the named components are patched to `Managed`
- Components not named are left exactly as they are (no accidental disables)
- If no DataScienceCluster exists yet, one is created from the base manifest before patching
- **If the operator is not installed, the command fails immediately** with
  `Operator not found. Run 'rhoai platform init' first.` — it does not attempt to
  install anything.

### Syntax

```bash
rhoai platform enable <component> [<component> ...]
```

### Available components

| Component | Description |
|---|---|
| `dashboard` | RHOAI web dashboard |
| `kserve` | Model serving (KServe) |
| `workbenches` | Jupyter notebooks |
| `modelregistry` | Model registry |
| `aipipelines` | AI/ML pipelines (Argo-based) |
| `trustyai` | TrustyAI bias / explainability |
| `ray` | Distributed compute (Ray) |
| `kueue` | Job queue management |
| `trainingoperator` | Kubeflow Training Operator |
| `trainer` | Training workflows |
| `mlflowoperator` | MLflow experiment tracking |
| `feastoperator` | Feature store (Feast) |
| `llamastackoperator` | LlamaStack operator |
| `ogx` | OpenShift AI Extensions |
| `sparkoperator` | Apache Spark operator |

### Output

After patching the named components, `enable` echoes the components it just enabled
followed by every component currently `Managed` in the DataScienceCluster:

```
Enabling: dashboard, kserve

  ✔  Enabled: dashboard, kserve

  Components
    ✔  dashboard
    ✔  kserve
    ✔  workbenches
```

### Examples

```bash
# Enable a single component
rhoai platform enable dashboard

# Enable multiple components in one call
rhoai platform enable dashboard kserve workbenches modelregistry

# Common production set
rhoai platform enable dashboard kserve workbenches modelregistry aipipelines trustyai

# With a config file
rhoai platform enable kserve --config /path/to/my-config.yaml
```

---

## `rhoai platform disable`

Disables one or more DSC components (sets `managementState` to `Removed`). Requires
`init` to have run first and a DataScienceCluster to already exist.

**Behaviour:**
- Idempotent — components already `Removed` stay `Removed`
- Components not named are left exactly as they are (no accidental enables)
- Raises an error immediately if the component name is unknown (typo guard)

### Syntax

```bash
rhoai platform disable <component> [<component> ...]
```

### Examples

```bash
# Disable a single component
rhoai platform disable trustyai

# Disable multiple components in one call
rhoai platform disable ray kueue trainingoperator
```

> **Components to disable must be passed as CLI arguments** — the config file has no
> `disable:` key and cannot express which components to turn off. The `--config` flag
> only carries operator/cluster/namespace settings.

> **Note:** `disable` requires the DataScienceCluster to already exist. If it does
> not (i.e. `enable` or `setup` was never run), the command will exit with an error.

---

## `rhoai platform setup`

Full one-shot bootstrap — runs `init` followed by enabling DSC components.

### Flags

| Flag | Default | Description |
|---|---|---|
| `--channel` | `stable` | OLM channel. Overrides `operator.channel` in config. |
| `--version` | _(none)_ | Pin to a specific CSV. Accepts bare semver or full CSV name. |
| `--source` | `redhat-operators` | CatalogSource name. |
| `--config`, `-c` | _(none)_ | Path to a config YAML file. |

**Two modes:**

| Mode | When | Behaviour |
|---|---|---|
| **Full** | `components` not set in config (default) | Applies `shared/dsc.yaml` as-is — enables whatever is `Managed` in that file |
| **Selective** | `components` list set in config | Merge-patches only the listed components to `Managed`; **every other component keeps its current state** (not reset to `Removed`) |

Use `setup` when you want a single command to go from zero to a running platform.
Use `init` + `enable`/`disable` when you need day-2 fine-grained control.

> **Currently Config file can only enable components, not disable them.**  
> The `components:` key is a flat list of names to set `Managed`. There is no
> `disable:` / `enable:` sub-key — passing a nested structure would cause an error.
> `setup` never calls `remove_component()` internally, so it cannot turn a component
> off via config. To disable components, use `rhoai platform disable <component>`
> after `setup`:
>
> ```bash
> rhoai platform setup --config my-config.yaml   # enables listed components
> rhoai platform disable dashboard ray            # disables what you don't want
> ```

```bash
# Full install — applies dsc.yaml defaults
rhoai platform setup

# Selective install — only the listed components are enabled
rhoai platform setup --config my-config.yaml
```

**Config file for selective setup:**

```yaml
# my-config.yaml
operator:
  channel: stable-3.x
  source: redhat-operators
components:
  - dashboard
  - kserve
  - workbenches
  - modelregistry
```

```bash
rhoai platform setup --config my-config.yaml
```

The same [already-installed operator](#already-installed-operator) logic from `init`
applies here — `setup` will never reinstall the operator if the CSV is already
`Succeeded`.

---

## `rhoai platform uninstall`

Performs a **complete teardown** of all RHOAI platform resources, leaving the cluster
clean for a fresh reinstall.

**Default behaviour: workload namespaces are deleted.**  
Pass `--keep-workload-ns` to preserve user notebooks, pipelines, and inference
services across the reinstall.

**Deletion sequence:**
1. DataScienceCluster — operator deprovisions components gracefully first, then waits
   up to 120s (hardcoded) for component pods to drain
2. DSCInitialization
3. ClusterServiceVersion (CSV), Subscription, OperatorGroup, InstallPlans
4. Cluster-wide sweep of any stale rhods/rhoai/opendatahub CSVs
5. Dependency operators auto-detected on the cluster:
   - `servicemeshoperator3` (installed when NIM / gateway routing is enabled)
   - `servicemeshoperator` (legacy Maistra v2)
   — each is **only** cleaned up if its Subscription is present; the CLI never touches
   operators it did not install
6. CRDs registered by the operator (label-based + pattern sweep)
7. Stale APIServices — removed before namespace deletion to prevent
   `NamespaceDeletionDiscoveryFailure` from keeping namespaces stuck in `Terminating`
8. Validating and mutating webhooks matching rhoai/rhods/kserve/trustyai patterns
9. Cluster-scoped RBAC (ClusterRoles / ClusterRoleBindings)
10. Operator namespace (`redhat-ods-operator`)
11. Workload namespaces: `redhat-ods-applications`, `rhods-notebooks`,
    `rhoai-model-registries`, `redhat-ods-monitoring`
    — also any namespace labelled `opendatahub.io/generated-namespace=true`
    — skipped automatically if never created

Every step is safe on a partially-installed cluster — missing resources are skipped
with a log message rather than raising an error.

**Stuck namespace handling:** if any namespace gets stuck in `Terminating` after
deletion, the CLI automatically cleans stale APIServices, strips all resource
finalizers inside the namespace, and removes the namespace finalizer — then waits
for removal to complete.

### Flags

| Flag | Default | Description |
|---|---|---|
| `--yes`, `-y` | `false` | Skip the confirmation prompt |
| `--keep-workload-ns` | _(not set)_ | Preserve workload namespaces instead of deleting them |
| `--config`, `-c` | _(none)_ | Path to a config YAML file (only needed for non-default namespaces or resource names) |

> **Default is to delete workload namespaces.** The primary use case is testing and
> development where a clean cluster state is required for each reinstall. Pass
> `--keep-workload-ns` only when you want to preserve user notebooks, pipelines,
> inference services, and PVCs across a platform reinstall.

### Examples

```bash
# Full clean wipe — cluster ready for fresh reinstall (default)
rhoai platform uninstall -y

# Interactive prompt
rhoai platform uninstall

# Preserve user workloads (notebooks, pipelines, models)
rhoai platform uninstall -y --keep-workload-ns

# Non-default namespaces — must pass the same config used at install time
rhoai platform uninstall --config /path/to/my-config.yaml -y
```

> **When is `--config` needed?**  
> `uninstall` uses the config only to know **where to look**: operator namespace,
> workload namespace, DSC name, and DSCI name. All of these default to the standard
> RHOAI values (`redhat-ods-operator`, `redhat-ods-applications`, `default-dsc`,
> `default-dsci`). If you installed with a custom config that changed any of these,
> pass the same config to `uninstall` — otherwise it will look in the wrong places,
> find nothing, and silently skip every deletion step.

---

## `rhoai platform status`

Reports the health of the installed RHOAI platform.

```bash
rhoai platform status
```

**Sample output:**

```
RHOAI Platform
  Operator              rhods-operator.<version>    ✔  Succeeded
  Initialization        default-dsci                ✔  Ready
  DataScienceCluster    default-dsc                 ✔  Ready

  Components
    ✔  aipipelines
    ✔  dashboard
    ✔  kserve
    ✔  modelregistry
    ✔  workbenches

Platform is healthy.
```

Exits with code `1` if any check fails, with an actionable hint:

```
Platform is not healthy.
  ℹ  Run 'rhoai platform setup' to reinstall.
  ℹ  For details run: rhoai --log-level DEBUG platform status
```

---

## `rhoai platform inspect`

Displays factual cluster information. Read-only — makes no changes.

```bash
rhoai platform inspect
```

**Sample output:**

```
Dumping cluster info...

⚙  Cluster
  OpenShift    <version>
  Topology     Multi-node  (6 nodes: 3 master, 3 workers)

⛶  Worker Nodes
  NAME              CPU         MEMORY
  <node-1>          4 cores     15 GiB
  <node-2>          4 cores     15 GiB
  <node-3>          4 cores     15 GiB

💾  Storage Classes
  NAME          BOUND
  <sc-name>     40 GiB
```

---

## Typical workflows

### Fresh install — selective components

```bash
oc login <cluster-url>
rhoai platform init --channel stable-3.x
rhoai platform enable dashboard kserve workbenches modelregistry
rhoai platform status
```

### Fresh install — one-shot via config

```bash
# my-config.yaml: operator.channel + components list
rhoai platform setup --config my-config.yaml
rhoai platform status
```

### Day-2 component changes

```bash
# Add a component after initial setup
rhoai platform enable trustyai

# Remove a component that is no longer needed
rhoai platform disable ray kueue
```

### Re-run init safely on an already-installed cluster

```bash
# Safe — detects existing CSV, skips install, re-applies DSCI if needed
rhoai platform init --channel stable-3.x
```

### Teardown and reinstall (fresh cluster state)

```bash
# Full wipe — deletes all namespaces, CRDs, webhooks, RBAC
rhoai platform uninstall -y

# Reinstall
rhoai platform init --channel stable-3.x
rhoai platform enable dashboard kserve workbenches modelregistry
```

### Teardown preserving user workloads

```bash
# Removes the operator but keeps notebooks, pipelines, and models
rhoai platform uninstall -y --keep-workload-ns
```

### Debug a failing command

```bash
# Run any command with full log output
rhoai --log-level DEBUG platform init --channel stable-3.x
rhoai --log-level DEBUG platform status
```

### Using a config file (CI/CD)

```yaml
# rhoai-config.yaml
operator:
  channel: stable-3.x
  source: redhat-operators
components:
  - dashboard
  - kserve
  - workbenches
  - modelregistry
  - aipipelines
```

```bash
rhoai platform setup --config rhoai-config.yaml
```
