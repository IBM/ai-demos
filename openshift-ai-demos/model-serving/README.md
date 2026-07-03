# Model Serving on Red Hat OpenShift AI

Deploy and serve models on Red Hat OpenShift AI using KServe InferenceService.


---

## Prerequisites

Before deploying models, complete the setup steps described in the [OpenShift AI README](/openshift-ai-demos/README.md#getting-started):

1. Configure `DSCInitialization` and `DataScienceCluster`
2. Create the S3 secret with your object storage credentials

---

## Available Model Categories

### Generative Models
Deploy and serve Large Language Models (LLMs) and other generative AI models on Red Hat OpenShift AI.

[→ Explore Generative Models](generative-models/)

---

### Safety Models
Deploy content moderation, guardrails, and safety detection models on Red Hat OpenShift AI.

[→ Explore Safety Models](safety-models/)

---

## S3 Model Storage

Models must be stored in your S3-compatible object storage with the following structure:

```
s3://<bucket-name>/
└── models/
    ├── <model-name>/
    │   └── [model files]
    └── <another-model>/
        └── [model files]
```

Update the `spec.predictor.model.storage.path` value in each model's YAML file
to match your S3 bucket structure.

---

## Upload Model Artifacts to S3

The model-serving manifests read model files from the bucket configured in
[`s3-secret.yaml`](/openshift-ai-demos/shared/s3-secret.yaml). Download model
artifacts locally, upload them to the matching `models/<model-name>/` prefix,
then verify that the files are visible before applying an InferenceService.

### 1. Configure the S3 CLI

Install and configure the AWS CLI for your S3-compatible storage provider:

```bash
aws configure
```

If you use a non-AWS endpoint, set it once for the commands in this guide:

```bash
export S3_ENDPOINT_URL="https://<s3-endpoint>"

S3_ARGS=()
if [ -n "${S3_ENDPOINT_URL:-}" ]; then
  S3_ARGS+=(--endpoint-url "$S3_ENDPOINT_URL")
fi
```

### 2. Download model artifacts

Download each model into a local staging directory. For Hugging Face models, one
common option is `huggingface-cli`:

```bash
mkdir -p /tmp/rhoai-models

huggingface-cli download microsoft/Phi-3-mini-4k-instruct \
  --local-dir /tmp/rhoai-models/phi-3-mini-4k-instruct \
  --local-dir-use-symlinks False
```

Repeat the download for any additional model used by a demo.

### 3. Upload artifacts to the bucket

Upload the local model directory to the prefix referenced by the demo manifest:

```bash
aws s3 sync /tmp/rhoai-models/phi-3-mini-4k-instruct \
  s3://<bucket-name>/models/phi-3-mini-4k-instruct \
  "${S3_ARGS[@]}"
```

For the included safety model manifest, use the same pattern with the matching
prefix:

```bash
aws s3 sync /tmp/rhoai-models/granite-guardian-hap-125m \
  s3://<bucket-name>/models/granite-guardian-hap-125m \
  "${S3_ARGS[@]}"
```

### 4. Verify uploaded artifacts

List the target prefix and confirm that model files such as configuration,
tokenizer, and weight files are present:

```bash
aws s3 ls s3://<bucket-name>/models/phi-3-mini-4k-instruct/ \
  --recursive \
  "${S3_ARGS[@]}"
```

The prefix after `models/` must match the `spec.predictor.model.storage.path`
value in the InferenceService YAML. For example:

```yaml
storage:
  key: s3-creds
  path: models/phi-3-mini-4k-instruct
```

After the files are present and the `s3-creds` secret is applied in your data
science project namespace, deploy the model manifest with `oc apply`.

---

## Troubleshooting

**Model fails to load:**
- Verify S3 credentials in [`s3-secret.yaml`](/openshift-ai-demos/shared/s3-secret.yaml)
- Confirm model path exists in S3 bucket (check `storageUri` in YAML)
- Check pod logs: `oc logs <predictor-pod-name>`
- Verify S3 endpoint is accessible from cluster

**Insufficient resources:**
- Adjust resource limits in YAML files
- Verify cluster capacity: `oc describe nodes`
- Check if nodes have required CPU/memory available


## Resources

- [KServe Documentation](https://kserve.github.io/website/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Red Hat OpenShift AI Docs](https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/)
