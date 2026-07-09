# Model Serving on Red Hat OpenShift AI

Deploy and serve models on Red Hat OpenShift AI using KServe InferenceService.

---

## Available Model Categories

### Generative Models
Deploy and serve Large Language Models (LLMs) and other generative AI models on Red Hat OpenShift AI.

[→ Explore Generative Models](generative-models/)

---

### Embedding Models
Deploy and serve embedding models for vector search, semantic search, and RAG pipelines on Red Hat OpenShift AI.

[→ Explore Embedding Models](embedding-models/)

---

### Safety Models
Deploy content moderation, guardrails, and safety detection models on Red Hat OpenShift AI.

[→ Explore Safety Models](safety-models/)

---

## Model Storage Options

### Option 1: S3 Object Storage

Models stored in S3-compatible object storage with the following structure:

```
s3://<bucket-name>/
└── models/
    ├── <model-name>/
    │   └── [model files]
    └── <another-model>/
        └── [model files]
```

Update the `storage` section in each model's YAML file to match your S3 bucket structure:

```yaml
storage:
  key: s3-creds
  path: models/<model-name>
```

---

### Option 2: PVC (HuggingFace Download)

Download models directly from HuggingFace into a PersistentVolumeClaim using the  [model-downloader.yaml](shared/model-download.yaml). This is an alternative to S3 when object storage is not available.

**Download a model:**
```bash
export NAMESPACE=<your-project>
export MODEL_REPO=<hf-org>/<hf-model-name>   # e.g. ibm-granite/granite-embedding-125m-english
export MODEL_NAME=<model-name>                # must be lowercase RFC 1123 (e.g. prompt-guard, not Prompt-Guard)
export STORAGE_SIZE=5Gi
export STORAGE_CLASS=<your-storage-class>
export HF_TOKEN=<your-hf-token>               # optional, needed for private/gated models

envsubst '${MODEL_NAME},${NAMESPACE},${STORAGE_SIZE},${STORAGE_CLASS},${MODEL_REPO},${HF_TOKEN}' \
  < model-serving/shared/model-download.yaml | oc apply -f -
```

**Monitor the download:**
```bash
oc logs -f ${MODEL_NAME}-downloader
```

**Verify files after download:**
```bash
oc run verify --image=registry.access.redhat.com/ubi9/python-312:9.8-1783442686 \
  --restart=Never --rm -it \
  --overrides='{"spec":{"securityContext":{"runAsUser":1000,"fsGroup":1000},"containers":[{"name":"verify","image":"registry.access.redhat.com/ubi9/python-312:9.8-1783442686","command":["ls","-lh","/mnt/models/'"$MODEL_NAME"'/"],"volumeMounts":[{"mountPath":"/mnt/models","name":"v"}]}],"volumes":[{"name":"v","persistentVolumeClaim":{"claimName":"'"$MODEL_NAME"'-pvc"}}]}}'
```

**Clean up downloader pod:**
```bash
oc delete pod ${MODEL_NAME}-downloader
```

**Reference the PVC in your InferenceService:**
```yaml
storageUri: pvc://<model-name>-pvc/<model-name>
```

---

## Troubleshooting

**Model fails to load:**
- Verify S3 credentials in [`s3-secret.yaml`](/openshift-ai-demos/shared/s3-secret.yaml)
- Confirm model path exists in S3 bucket (check `storageUri` in YAML)
- Check pod logs: `oc logs <predictor-pod-name>`
- Verify S3 endpoint is accessible from cluster

**PVC download fails:**
- Check downloader pod logs: `oc logs ${MODEL_NAME}-downloader`
- Ensure `HF_TOKEN` is set for private/gated models
- Verify storage class exists: `oc get storageclass`

**Insufficient resources:**
- Adjust resource limits in YAML files
- Verify cluster capacity: `oc describe nodes`
- Check if nodes have required CPU/memory available

---

## Resources

- [KServe Documentation](https://kserve.github.io/website/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Red Hat OpenShift AI Docs](https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/)
