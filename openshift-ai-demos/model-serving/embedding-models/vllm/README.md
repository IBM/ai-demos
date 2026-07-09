# vLLM Embedding Runtime

Deploy and serve embedding models with the vLLM runtime on Red Hat OpenShift AI using Persistent Volume Claims (PVCs) for model storage.

This directory contains example deployments, testing instructions, and runtime configuration for serving embedding models with KServe and the OpenAI-compatible Embeddings API.

---

## Prerequisites

> Complete the [Getting Started](/openshift-ai-demos/README.md#getting-started) steps before proceeding. Ensure KServe is in `Managed` state and `DataScienceCluster` is in `Ready` state.

Before deploying a model, ensure you have:

- The following tools installed:
  - `oc`
  - `curl`
  - `jq`
  - `envsubst`
- A Hugging Face access token (`HF_TOKEN`) if the model requires authentication.

1. Create a data science project:
   ```bash
   export NAMESPACE=<your-project>
   oc new-project $NAMESPACE || oc project $NAMESPACE
   ```

2. Navigate to the OpenShift AI demos directory:
   ```bash
   cd ai-demos/openshift-ai-demos
   ```

3. Verify that a `StorageClass` is available for provisioning Persistent Volume Claims (PVCs):
   ```bash
   oc get storageclass
   ```

4. Create the vLLM CPU runtime:
   ```bash
   oc process -n redhat-ods-applications vllm-cpu-runtime-template | oc apply -f -
   ```

---

## Deploy Embedding Models

The following examples demonstrate how to deploy embedding models using the vLLM CPU runtime.

For each model:

- Download the model to a Persistent Volume Claim (PVC).
- Deploy the corresponding InferenceService.
- Verify that the deployment is ready.
- Test the model using the OpenAI-compatible Embeddings API.


### 1. Granite Embedding 125M English

**Model:** [ibm-granite/granite-embedding-125m-english](https://huggingface.co/ibm-granite/granite-embedding-125m-english)  
**Dimensions:** 768  
**Max tokens:** 512  
**Total size:** ~ 1GB  

**Resource Requirements:**
| Resource | Allocation |
|---|---|
| CPU | 8 cores |
| Memory | 16Gi |
| Storage | 2Gi PVC |

**Download the model to a persistent volume:**
  ```bash
  export MODEL_REPO=ibm-granite/granite-embedding-125m-english
  export MODEL_NAME=granite-embedding-125m-english
  export STORAGE_SIZE=2Gi
  export STORAGE_CLASS=$(oc get storageclass -o jsonpath='{.items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")].metadata.name}')

  envsubst '${MODEL_NAME},${NAMESPACE},${STORAGE_SIZE},${STORAGE_CLASS},${MODEL_REPO},${HF_TOKEN}' \
    < model-serving/shared/model-download.yaml | oc apply -f -
  ```

> To monitor the download, verify the downloaded files, and clean up the downloader pod, see [PVC Model Storage](../../README.md#option-2-pvc-huggingface-download).

**Deploy the model:**
```bash
oc apply -f model-serving/embedding-models/vllm/granite-125m-english.yaml
```

---

### 2. BGE-M3

**Model:** [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)  
**Dimensions:** 1024  
**Max tokens:** 8192  
**Total size:** ~ 4.59GB

**Resource Requirements:**
| Resource | Allocation |
|---|---|
| CPU | 8 cores |
| Memory | 16Gi |
| Storage | 5Gi PVC |

**Download the model to a persistent volume:**
```bash
export MODEL_REPO=BAAI/bge-m3
export MODEL_NAME=bge-m3
export STORAGE_SIZE=5Gi
export STORAGE_CLASS=$(oc get storageclass -o jsonpath='{.items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")].metadata.name}')

envsubst '${MODEL_NAME},${NAMESPACE},${STORAGE_SIZE},${STORAGE_CLASS},${MODEL_REPO},${HF_TOKEN}' \
  < model-serving/shared/model-download.yaml | oc apply -f -
```

> To monitor the download, verify the downloaded files, and clean up the downloader pod, see [PVC Model Storage](../../README.md#option-2-pvc-huggingface-download).

**Deploy the model:**
```bash
oc apply -f model-serving/embedding-models/vllm/bge-m3.yaml
```

---

### Verify the deployment

```bash
oc get inferenceservice

oc get pods -w | grep predictor
```

---

## Test the Model
The examples below use the OpenAI-compatible Embeddings API exposed by the vLLM runtime.
```bash
ISVC=<inferenceservice-name>
MODEL_URL=$(oc get inferenceservice $ISVC -o jsonpath='{.status.url}')
MODEL_NAME=$(curl -sk $MODEL_URL/v1/models | jq -r '.data[0].id')
```

> **Security Note:** The `-k` flag disables SSL certificate verification. Only use in development/testing environments.

### 1. List Models
Lists all models currently served by the runtime.

```bash
curl -sk $MODEL_URL/v1/models | jq .
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "bge-m3",
      "object": "model",
      "created": 1783575578,
      "owned_by": "vllm",
      "root": "/mnt/models",
      "parent": null,
      "max_model_len": 8192,
      "permission": [
        {
          "id": "modelperm-ad65d9624eb9de03",
          "object": "model_permission",
          "created": 1783575578,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}
```

### 2. Generate an Embedding
Generate an embedding for a single text input using the OpenAI-compatible Embeddings API.

```bash
curl -sk $MODEL_URL/v1/embeddings \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_NAME\",\"input\":\"What is machine learning?\"}" | jq .
```

```json
{
  "id": "embd-bf25400d42e0341b",
  "object": "list",
  "created": 1783575634,
  "model": "bge-m3",
  "data": [
    {
      "index": 0,
      "object": "embedding",
      "embedding": [
        -0.046485308557748795,
        -0.022186171263456345,
        -0.030788971111178398,
        .
        .
        .
        0.0047919112257659435,
        0.02460099197924137,
        0.007169000804424286
      ]
    }
  ],
  "usage": {
    "prompt_tokens": 7,
    "total_tokens": 7,
    "completion_tokens": 0,
    "prompt_tokens_details": null
  }
}
```

### 3. Batch Input
Generate embeddings for multiple text inputs in a single request.

```bash
curl -sk $MODEL_URL/v1/embeddings \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_NAME\",\"input\":[\"What is machine learning?\",\"How does RAG work?\",\"Explain vector search\"]}" \
  | jq '[.data[] | {index: .index, dimensions: (.embedding | length)}]'
```
```json
[
  {
    "index": 0,
    "dimensions": 1024
  },
  {
    "index": 1,
    "dimensions": 1024
  },
  {
    "index": 2,
    "dimensions": 1024
  }
]
```

### 4. Check Embedding Dimensions
Return the number of dimensions in the embedding vector.

```bash
curl -sk $MODEL_URL/v1/embeddings \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_NAME\",\"input\":\"test\"}" \
  | jq '.data[0].embedding | length'
```

```json
1024
```

### 5. Pooling API (vLLM native)
vLLM's native pooling endpoint that returns the same embedding vectors as `/v1/embeddings` in a different response format.

```bash
curl -sk $MODEL_URL/pooling \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_NAME\",\"input\":\"What is machine learning?\"}" | jq .
```

```json
{
  "id": "pooling-9867716949c9d145",
  "object": "list",
  "created": 1783575818,
  "model": "bge-m3",
  "data": [
    {
      "index": 0,
      "object": "pooling",
      "data": [
        -0.046485308557748795,
        -0.022186171263456345,
        -0.030788971111178398,
        -0.0025280159898102283,
        .
        .
        .
        0.0047919112257659435,
        0.02460099197924137,
        0.007169000804424286
      ]
    }
  ],
  "usage": {
    "prompt_tokens": 7,
    "total_tokens": 7,
    "completion_tokens": 0,
    "prompt_tokens_details": null
  }
}
```

---

## vLLM Configuration Reference

### Args

| Argument | Description |
|---|---|
| `--served-model-name` | Model name exposed via the API (`/v1/models`). Clients use this to call the model. |
| `--max-model-len` | Maximum context length (input plus generated tokens for generation models). Requests exceeding this limit are rejected. For embedding models, it limits the maximum input length. |
| `--max-num-seqs` | Maximum number of sequences (requests) processed concurrently in a single iteration. |
| `--dtype=bfloat16` | Weight precision. Uses half the memory of `float32` with minimal accuracy loss. |
| `--disable-custom-all-reduce` | Disables custom CUDA all-reduce kernel |
| `--trust-remote-code` | Allows loading custom model code from HuggingFace (needed for some models). |
| `--gpu-memory-utilization` | Fraction of available memory that vLLM is allowed to use. On GPU deployments, it limits GPU VRAM used for model execution and the KV cache. On CPU deployments, it limits the fraction of system RAM that the CPU backend may use during initialization. |
| `--hf-overrides` | Overrides the Hugging Face model configuration, such as enabling Matryoshka dimensions. |

### Environment Variables

| Variable | Description |
|---|---|
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | Timeout in seconds for a single engine iteration. Set high for CPU inference (e.g. `1200`). |
| `VLLM_CPU_KVCACHE_SPACE` | Amount of CPU memory (GB) reserved for the KV cache. |
