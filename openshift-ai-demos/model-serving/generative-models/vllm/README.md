# vLLM Runtime

Deploy and serve Large Language Models (LLMs) using vLLM runtime on Red Hat OpenShift AI.

---

## Prerequisites

1. Create a data science project:
   ```bash
   oc new-project <your-project> || oc project <your-project>
   ```

2. Navigate to the OpenShift AI demos directory:
   ```bash
   cd ai-demos/openshift-ai-demos
   ```

3. Create a model serving runtime:
   ```bash
   oc process -n redhat-ods-applications vllm-cpu-runtime-template | oc create -f -
   ```
---

## Available Models

### Phi-3-mini-4k-instruct

**Deploy:**
```bash
oc apply -f model-serving/generative-models/vllm/phi-3-mini-4k-instruct.yaml -n <your-project>
```

**Minimum Resource Requirements:**
| Resource | Allocation    |
| -------- | ------------- |
| Runtime  | vLLM (CPU)    |
| CPU      | 6 to 10 cores |
| Memory   | 16Gi to 20Gi  |

> These are the minimum resources specified in the InferenceService YAML. Actual usage may vary based on workload.


### Qwen2.5-1.5b-Instruct

**Deploy:**
```bash
oc apply -f model-serving/generative-models/vllm/qwen2.5-1.5b-instruct.yaml -n <your-project>
```

**Minimum Resource Requirements:**
| Resource | Allocation    |
| -------- | ------------- |
| Runtime  | vLLM (CPU)    |
| CPU      | 32 cores      |
| Memory   | 40Gi          |

> These are the minimum resources specified in the InferenceService YAML. Actual usage may vary based on workload.

> **Note:** Resource requirements vary based on model size, architecture, and vLLM configuration. Larger models or higher concurrency settings require more resources.

---

## Customizing vLLM Options

You can customize vLLM behavior by adding command-line arguments in the InferenceService YAML under `spec.predictor.model.args`.

**Example:**
```yaml
spec:
  predictor:
    model:
      modelFormat:
        name: vLLM
      args:
      - --disable-sliding-window
      - --max-model-len=4096
```

**Common vLLM Options:**
- `--max-model-len`: Maximum sequence length (default: model's max)
- `--disable-sliding-window`: Disable sliding window attention


For a complete list of vLLM options, see the [vLLM documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server).

---

## Verify Deployment

> **Deployment Time:** Model deployment typically takes 5-10 minutes depending on model size, S3 download speed, and cluster resources.

```bash
# List InferenceServices
oc get inferenceservice

# Check status (e.g., phi3 or qwen)
oc describe inferenceservice <inferenceservice-name>

# Watch predictor pod status (e.g., phi3-predictor-xxxxx)
oc get pods -w | grep predictor
```

---

## Test the Model

```bash
# Get model URL (replace <inferenceservice-name> with phi3 or qwen)
MODEL_URL=$(oc get inferenceservice <inferenceservice-name> -o jsonpath='{.status.url}')

# Get model name (requires jq)
MODEL_NAME=$(curl -sk "$MODEL_URL/v1/models" | jq -r '.data[0].id')

# Send test request
curl -sk -X POST "$MODEL_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_NAME\",
    \"prompt\": \"What is machine learning?\",
    \"max_tokens\": 100
  }"
```

> **Security Note:** The `-k` flag in curl commands disables SSL certificate verification and should only be used in development/testing environments. For production, configure proper certificates or remove the `-k` flag.
