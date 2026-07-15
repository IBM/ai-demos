# Generative Model Serving with vLLM

Deploy and serve Large Language Models (LLMs) with the vLLM runtime on Red Hat OpenShift AI.

This directory includes example model deployments, API usage examples, and configuration guidance for the vLLM runtime with KServe.

## Demo Video

**Deploy and Test the Qwen3 Model**

This video walks through creating the vLLM CPU `ServingRuntime`, deploying the Qwen3 model with KServe, and testing inference using the OpenAI-compatible APIs.

https://github.com/user-attachments/assets/1f16e203-7ece-4adb-aa6c-f19f88b377b8

---

## Prerequisites
> Complete the [Getting Started](/openshift-ai-demos/README.md#getting-started) steps before proceeding. Ensure KServe is in the `Managed` state and `DataScienceCluster` is in the `Ready` state.

Before deploying a model, ensure you have:

- The following tools installed:
  - `oc`
  - `curl`
  - `jq`

1. Create a data science project:
   ```bash
   oc new-project <your-project> || oc project <your-project>
   ```

2. Navigate to the OpenShift AI demos directory:
   ```bash
   cd ai-demos/openshift-ai-demos
   ```

3. Create the vLLM CPU `ServingRuntime`:
   ```bash
   oc process -n redhat-ods-applications vllm-cpu-runtime-template | oc apply -f -
   ```
---

## Model Storage

The example deployments use different model storage backends:

- **Phi-3-mini-4k-instruct** and **Qwen2.5-1.5B-Instruct** load models from an S3-compatible object storage backend.
- **Qwen3-4B-Instruct-2507** downloads the model from Hugging Face at deployment time using `storageUri`.

---

## Deploy Models

### 1. Phi-3-mini-4k-instruct

**Model:** [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)  
**Parameters:** 3.8B  
**Context:** 4K tokens  

A lightweight instruction-tuned model from the Phi-3 family by Microsoft. Designed for resource-constrained and low-latency environments, it provides strong performance on reasoning, mathematics, coding, and general language understanding tasks.

**Resource Requirements:**
| Resource | Allocation    |
| -------- | ------------- |
| CPU      | 6 to 10 cores |
| Memory   | 16Gi to 20Gi  |

**Deployment configuration:**
- Sliding window attention is disabled using `--disable-sliding-window`.

**Deploy:**
```bash
oc apply -f model-serving/generative-models/vllm/phi-3-mini-4k-instruct.yaml
```


### 2. Qwen2.5-1.5b-Instruct

**Model:** [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)  
**Parameters:** 1.5B  
**Context:** 32K tokens (up to 8K token generation) 

A compact instruction-tuned language model from the Qwen2.5 series by Alibaba Cloud. It provides strong performance on instruction following, reasoning, coding, mathematics, structured output generation (including JSON), and multilingual tasks while maintaining a relatively small resource footprint.  

**Resource Requirements:**
| Resource | Allocation    |
| -------- | ------------- |
| CPU      | 32 cores      |
| Memory   | 40Gi          |

**Deploy:**
```bash
oc apply -f model-serving/generative-models/vllm/qwen2.5-1.5b-instruct.yaml
```

### 3. Qwen3-4B-Instruct-2507

**Model:** [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)  
**Parameters:** 4B  
**Context:** 262K tokens (capped at 8196 in this deployment) 

An updated instruction-tuned model in the Qwen3 series from Alibaba Cloud. It provides improved instruction following, reasoning, coding, mathematics, multilingual understanding, tool use, and long context performance. This deployment uses the non-thinking variant with tool calling enabled via the Hermes parser.

**Resource Requirements:**
| Resource | Allocation |
| -------- | ---------- |
| CPU      | 32 cores   |
| Memory   | 40Gi       |

**Deployment configuration:**
- Tool calling enabled via `--enable-auto-tool-choice` with `--tool-call-parser=hermes`.
- Native context length is 262K tokens. This deployment limits it to 8196 using `--max-model-len` to reduce memory usage.
- GPU memory utilization is set to `0.4`.
- KV cache space is set to 12 GiB.

**Deploy:**
```bash
oc apply -f model-serving/generative-models/vllm/qwen3-4b-instruct-2507.yaml 
```

---

## Verify Deployment

> **Deployment Time:** Model deployment typically takes 5-10 minutes depending on model size, download speed, and cluster resources.

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
The following example uses the OpenAI-compatible Completions API exposed by the deployed vLLM runtime.

```bash
ISVC=<inferenceservice-name>

# Get the inference endpoint
MODEL_URL=$(oc get inferenceservice ${ISVC} -o jsonpath='{.status.url}')

# Retrieve the served model name
MODEL_NAME=$(curl -sk "${MODEL_URL}/v1/models" | jq -r '.data[0].id')

echo "Inference endpoint: ${MODEL_URL}"
echo "Model: ${MODEL_NAME}"
```

### 1. List Models

Lists all models currently served by the runtime.
```bash
curl -sk $MODEL_URL/v1/models | jq .
```

**Example output (truncated):**

```text
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-4b",
      "object": "model",
      "created": 1784120145,
      "owned_by": "vllm",
      "root": "/mnt/models",
      "parent": null,
      "max_model_len": 8196,
      "permission": [
        {
          .
          .
          .
        }
      ]
    }
  ]
}
```

### 2. Generate a Completion

Generate a text completion using the OpenAI-compatible Completions API.

```bash
curl -sk "${MODEL_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"prompt\": \"What is machine learning?\",
    \"max_tokens\": 100
  }" | jq
```

**Example output (truncated):**

```text
"choices": [
  {
    "index": 0,
    "text": " What are the types of machine learning?\n\nMachine learning is a subset of artificial intelligence (AI) that enables computers to learn from data and improve their performance on a task over time without being explicitly programmed...",
    .
    .
    .
  }
]
```

### 3. Chat Completion

Generate a response using the OpenAI-compatible Chat Completions API.

```bash
curl -sk "${MODEL_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": \"Explain Retrieval-Augmented Generation (RAG).\"
      }
    ],
    \"max_tokens\": 150
  }" | jq
```

**Example output (truncated):**

```text
"message": {
  "role": "assistant",
  "content": "**Retrieval-Augmented Generation (RAG)** is a technique in artificial intelligence that combines **information retrieval** and **natural language generation** to improve the accuracy, relevance,..."
  .
  .
  .
}
```

### 4. Stream Responses

Receive tokens as they are generated.

```bash
curl -skN "${MODEL_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": \"Write a haiku about Kubernetes.\"
      }
    ],
    \"stream\": true
  }"
```

**Example output (truncated):**

```text
data: {"id":"chatcmpl-9742048e12ff74e7","object":"chat.completion.chunk","created":1784121169,"model":"qwen3-4b","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}],"prompt_token_ids":null,"prompt_text":null}

data: {"id":"chatcmpl-9742048e12ff74e7","object":"chat.completion.chunk","created":1784121169,"model":"qwen3-4b","choices":[{"index":0,"delta":{"content":"Containers","tool_calls":[]},"logprobs":null,"finish_reason":null,"token_ids":null}]}

data: {"id":"chatcmpl-9742048e12ff74e7","object":"chat.completion.chunk","created":1784121169,"model":"qwen3-4b","choices":[{"index":0,"delta":{"content":" in","tool_calls":[]},"logprobs":null,"finish_reason":null,"token_ids":null}]}

data: {"id":"chatcmpl-9742048e12ff74e7","object":"chat.completion.chunk","created":1784121169,"model":"qwen3-4b","choices":[{"index":0,"delta":{"content":" clusters","tool_calls":[]},"logprobs":null,"finish_reason":null,"token_ids":null}]}
.
.
.
data: {"id":"chatcmpl-9742048e12ff74e7","object":"chat.completion.chunk","created":1784121169,"model":"qwen3-4b","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"system_fingerprint":"vllm-0.21.0-7b9ac82d"}

data: [DONE]
```

> **Security Note:** The `-k` flag in curl commands disables SSL certificate verification and should only be used in development/testing environments. For production, configure proper certificates or remove the `-k` flag.

---

## Customizing vLLM Configuration

You can customize the vLLM runtime by configuring command-line arguments in `spec.predictor.model.args` and environment variables in `spec.predictor.model.env` of the `InferenceService`.

For example:

```yaml
spec:
  predictor:
    model:
      modelFormat:
        name: vLLM
      args:
        - --max-model-len=4096
      env:
        - name: VLLM_CPU_KVCACHE_SPACE
          value: "12"
```

The following sections describe some commonly used vLLM command-line arguments and environment variables.

## vLLM Configuration Reference

### Args

| Argument | Description |
|---|---|
| `--max-model-len` | Maximum context length (input + output tokens). Reduce to save memory. |
| `--gpu-memory-utilization` | Fraction of available memory vLLM may use (0.0–1.0). On CPU deployments, limits the fraction of system RAM used during initialization. Lower this if the pod fails to start with an OOM error. |
| `--disable-sliding-window` | Disable sliding window attention (required by some models). |
| `--enable-auto-tool-choice` | Enable automatic tool/function calling support. |
| `--tool-call-parser` | Specifies the parser used to interpret tool or function calls generated by the model. The parser must match the tool-calling format expected by the model. |

### Environment Variables

| Variable | Description |
|---|---|
| `VLLM_CPU_KVCACHE_SPACE` | Amount of system memory (GiB) reserved for the CPU KV cache. |

For a complete list of options, see the [vLLM documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server).
