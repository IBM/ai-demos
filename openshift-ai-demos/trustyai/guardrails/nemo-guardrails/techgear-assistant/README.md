# NeMo Guardrails Demo - TechGear Electronics Store Assistant

A guardrail is a semi or fully deterministic control that restricts or guides LLM behavior. Guardrails can block specific topics, steer conversations, or trigger actions such as escalating to a human.

NeMo Guardrails is an open source toolkit for managing LLM behavior at runtime. It uses a rule-based language called Colang to define conversation flows, restrict topics, and enforce policies. The framework is LLM agnostic and designed for low latency deployments, including GPU-accelerated environments.

## Demo Overview

This demo showcases a simple **e-commerce scenario** for **TechGear Electronics**, and shows how NeMo Guardrails can be used to keep interactions safe, relevant, and on-brand.  

This demo uses an **instructions-based** approach, where product information is embedded directly in the bot's instructions. The LLM generates natural responses from this context, without relying on predefined flows, while guardrails ensure interactions remain safe and within scope.

The configuration utilizes a `Qwen2.5-1.5B-Instruct` model deployed via KServe, integrated with input and output guardrails to ensure safe, accurate, and on-brand customer interactions.

## How It Works

```
User Input
    ↓
Input Guardrails (PII, length, content, competitor, scope)
    ↓
Flow Matching (greeting → predefined | other → general)
    ↓
LLM Generation (uses instructions + sample conversations)
    ↓
Output Guardrails (PII masking)
    ↓
Response
```

---
### Prerequisites

Complete the setup steps in the [OpenShift AI README](/openshift-ai-demos/README.md#getting-started):
1. Configure DSCInitialization and DataScienceCluster
2. Create S3 secret with your storage credentials

### 1. Model Configuration

We will be deploying the `Qwen2.5-1.5B-Instruct` model and use it as self-hosted vLLM backend (with an OpenAI-compatible API) for NeMo Guardrails.

#### Deploy the Model

Navigate to the OpenShift AI demos directory:
```bash
cd ai-demos/openshift-ai-demos
```

Create and switch to the demo project:
```bash
oc new-project trustyai-demo || oc project trustyai-demo
```

Deploy the vLLM runtime and model (see [vLLM Model Serving guide](/openshift-ai-demos/model-serving/generative-models/vllm/README.md) for details):
```bash
oc process -n redhat-ods-applications vllm-cpu-runtime-template | oc create -f -
oc apply -f model-serving/generative-models/vllm/qwen2.5-1.5b-instruct.yaml
```

Wait for the model pod to be ready, for example `qwen-predictor-XXXXXX`. It may take a approximately 5-8 minutes to be ready.

Test the raw model endpoint:
```bash
RAW_MODEL=https://$(oc get route qwen -o jsonpath='{.spec.host}')
curl -ks -X POST "$RAW_MODEL/v1/completions" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen",
    "prompt": "The capital of France is",
    "max_tokens": 20
  }' | jq
```

### 2. Configure NeMo Guardrails

#### 2.1 Create a OPENAI token secret:
> **Note:** For demo purposes, you can use any string value as the token (e.g., "demo-token").
```bash
oc create secret generic openai-key-secret --from-literal=token=<YOUR_API_TOKEN>
```

#### 2.2 Create the Guardrails Configuration

The guardrails configuration is organized into separate files:

**Configuration Structure:**
```
configurations/
├── config.yaml    # Instructions, model config, and rails settings
├── rails.co       # Colang conversation flows and guardrail logic
└── actions.py     # Custom Python validation functions
```

#### 2.2.1 config.yaml
This includes `instructions`, `sample conversations`, `model configuration`, and `rails config`:

**Instructions (Bot Context with Product Catalog)**

```yaml
instructions:
  - type: general
    content: |
      TechGear Electronics Assistant
      
      PRODUCTS:
      1. TechGear Pro 15 Laptop - $1,299 (i7, 16GB RAM, RTX 4060)
      2. TechGear Phone X1 - $899 (Snapdragon 8 Gen 2, 120Hz AMOLED)
      
      POLICIES: Free shipping >$50, 30-day returns, 1-3yr warranty
```

> **Note:** Product information is embedded in instructions (no RAG/knowledge base). Suitable for small catalogs (2-10 products).

**Sample Conversations**

Guide the LLM's response style with examples:
```yaml
sample_conversation: |
  user "What products do you have?"
    ask general question
  bot provide response
    "We offer two excellent products: TechGear Pro 15 laptop for $1,299..."
```

**Model Configuration**

```yaml
models:
  - type: main
    engine: vllm_openai
    parameters:
      openai_api_base: "http://qwen-predictor.trustyai-demo.svc.cluster.local:8080/v1"
      model_name: "qwen"
```

Where,
```
- type: main            # Identifies the primary LLM used for conversations
- engine: vllm_openai   # Uses a self-hosted vLLM backend with an OpenAI-compatible API
- parameters            # Provider-specific connection details, including the model endpoint and name
```

For more information please refer [#models-configuration](https://docs.nvidia.com/nemo/guardrails/latest/configure-rails/configuration-reference.html#models-configuration)

**Rails Config:**  
Built-in Microsoft Presidio integration for PII detection:
```yaml
  config:
    sensitive_data_detection:
      input:
        entities:
          - PERSON
          - EMAIL_ADDRESS
          - PHONE_NUMBER
          - CREDIT_CARD
      output:
        entities:
          - PERSON
          - EMAIL_ADDRESS
          - PHONE_NUMBER
          - CREDIT_CARD
```

**Input Rails**  
Run before the LLM is invoked to validate, filter, and modify user input
```yaml
  input:
    flows:
      - detect sensitive data on input
      - check message length
      - check forbidden content
      - check competitor mentions
      - check out of scope requests
```
**Input Guardrails:**  
- **PII Detection**: Blocks sensitive personal information
- **Message Length**: Enforces 100-word limit for better responses
- **Content Filtering**: Blocks profanity, violence, illegal activities
- **Competitor Blocking**: Redirects mentions of Apple, Samsung, Dell, etc.
- **Scope Validation**: Blocks medical, legal, political, financial advice

**Output Rails**  
Run after the LLM responds to validate and filter bot responses
```yaml
  output:
    flows:
      - detect sensitive data on output
```
**Output Guardrails:**
- **PII Masking**: Prevents leaking personal information in responses

#### 2.2.2 rails.co (Colang Flows):
A Colang script is a `.co` file composed of flow definitions. Each flow describes the desired interaction between the user and the bot.

**1. Greeting Flow** (Predefined Responses)
```colang
define flow greeting
  user express greeting
  bot express greeting
  bot introduce store
```

**2. General Conversation Flow** (LLM-Generated)
```colang
define flow general conversation
  user ask general question  # "..." wildcard matches any input
  bot provide response       # Undefined → triggers LLM
```

**3. Guardrail Flows** (Deterministic Checks)
- `check competitor mentions` - Blocks competitor brands
- `check out of scope requests` - Blocks off-topic keywords
- `check message length` - 100-word limit
- `check forbidden content` - Profanity/violence filter
- `detect sensitive data` - PII detection (built-in)


See the complete Colang flows in [configurations/rails.co](configurations/rails.co)

Reference: [NeMo Guardrails Colang Documentation](https://docs.nvidia.com/nemo/guardrails/latest/configure-rails/colang/colang-2/getting-started/index.html)

#### 2.2.3 actions.py (Custom Actions):

Custom Python actions implement deterministic checks and business logic:

**Custom Actions** (Deterministic Python Functions):
- `check_message_length()` - 100-word limit
- `check_forbidden_content()` - Profanity/violence filter
- `check_competitor_mentions()` - Blocks: Apple, Samsung, Dell, HP, Lenovo, Asus, Acer, Microsoft Surface, Google Pixel, OnePlus, Xiaomi, MacBook, iPhone, Galaxy
- `check_out_of_scope()` - Blocks: medical, health, legal, political, financial topics


**Create the ConfigMap from the configuration directory:**

```bash
oc create configmap nemo-config --from-file=trustyai/guardrails/nemo-guardrails/techgear-assistant/configurations/
```

This automatically creates a ConfigMap with keys for each file in the directory (`config.yaml`, `rails.co`, `actions.py`).

#### 2.3 Create the NeMo Guardrails Custom Resource

```bash
oc create -f - <<EOF
apiVersion: trustyai.opendatahub.io/v1alpha1
kind: NemoGuardrails
metadata:
  name: nemo-guardrails-cr
  annotations:
    security.opendatahub.io/enable-auth: 'false'  # Disabled for demo purposes only
spec:
  nemoConfigs:
    - name: config
      default: true
      configMaps:
        - nemo-config
  env:
    - name: "OPENAI_API_KEY"
      valueFrom:
        secretKeyRef:
          name: "openai-key-secret"
          key: "token"
EOF
```

The operator mounts the ConfigMap to `/app/config/config/`
Final structure:
```
  /app/config/config/
  ├── config.yaml
  ├── rails.co
  └── actions.py
```

#### ❗NOTE: Increase route timeout: 
OpenShift routes have a default timeout of 30 seconds.
NeMo Guardrails may exceed this limit due to multiple guardrail checks and LLM inference latency.  
Tune the timeout according to your model size and expected response times.
```bash
oc annotate route nemo-guardrails-cr --overwrite haproxy.router.openshift.io/timeout=10m
```

### 3. Testing the TechGear Electronics Assistant

Set the Guardrails route:
```bash
GUARDRAILS_ROUTE=https://$(oc get routes/nemo-guardrails-cr -o jsonpath='{.status.ingress[0].host}')
```

---

## Example Interactions

### 1. Product Inquiry 

**Request:**
```bash
curl -k -X POST \
  $GUARDRAILS_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(oc whoami -t)" \
  -d '{
    "model": "qwen",
    "messages": [
      {
        "role": "user",
        "content": "What products do you have?"
      }
    ]
  }' | jq
```

**Response:**
```json
{
  "message": {
    "content": "We offer two products: TechGear Pro 15 laptop ($1,299) and TechGear Phone X1 ($899). Which one interests you?",
    "role": "assistant"
  }
}
```

### 2. Competitor Blocking (Guardrail Test)

**Request:**
```bash
curl -k -X POST \
  $GUARDRAILS_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(oc whoami -t)" \
  -d '{
    "model": "qwen",
    "messages": [
      {
        "role": "user",
        "content": "How does this compare to Apple MacBook?"
      }
    ]
  }' | jq
```

**Response:**
```json
{
  "message": {
    "content": "I appreciate your interest, but I can only provide information about TechGear products. However, I'd be happy to help you find a comparable TechGear product that meets your needs!",
    "role": "assistant"
  }
}
```

### 3. PII Detection (Guardrail Test)

**Request:**
```bash
curl -k -X POST \
  $GUARDRAILS_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(oc whoami -t)" \
  -d '{
    "model": "qwen",
    "messages": [
      {
        "role": "user",
        "content": "Send the invoice to john.doe@example.com"
      }
    ]
  }' | jq
```

**Response:**
```json
{
  "message": {
    "content": "I don't know the answer to that.",
    "role": "assistant"
  }
}
```

### 4. Out-of-Scope Request (Guardrail Test)

**Request:**
```bash
curl -k -X POST \
  $GUARDRAILS_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(oc whoami -t)" \
  -d '{
    "model": "qwen",
    "messages": [
      {
        "role": "user",
        "content": "What do you think about the president?"
      }
    ]
  }' | jq
```

**Response:**
```json
{
  "message": {
    "content": "I'm a TechGear Electronics assistant and can only help with product inquiries and tech support. Is there anything tech-related I can help you with?",
    "role": "assistant"
  }
}
```

### 5. Message Length Validation (Guardrail Test)

**Request:**
```bash
LONG_QUERY="I am looking for a laptop that has excellent performance for video editing and 3D rendering work with at least 32GB of RAM and a dedicated graphics card preferably NVIDIA RTX series and a large high resolution display preferably 4K and fast SSD storage of at least 1TB and good battery life for portability and lightweight design under 5 pounds and excellent build quality with aluminum chassis and good keyboard for long typing sessions and multiple USB-C ports for connectivity and Thunderbolt support and WiFi 6E and Bluetooth 5.2 and a reasonable price under three thousand dollars if possible and also need it to have good cooling system because I will be using it for extended periods and warranty coverage is important too preferably at least 2 years with on-site support"

curl -k -X POST \
  "$GUARDRAILS_ROUTE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(oc whoami -t)" \
  -d "{
    \"model\": \"qwen\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": \"${LONG_QUERY}\"
      }
    ]
  }" | jq
```

**Response:**
```json
{
  "message": {
    "content": "Your message is quite long. Could you please break it down into smaller questions?",
    "role": "assistant"
  }
}
```

### 6. Customer Support Flow

**Request:**
```bash
curl -k -X POST \
  $GUARDRAILS_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(oc whoami -t)" \
  -d '{
    "model": "qwen",
    "messages": [
      {
        "role": "user",
        "content": "What is your return policy?"
      }
    ]
  }' | jq
```

**Response:**
```json
{
  "message": {
    "content": "We have a 30-day return window from delivery date. Items must be in original condition. Refunds are processed within 5-7 business days. Customers pay return shipping unless the item is defective.",
    "role": "assistant"
  }
}
```

### 6. Product Inquiry Flow

**Request:**
```bash
curl -k -X POST \                               
  $GUARDRAILS_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(oc whoami -t)" \
  -d '{
    "model": "qwen",
    "messages": [
      {
        "role": "user",
        "content": "What is the RAM of TechGear Pro 15 Laptop?"
      }
    ]
  }' | jq
```

**Response:**
```json
{
  "message": {
    "content": "The TechGear Pro 15 has 16GB DDR5 RAM.",
    "role": "assistant"
  }
}
```

#### References:
- [Deploying NeMo Guardrails](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.2/html/enabling_ai_safety_with_guardrails/deploying-nemo-guardrails_nemo-guardrails)
- https://docs.nvidia.com/nemo/guardrails/latest/configure-rails/configuration-reference.html
- https://www.pinecone.io/learn/nemo-guardrails-intro/
