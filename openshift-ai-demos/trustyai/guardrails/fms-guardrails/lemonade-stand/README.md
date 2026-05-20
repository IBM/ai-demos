# TrustyAI Lemonade Stand Demo

In this example, we'll imagine we run a successful lemonade stand and want to deploy a customer service
agent so our customers can learn more about our products. We'll want to make sure all conversations with
the agent are family friendly, and that it does not promote our rival fruit juice vendors. 

---
### Prerequisites

Complete the setup steps in the [OpenShift AI README](/openshift-ai-demos/README.md#getting-started):
1. Configure DSCInitialization and DataScienceCluster
2. Create S3 secret with your storage credentials

### 1. Deploy Phi3 Model

Navigate to the OpenShift AI demos directory:
```bash
cd ai-demos/openshift-ai-demos
```

Create project:
```bash
oc new-project trustyai-demo || oc project trustyai-demo
```

Deploy the model (see [vLLM Model Serving guide](/openshift-ai-demos/model-serving/generative-models/vllm/README.md) for details):
```bash
oc process -n redhat-ods-applications vllm-cpu-runtime-template | oc create -f -
oc apply -f model-serving/generative-models/vllm/phi-3-mini-4k-instruct.yaml
```

Wait for the model pod to spin up, should look something like `phi3-predictor-XXXXXX`


Then, in a new terminal tab, we can send some prompts to the model via the included `prompt.py` helper script.
```bash
RAW_MODEL=https://$(oc get route phi3 -o jsonpath='{.spec.host}')
curl -ks -X POST "$RAW_MODEL/v1/chat/completions" -H 'accept: application/json' -H 'Content-Type: application/json'   -d '{
    "model": "phi3",
    "messages": [
      {
        "content": "The capital of France is",
        "role": "user"
      }
]}' | jq
````


---
### 2. Guardrails
In the following section, we'll walk through the configurations that have been provided inside of the `guardrails` folder. Everything is already written for you, so there's no need to add anything to the yaml files unless you're trying to experiment!


#### 2.1 Deploy the Hateful And Profane (HAP) language detector
This will use IBM's [granite-guardian-hap-125m](https://huggingface.co/ibm-granite/granite-guardian-hap-125m) model, which is a small model.

See the [Safety Models guide](/openshift-ai-demos/model-serving/safety-models/guardrails-detector-huggingface/README.md) for details.

```bash
oc process -n redhat-ods-applications guardrails-detector-huggingface-serving-template | oc create -f -
oc apply -f model-serving/safety-models/guardrails-detector-huggingface/granite-guardian-hap-125m.yaml
```
Wait for the `guardrails-detector-ibm-hap-predictor-xxx` pod to spin up


#### 2.2) Configure the Guardrails Orchestrator
Open `guardrails/configmap_orchestrator.yaml`. This is our configuration for the guardrails orchestrator:

```yaml
  config.yaml: |
    chat_generation:
      service:
        hostname: phi3-predictor.trustyai-demo.svc.cluster.local
        port: 8080
    detectors:
      regex_competitor:
        type: text_contents
        service:
            hostname: "localhost"
            port: 8080
        chunker_id: whole_doc_chunker
        default_threshold: 0.5
      hap:
        type: text_contents
        service:
          hostname: guardrails-detector-ibm-hap-predictor.trustyai-demo.svc.cluster.local
          port: 8000
        chunker_id: whole_doc_chunker
        default_threshold: 0.5
```

Here, we've defined the location of our `chat_generation` model server, and the locations of our detector servers. The `hap` detector is reachable via the service that is created by the KServe deployment (`phi3-predictor.trustyai-demo.svc.cluster.local`), while our regex detector sidecar will be launched at `localhost:8080`- this will always be the case when using the regex detector sidecar. 


#### 2.3) Configure our regex detector
Open `guardrails/configmap_vllm_gateway.yaml`. This is where we can configure our local detectors and our "preset" guardrailing pipeines. 

We have used the following regex pattern to filter out converstations about our rival juice vendors:
```regexp
\b(?i:apple|cranberry|grape|orange|pineapple|)\b
```
This will flag anything that matches that regex pattern as a detection- in this case, any mention of the words `apple`, `cranberry`, `grape`, `orange`, or `pineapple` regardless of case.


#### 2.4) Configure the Guardrails Gateway
Again, looking inside  `guardrails/configmap_vllm_gateway.yaml`:

The guardrails gateway provides two main features:
1) It provides the OpenAI `v1/chat/completions` API, which lets you hotswap between unguardrailed and guardrailed models
2) It lets you create guardrail "presets" baked into the endpoint.

#### Detectors 
First, we've set up the detectors that we want to use:
```yaml
 detectors:
  - name: regex_competitor
    input: true
    output: true
    detector_params:
      regex:    
        - \b(?i:orange|apple|cranberry|pineapple|grape)\b
  - name: hap
    input: true
    output: true
    detector_params: {}
```
Here, we've referred to the two detector names we created in the Orchestrator Configuration (`hap` and `regex_competitor`).
For both detectors, we specify `input: true` and `output: true`, meaning we will use them for both input and output
detectors. If you want to run a detector on only input or only output, you can change these flags. Finally, for the `regex_competitor`
detector, we define the specific regex that we described earlier.

#### Defining Presets
Next, we've created three preset pipelines for those detectors:
```yaml
routes:
  - name: all
    detectors:
      - regex_competitor
      - hap
  - name: hap
    detectors:
      - hap
  - name: passthrough
    detectors:
```
First is `all`, which will be served at `$GUARDRAILS_GATEWAY_URL/all/v1/chat/completions`, which will use both the `regex_competitor` and `hap` detectors. Next is `hap` will just uses the `hap` detector, and finally we have the `passthrough` preset, which does not use any detectors. 

### 2.5) Applying the ConfigMaps
Now that we've explored the configuration, let's deploy the configmap:
```bash
oc apply -f trustyai/guardrails/fms-guardrails/lemonade-stand/guardrails/configmap_vllm_gateway.yaml
```

Now, we can apply the final two configmaps needed to configure the orchestrator:
```bash
oc apply -f trustyai/guardrails/fms-guardrails/lemonade-stand/guardrails/configmap_orchestrator.yaml
```

### 2.6) Deploy the Orchestrator
```bash
oc apply -f trustyai/guardrails/fms-guardrails/lemonade-stand/guardrails/orchestrator_cr.yaml
```

### 2.7) Check the Orchestrator Health
```bash
ORCH_ROUTE_HEALTH=$(oc get routes guardrails-orchestrator-health -o jsonpath='{.spec.host}')
curl -sk https://$ORCH_ROUTE_HEALTH/info | jq
```
If everything is okay, it should return:

```json
{
  "services": {
    "openai": {
      "status": "HEALTHY"
    },
    "hap": {
      "status": "HEALTHY"
    },
    "regex_competitor": {
      "status": "HEALTHY"
    }
  }
}
```

---
### 3. Have a play around with Guardrails!
```bash
GUARDRAILS_GATEWAY=https://$(oc get routes guardrails-orchestrator-gateway -o jsonpath='{.spec.host}')
RAW_MODEL=https://$(oc get route phi3 -o jsonpath='{.spec.host}')
```

#### ❗NOTE: Increase the HTTP route's default timeout by adding following annotation to the gateway route : 
```
oc annotate route guardrails-orchestrator-gateway --overwrite haproxy.router.openshift.io/timeout=10m
```

The available endpoints are:

- `$GUARDRAILS_GATEWAY/passthrough`: query the raw, unguardrailed model. 
- `$GUARDRAILS_GATEWAY/hap`: query using with the HAP detector.
- `$GUARDRAILS_GATEWAY/all`: query with all available detectors, so the HAP and competitor-check detectors. 


Some cool queries to try:
### Query 1:"Is orange juice good?"
#### Raw Model:
```bash
curl -ks -X POST "$RAW_MODEL/v1/chat/completions" -H 'accept: application/json' -H 'Content-Type: application/json' -d '{
    "model": "phi3",
    "messages": [
      {
        "content": "Is orange juice good?",
        "role": "user"
      }
]}' | jq
```
Returns: 
```
{
  "id": "chatcmpl-96bae98e0b1b44fdb62c295edbed9a72",
  "object": "chat.completion",
  "created": 1758890489,
  "model": "phi3",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "reasoning_content": null,
        "content": " The question \"Is orange juice good?\" is very subjective and highly dependent on individual preferences and dietary needs. Many people find that orange juice is a refreshing and convenient source of vitamin C and other nutrients. However, it can also be high in natural sugars and calories, which may not be suitable for everyone, especially those with particular health concerns like diabetes. Balance and moderation are key with any food or drink, including orange juice, to ensure it fits within one's nutritional goals.",
        "tool_calls": []
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": 32007
    }
  ],
  "usage": {
    "prompt_tokens": 9,
    "total_tokens": 126,
    "completion_tokens": 117,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "kv_transfer_params": null
}
```

#### Guardrails, `/passthrough` endpoint:
```bash
curl -ks -X \
 POST "$GUARDRAILS_GATEWAY/passthrough/v1/chat/completions" -H 'accept: application/json' \
 -H 'Content-Type: application/json' \
 -d '{
    "model": "phi3",
    "messages": [
      {
        "content": "Is orange juice good?",
        "role": "user"
      }
]}' | jq
```
Returns: 
```
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "audio": null,
        "content": " Yes, orange juice can be considered good when discussing its nutritional benefits and consumed in moderation within a balanced diet. It is a good source of vitamin C, which is essential for a healthy immune system and skin health. It also contains other nutrients such as potassium, folate, and vitamin B vitamins. However, it's important to choose pure orange juice rather than those with added sugars or preservatives.",
        "refusal": null,
        "role": "assistant",
        "tool_calls": null
      }
    }
  ],
  "created": 1758890583,
  "detections": null,
  "id": "chatcmpl-2335cf9ae2424b1a8031aa5c28b89000",
  "model": "phi3",
  "object": "chat.completion",
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "completion_tokens": 103,
    "prompt_tokens": 9,
    "total_tokens": 112
  },
  "warnings": null
}
```

#### Guardrails, `/all` endpoint:
```bash
curl -ks -X \
 POST "$GUARDRAILS_GATEWAY/all/v1/chat/completions" -H 'accept: application/json' \
 -H 'Content-Type: application/json' \
 -d '{
    "model": "phi3",
    "messages": [
      {
        "content": "Is orange juice good?",
        "role": "user"
      }
]}' | jq
```
Returns: 
```
{
  "choices": [],
  "created": 1758952237,
  "detections": {
    "input": [
      {
        "message_index": 0,
        "results": [
          {
            "detection": "custom-regex",
            "detection_type": "regex",
            "detector_id": "regex_competitor",
            "end": 9,
            "score": 1,
            "start": 3,
            "text": "orange"
          }
        ]
      }
    ],
    "output": null
  },
  "id": "05604a0810a04c6d916dab5d9d05a996",
  "model": "phi3",
  "object": "",
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "completion_tokens": 0,
    "prompt_tokens": 6,
    "total_tokens": 0
  },
  "warnings": [
    {
      "message": "Unsuitable input detected. Please check the detected entities on your input and try again with the unsuitable input removed.",
      "type": "UNSUITABLE_INPUT"
    }
  ]
}
```

### Query 2: "I don't want your damn lemons! What the Hell am I supposed to do with these?"
#### Raw Model:
```bash
curl -ks -X \
 POST "$RAW_MODEL/v1/chat/completions" -H 'accept: application/json' \
 -H 'Content-Type: application/json' \
 -d '{
    "model": "phi3",
    "messages": [
      {
        "content": "I dont want your damn lemons! What the Hell am I supposed to do with these?",
        "role": "user"
      }
]}' | jq
```
Returns: 
```
{
  "id": "chatcmpl-ad898e4904e74099a3f6bd3c277b30c4",
  "object": "chat.completion",
  "created": 1758952323,
  "model": "phi3",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "reasoning_content": null,
        "content": " I apologize if my previous message came across as suggesting lemons, which they were not. It was purely a lighthearted attempt to shift an argument off discussing things you might dislike. However, if you're upset or having a bad day, I'm here to help. What can I assist you with today that might resonate more positively?",
        "tool_calls": []
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": 32007
    }
  ],
  "usage": {
    "prompt_tokens": 23,
    "total_tokens": 100,
    "completion_tokens": 77,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "kv_transfer_params": null
}
```

#### Guardrails, `/all` endpoint:
```bash
curl -ks -X \
 POST "$GUARDRAILS_GATEWAY/all/v1/chat/completions" -H 'accept: application/json' \
 -H 'Content-Type: application/json' \
 -d '{
    "model": "phi3",
    "messages": [
      {
        "content": "I dont want your damn lemons! What the Hell am I supposed to do with these?",
        "role": "user"
      }
]}' | jq
```
Returns: 
```
{
  "choices": [],
  "created": 1758952390,
  "detections": {
    "input": [
      {
        "message_index": 0,
        "results": [
          {
            "detection": "single_label_classification",
            "detection_type": "LABEL_1",
            "detector_id": "hap",
            "end": 75,
            "score": 0.9924468994140624,
            "start": 0,
            "text": "I dont want your damn lemons! What the Hell am I supposed to do with these?"
          }
        ]
      }
    ],
    "output": null
  },
  "id": "0fec9c8fb71e4d168f7589742d044290",
  "model": "phi3",
  "object": "",
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "completion_tokens": 0,
    "prompt_tokens": 20,
    "total_tokens": 0
  },
  "warnings": [
    {
      "message": "Unsuitable input detected. Please check the detected entities on your input and try again with the unsuitable input removed.",
      "type": "UNSUITABLE_INPUT"
    }
  ]
}
```

### Query 3: "Can you list some healthy fruit juices?"
#### Raw Model:
```bash
curl -ks -X \
 POST "$RAW_MODEL/v1/chat/completions" -H 'accept: application/json' \
 -H 'Content-Type: application/json' \
 -d '{
    "model": "phi3",
    "messages": [
      {
        "content": "Can you list some healthy fruit juices?",
        "role": "user"
      }
]}' | jq
```
Returns: 
```
{
  "id": "chatcmpl-787832f8da8b4f098ed55e7c06d81cb4",
  "object": "chat.completion",
  "created": 1758952595,
  "model": "phi3",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "reasoning_content": null,
        "content": " Certainly! Healthy fruit juices typically contain natural sugars, vitamins, minerals, and antioxidants. Here are some options that are considered healthy when consumed in moderation:\n\n\n1. **Apple Juice** - Apple juice can provide vitamin C and potassium, but since it's from whole apples, it retains most of the fiber. Opt for 100% fruit juice without added sugars.\n\n\n2. **Cranberry Juice** - Rich in antioxidants and vitamins, cranberry juice can be good for urinary tract health and has anti-inflammatory properties. Just look for natural varieties without added sugars.\n\n\n3. **Watermelon Juice** - A hydrating option packed with vitamins (C and B5) and has mild diuretic properties. Full of water, it's beneficial for summer hydration.\n\n\n4. **Orange Juice** - A strong source of vitamin C and folate; especially with the peel, since it contains fiber and is less processed. Unsweetened or low-sugar varieties are best.\n\n\n5. **Pineapple Juice** - Loaded with bromelain, an enzyme that is beneficial for digestion. It's also a source of vitamin C and manganese.\n\n\n6. **Grape Juice** - Offers vitamin C and numerous phytonutrients. Grape juice can also aid digestion when consumed with meals due to its high acidity.\n\n\nRemember, while fruit juices can be part of a healthy diet when chosen wisely and consumed in moderation, whole fruits are generally preferable due to their fiber content which aids in digestion and satiety.",
        "tool_calls": []
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": 32007
    }
  ],
  "usage": {
    "prompt_tokens": 13,
    "total_tokens": 439,
    "completion_tokens": 426,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "kv_transfer_params": null
}
```

#### Guardrails, `/all` endpoint:
```bash
curl -ks -X \
 POST "$GUARDRAILS_GATEWAY/all/v1/chat/completions" -H 'accept: application/json' \
 -H 'Content-Type: application/json' \
 -d '{
    "model": "phi3",
    "messages": [
      {
        "content": "Can you list some healthy fruit juices?",
        "role": "user"
      }
]}' | jq
```
Returns: 
```
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "audio": null,
        "content": " Certainly! Fruit juices can be a part of a healthy diet when consumed in moderation due to their vitamins, minerals, and antioxidants. Here are some fruit juices that are generally considered healthy:\n\n\n1. **Apple Juice** - Pure apple juice made from whole apples without added sugars is a good source of vitamin C and antioxidants.\n\n2. **Orange Juice** - Whole orange juice contains high levels of vitamin C, potassium, calcium, and folate.\n\n3. **Pineapple Juice** - Rich in bromelain, an enzyme that helps digestion, and vitamin C for immune health.\n\n4. **Cranberry Juice** - A good choice for urinary tract health due to its high levels of proanthocyanidins.\n\n5. **Berry Juices (Strawberries, Blueberries, Raspberries, Blackberries)** - Full of antioxidants and vitamins; berry juices are often less sweet than apples and oranges.\n\n6. **Pomegranate Juice** - High in polyphenols, which are compounds with anti-inflammatory and antioxidant properties.\n\n\nNote that although these juices can be part of a healthy diet, juices should not be consumed in place of whole fruits, as whole fruits provide dietary fiber which is lacking in most fruit juices. Additionally, it's advisable to limit juice intake and pay attention to the product label to ensure it's not a refined product.",
        "refusal": null,
        "role": "assistant",
        "tool_calls": null
      }
    }
  ],
  "created": 1758952861,
  "detections": {
    "input": null,
    "output": [
      {
        "choice_index": 0,
        "results": [
          {
            "detection": "custom-regex",
            "detection_type": "regex",
            "detector_id": "regex_competitor",
            "end": 214,
            "score": 1,
            "start": 209,
            "text": "Apple"
          },
          {
            "detection": "custom-regex",
            "detection_type": "regex",
            "detector_id": "regex_competitor",
            "end": 235,
            "score": 1,
            "start": 230,
            "text": "apple"
          },
          {
            "detection": "custom-regex",
            "detection_type": "regex",
            "detector_id": "regex_competitor",
            "end": 346,
            "score": 1,
            "start": 340,
            "text": "Orange"
          },
          {
            "detection": "custom-regex",
            "detection_type": "regex",
            "detector_id": "regex_competitor",
            "end": 369,
            "score": 1,
            "start": 363,
            "text": "orange"
          },
          {
            "detection": "custom-regex",
            "detection_type": "regex",
            "detector_id": "regex_competitor",
            "end": 458,
            "score": 1,
            "start": 449,
            "text": "Pineapple"
          },
          {
            "detection": "custom-regex",
            "detection_type": "regex",
            "detector_id": "regex_competitor",
            "end": 568,
            "score": 1,
            "start": 559,
            "text": "Cranberry"
          }
        ]
      }
    ]
  },
  "id": "chatcmpl-25f5a85666534bc098b548715974223e",
  "model": "phi3",
  "object": "chat.completion",
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "completion_tokens": 382,
    "prompt_tokens": 13,
    "total_tokens": 395
  },
  "warnings": [
    {
      "message": "Unsuitable output detected.",
      "type": "UNSUITABLE_OUTPUT"
    }
  ]
}
```

### More info:
- [Redhat Doc](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.23/html/monitoring_data_science_models/configuring-the-guardrails-orchestrator-service_monitor)
- [Github fms-guardrails-orchestrator](https://github.com/trustyai-explainability/fms-guardrails-orchestrator)
