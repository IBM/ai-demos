# Guardrails Detector HuggingFace Runtime

Deploy safety detection models using the Guardrails Detector HuggingFace runtime.

---

## Prerequisites

Create a data science project:
```bash
oc new-project <your-project> || oc project <your-project>
```

Navigate to the OpenShift AI demos directory:
```bash
cd ai-demos/openshift-ai-demos
```

Create the guardrails detector runtime:
```bash
oc process -n redhat-ods-applications guardrails-detector-huggingface-serving-template | oc create -f -
```

---

## Available Models

### Granite Guardian HAP 125M

**Model:** IBM's [granite-guardian-hap-125m](https://huggingface.co/ibm-granite/granite-guardian-hap-125m)

**Purpose:** Detects Hate speech, Abusive language, and Profanity (HAP) in text

**Deploy:**
```bash
oc apply -f model-serving/safety-models/guardrails-detector-huggingface/granite-guardian-hap-125m.yaml -n <your-project>
```

**Minimum Resource Requirements:**
| Resource | Allocation |
| -------- | ---------- |
| Runtime  | Guardrails Detector HuggingFace |
| CPU      | 4 to 8 cores |
| Memory   | 8Gi to 10Gi |

> These are the minimum resources specified in the InferenceService YAML. Actual usage may vary based on workload.

---

## Verify Deployment

> **Deployment Time:** Model deployment typically takes 3-5 minutes depending on model size and cluster resources.

```bash
# List InferenceServices
oc get inferenceservice

# Check status
oc describe inferenceservice guardrails-detector-ibm-hap

# Watch predictor pod status
oc get pods -w | grep guardrails-detector-ibm-hap-predictor
```

---

## Test the Model

```bash
# Get model URL
MODEL_URL=$(oc get inferenceservice guardrails-detector-ibm-hap -o jsonpath='{.status.url}')
```
#### Example 1: Request containing only safe content

The following request contains only safe text inputs. Since no violations are detected, the API returns an empty list for each input.
```
curl -k -X POST "$MODEL_URL/api/v1/text/contents" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      "This is the 1st test",
      "This is the 2nd test"
    ],
    "detector_params": {}
  }'
```
Output:
```
[[],[]]
```

#### Example 2: Request containing both flagged and safe content

The following request contains one text input that is flagged by the detector and one safe input. The response includes the detection details for the flagged text, while the safe text returns an empty list.
```
# curl -k -X POST "$MODEL_URL/api/v1/text/contents"   -H "Content-Type: application/json"   -d '{
    "contents": [
      "I hate you and I want to destroy everything",
      "This is a perfectly safe sentence."
    ],
    "detector_params": {}
  }'
```

Output:
```
[
  [
    {
      "start": 0,
      "end": 43,
      "text": "I hate you and I want to destroy everything",
      "detection": "single_label_classification",
      "detection_type": "LABEL_1",
      "score": 0.9328954219818115,
      "evidences": [],
      "metadata": {}
    }
  ],
  []
]
```
In this example, the first input is identified as LABEL_1 with a confidence score of 0.93, while the second input does not trigger any detections.

---

## Use Cases

- **Content Moderation** - Filter inappropriate content in user-generated text
- **Chat Safety** - Detect harmful language in chatbot conversations
- **Comment Filtering** - Screen comments for hate speech and profanity
- **Guardrails** - Add safety layers to LLM applications

---

## Resources

- [Granite Guardian HAP Model](https://huggingface.co/ibm-granite/granite-guardian-hap-125m)
