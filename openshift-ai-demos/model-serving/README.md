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

Update the `storageUri` in each model's YAML file to match your S3 bucket structure.

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
