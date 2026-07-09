# Red Hat OpenShift AI Demos
Demonstrations for Red Hat OpenShift AI (RHOAI) showcasing model serving, LLM evaluation, and guardrails.


## Getting Started

### 1. Prerequisites
- OpenShift Cluster
- Cluster admin access
- Red Hat OpenShift AI operator installed

### 2. Clone and Configure OpenShift AI
```bash
# Clone the repository
git clone https://github.com/IBM/ai-demos.git
cd ai-demos/openshift-ai-demos/

# Create DSCInitialization
oc create -f shared/dsci.yaml

# Verify DSCInitialization is in Ready state
oc get dsci
NAME           AGE   PHASE   CREATED AT
default-dsci   25h   Ready   2025-09-23T04:33:53Z

# Deploy DataScienceCluster
oc create -f shared/dsc.yaml

# Verify DataScienceCluster is in Ready state
oc get dsc
NAME          READY   REASON
default-dsc   True   
```

### 3. S3 Storage *(required based on use case)*

> **Note:** S3 storage is required for demos that use S3-based model serving or store data in object storage (e.g. AutoRAG, document ingestion pipelines). If you are only serving models downloaded directly into a PVC, you can skip this step. Refer to [model storage options](model-serving/README.md#model-storage-options) for details.

Update [`s3-secret.yaml`](shared/s3-secret.yaml) with your credentials:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `AWS_S3_BUCKET`
- `AWS_S3_ENDPOINT`

Apply the secret to your target data science project namespace:
```bash
oc new-project <your-project> || oc project <your-project>
oc apply -f shared/s3-secret.yaml
```

## Available Demos

### Model Serving
Deploy and serve AI models using KServe on Red Hat OpenShift AI.

[→ Explore Model Serving](model-serving/)

---

### TrustyAI
Explore Responsible AI workflows, LLM safety evaluation, and real-time guardrails.

[→ Explore TrustyAI](trustyai/)

---
## Shared Resources
[`shared/`](shared/) directory contains reusable cluster configuration and setup resources:
- [`dsci.yaml`](shared/dsci.yaml) - DSCInitialization
- [`dsc.yaml`](shared/dsc.yaml) - DataScienceCluster
- [`s3-secret.yaml`](shared/s3-secret.yaml) - S3 credentials template


## Resources
- [Red Hat OpenShift AI Docs](https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/)
