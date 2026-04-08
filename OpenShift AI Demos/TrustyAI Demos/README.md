# TrustyAI Demos 
This repository also includes TrustyAI demonstration examples, such as LLM evaluation, guardrail, bias monitoring and data drift detection.

### Prerequisites
#### 1. Enabling the TrustyAI component for RawDeployment  
Configure `DSCInitialization` and `DataScienceCluster` resources for RawDeployment
```
# Create DSCInitialization
oc create -f common/setup/dsci.yaml

# Verfiy DSCInitialization is in Ready state
oc get dsci
NAME           AGE   PHASE   CREATED AT
default-dsci   25h   Ready   2025-09-23T04:33:53Z

# Deploy DataScienceCluster
oc create -f common/setup/dsc.yaml

# Verfiy DataScienceCluster is in Ready state
oc get dsc
NAME          READY   REASON
default-dsc   True    
```

#### 2. Create a S3 secret  
In this demo, we will be using all the models from IBM Cloud S3 bucket.

Before applying the secret, update [s3-secret](common/setup/s3-secret.yaml) with your S3 credentials and endpoint details, including AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, AWS_S3_BUCKET, and AWS_S3_ENDPOINT.

```
oc new-project trustyai-demo || oc project trustyai-demo
oc apply -f common/setup/s3-secret.yaml
```

## Large Language Models Demos
#### 1. Evaluation Demo  
This demo will quickly get you started running an evaluation against an InferenceService which is already deployed and running in your namespace.  
Demo - [Evaluating large language models](eval-quickstart-demo/)

#### 2. Guardrail Demo  
**FMS Guardrail Demo:**  
A demo showing manual configuration of guardrails  
Demo - [Lemonade Stand Demo](guardails-lemonade-stand-demo/)

**NeMo Guardrail Demo:**  
Demonstrates conversational guardrails using NVIDIA NeMo Guardrails, including topic blocking, input validation, and sensitive data detection.  
Demo - [NeMo-Guardrails Demo](nemo-guardrails-demo/)

## Machine Learning Models Demos
#### 1. Data Drift Demo  
How to detect if the production data your models are receiving matches the data they were trained on.  
Demo - [Data Drift Demo](trustyai-svc-demos/data-drift/)

#### 2. Bias Monitoring Demo [Coming Soon]
How to use TrustyAI to examine your deployed models for unfair biases.  

#### 3. Anomaly Detection [Coming Soon] 
How to identify and log anomalous inbound data, such as to clean or enrich your training data.  

#### 4. Explainability Demo [Coming Soon]
How to get per-point explanations of your models' predictions.  

## More information
- [TrustyAI Notes Repo](https://github.com/trustyai-explainability/reference/tree/main)
- [TrustyAI Github](https://github.com/trustyai-explainability)
