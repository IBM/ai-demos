# Fraud Detection Classification with ONNX Runtime via Triton Inference Server

This guide demonstrates how to train and deploy a fraud detection classification model using ONNX Runtime served through Triton Inference Server. The example shows how to deploy a trained model and perform inference.

## Prerequisites

* Podman or Docker installed
* `curl` and `jq` command line tools
* Internet connection (required for downloading the dataset from Kaggle)

## Step 1: Navigate to the Working Directory

Before starting, navigate to the containerized demos directory:

```shell
cd ai-demos/containerized-demos
```

All subsequent commands assume you are running them from this directory.

## Step 2: Build the Training Environment Container

Build the container image used for training the ONNX model:

```shell
podman build -f shared/Containerfile -t localhost/build_env shared/
```

This creates a container image with all required dependencies for training the model.

## Step 3: Train and Generate the Model

Run the container to train the fraud detection model and generate the ONNX model file:

```shell
podman run --rm --name fraud-detection \
  -v $(pwd)/model-serving/predictive-models/triton-onnx/fraud-detection:/app:Z \
  --entrypoint="/bin/sh" \
  localhost/build_env \
  -c "cd /app && python3.12 build_fraud_detection_onnx_model.py"
```

**What this does**

* Downloads the credit card fraud dataset from Kaggle
* Trains a `RandomForestClassifier` model
* Converts the trained model to ONNX format
* Saves the model to
  `containerized-demos/model-serving/predictive-models/triton-onnx/fraud-detection/model-repository/fraud-detection/1/model.onnx`

## Step 4: Generate the Model Configuration

Generate the Triton model configuration file dynamically:

```shell
make generate-config APP=fraud-detection
```

**What this does**

* Starts a temporary Triton server
* Retrieves the auto generated configuration
* Converts it to Protocol Buffer format
* Saves the configuration file to
  `containerized-demos/model-serving/predictive-models/triton-onnx/fraud-detection/model-repository/fraud-detection/config.pbtxt`

## Step 5: Run the Triton Inference Server

Start the Triton Inference Server with the trained fraud detection model:

```shell
make run APP=fraud-detection
```

**What this does**

* Starts the Triton server in a container
* Mounts the model repository
* Exposes port

The server runs in detached mode and is accessible at:

```
http://localhost:8000
```

## Step 6: Test the Deployment

### 6.1 Check Server Health

Verify that the Triton server is running and the model is loaded:

```shell
curl -X POST http://0.0.0.0:8000/v2/repository/index
```

**Expected output**

```json
[{"name":"fraud-detection","version":"1","state":"READY"}]
```

### 6.2 Test with Fraudulent Transaction Data

Run inference using a sample fraudulent transaction:

```shell
curl -X POST http://0.0.0.0:8000/v2/models/fraud-detection/infer \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d @model-serving/predictive-models/triton-onnx/fraud-detection/data/sample-input-data/sample-fraud.json | jq
```

**Expected output**

```json
{
  "model_name": "fraud-detection",
  "model_version": "1",
  "outputs": [
    {
      "name": "label",
      "datatype": "INT64",
      "shape": [
        1,
        1
      ],
      "data": [
        1
      ]
    },
    {
      "name": "probabilities",
      "datatype": "FP32",
      "shape": [
        1,
        2
      ],
      "data": [
        -1.1920928955078126E-7,
        1.0000001192092896
      ]
    }
  ]
}
```

**Interpretation**

* `label: [1]` indicates a fraudulent transaction
* `probabilities: [non_fraud_probability, fraud_probability]` shows the predicted confidence for each class

### 6.3 Test with Non Fraudulent Transaction Data

Run inference using a legitimate transaction:

```shell
curl -X POST http://0.0.0.0:8000/v2/models/fraud-detection/infer \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d @model-serving/predictive-models/triton-onnx/fraud-detection/data/sample-input-data/sample-non-fraud.json | jq
```

## Step 7: Cleanup

Stop the Triton server and clean up generated resources:

```shell
# Stop the running container
podman stop $(podman ps -q --filter ancestor=quay.io/powercloud/tritonserver:latest)

# Optional: remove generated files
make clean APP=fraud-detection
