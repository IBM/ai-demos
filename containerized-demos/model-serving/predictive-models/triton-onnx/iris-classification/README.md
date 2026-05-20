# Iris Classification with ONNX Runtime via Triton Inference Server

This guide demonstrates how to train an Iris classification model using the classic Iris dataset, deploy it with Triton Inference Server using the ONNX Runtime backend and perform inference.

## Prerequisites

* Podman or Docker installed
* `curl` and `jq` command line tools
* Internet connection (for downloading dependencies)

## Step 1: Navigate to the Working Directory

Before starting, navigate to the containerized demos directory:

```shell
cd ai-demos/containerized-demos
```

## Step 2: Build the Training Environment Container

Build the container image used for training the ONNX model:

```shell
podman build -f shared/Containerfile -t localhost/build_env shared/
```

This creates a container image with all required dependencies for model training.

## Step 3: Train and Generate the Model

Run the container to train the Iris classification model and generate the ONNX model file:

```shell
podman run --rm --name iris-classification \
  -v $(pwd)/model-serving/predictive-models/triton-onnx/iris-classification:/app:Z \
  --entrypoint="/bin/sh" \
  localhost/build_env \
  -c "cd /app && python3.12 build_iris_onnx_model.py"
```

**What this does:**

* Loads the Iris dataset from scikit-learn
* Trains a `RandomForestClassifier` model
* Converts the trained model to ONNX format
* Saves the model to
  `containerized-demos/model-serving/predictive-models/triton-onnx/iris-classification/model-repository/iris-classification/1/model.onnx`

## Step 4: Generate the Model Configuration

Generate the Triton model configuration file dynamically:

```shell
make generate-config APP=iris-classification
```

**What this does:**

* Starts a temporary Triton server
* Retrieves the auto generated model configuration
* Converts it to Protocol Buffer format
* Saves the configuration file to
  `containerized-demos/model-serving/predictive-models/triton-onnx/iris-classification/model-repository/iris-classification/config.pbtxt`

## Step 5: Run the Triton Inference Server

Start the Triton Inference Server with the trained Iris classification model:

```shell
make run APP=iris-classification
```

**What this does:**

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

**Expected output:**

```json
[{"name":"iris-classification","version":"1","state":"READY"}]
```

### 6.2 Test with Sample Data

Run inference using sample Iris input data:

```shell
curl -X POST http://0.0.0.0:8000/v2/models/iris-classification/infer \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d @model-serving/predictive-models/triton-onnx/iris-classification/data/sample-input-data/data.json | jq
```

**Expected output:**

```json
{
  "model_name": "iris-classification",
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
        0
      ]
    },
    {
      "name": "probabilities",
      "datatype": "FP32",
      "shape": [
        1,
        3
      ],
      "data": [
        1.0000001192092896,
        0.0,
        0.0
      ]
    }
  ]
}
```

**Interpretation**

* `label: [0]` indicates **Iris Setosa** (class 0)
* `probabilities: [1.0, 0.0, 0.0]` shows 100 percent confidence for Setosa and 0 percent for Versicolor and Virginica

## Step 7: Cleanup

Stop the Triton server and clean up generated resources:

```shell
# Stop the running container
podman stop $(podman ps -q --filter ancestor=quay.io/powercloud/tritonserver:latest)

# Optional: remove generated files
make clean APP=iris-classification
