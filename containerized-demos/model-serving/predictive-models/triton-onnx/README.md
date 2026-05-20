# Triton Inference Server with ONNX Runtime

Triton Inference Server with ONNX Runtime is a deployment solution that enables efficient, scalable, and high performance serving of machine learning models in the ONNX format.

---

## Available Demos

### Iris Classification
Multi-class classification using the classic Iris dataset.

[→ View Iris Classification Guide](iris-classification/)

---

### Fraud Detection
Binary classification for credit card fraud detection.

[→ View Fraud Detection Guide](fraud-detection/)

---
## Shared Utilities

The `shared/` directory contains utilities.

The following scripts are used to generate model configuration file for the Triton server:

- **`json_to_proto.py`**: Converts JSON model configuration files to Protocol Buffer format
- **`json_to_proto.sh`**: Shell script wrapper for the conversion utility
