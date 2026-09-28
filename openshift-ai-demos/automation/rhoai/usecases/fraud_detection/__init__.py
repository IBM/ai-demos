"""Fraud Detection use case.

Deploys a complete fraud detection solution on RHOAI:
    - KServe model serving
    - TrustyAI monitoring

Public interface (consumed by usecases/registry.py):
    deploy(config)
    verify(config)
    cleanup(config)
"""

from rhoai.usecases.fraud_detection.cleanup import cleanup
from rhoai.usecases.fraud_detection.deploy import deploy
from rhoai.usecases.fraud_detection.verify import verify

__all__ = ["deploy", "verify", "cleanup"]
