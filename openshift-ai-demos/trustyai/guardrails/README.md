# Guardrails

Protect LLM applications by deploying safety guardrails that monitor and validate user inputs and model outputs in real-time.

---

## Available Frameworks

### FMS Guardrails
An open-source orchestration service for configuring and running a pipeline of independent detectors (such as HAP or competitor-brand checkers) against user prompts and model completions.

- **Available Demos**:
  - **Lemonade Stand**: Configure regex filters and a Granite Guardian Hate, Abuse, and Profanity (HAP) model detector to protect a customer service chatbot.

[→ View FMS Guardrails Guide](fms-guardrails/lemonade-stand/)

---

### NeMo Guardrails
NVIDIA's toolkit for defining conversational guardrails using Colang. It allows you to declare rule-based dialog flows, restrict topics, and trigger custom programmatic validation actions.

- **Available Demos**:
  - **TechGear Assistant**: Define Colang flows and custom Python actions to protect an e-commerce chatbot from off-topic requests, PII leakage, and competitor brand mentions.

[→ View NeMo Guardrails Guide](nemo-guardrails/techgear-assistant/)
