# OGX Demos

## What is OGX?

OGX is an open source AI application server. It brings together model inference, vector stores, file storage, tool calling, and agentic orchestration behind a single OpenAI-compatible API. Applications can be written in any language, while models and providers can be changed without modifying application code.

In the demos, OGX runs as a component of Red Hat OpenShift AI.


## Prerequisites

Complete the setup steps in the [OpenShift AI README](../README.md#getting-started):

1. Configure `DSCInitialization` and `DataScienceCluster`. The DataScienceCluster must have both **KServe** (model serving) and **OGX** enabled.

2. Ensure the cluster has a default `StorageClass` for provisioning persistent volumes.

## Available Demos

### TechMart Customer Service Assistant

A customer service assistant for a fictional electronics retailer, combining RAG over policy documents with MCP tool calls against a PostgreSQL order database. Uses Qwen3-4B-Instruct served with vLLM and a Flask-based chat UI.
 
[→ Explore TechMart Customer Service](techmart-customer-service/)

---

## Shared Resources

[`shared/`](shared/) contains resources shared across OGX demos.

- [`ogx-metadata-postgres.yaml`](shared/ogx-metadata-postgres.yaml) deploys the PostgreSQL instance used by the OGX server to store its internal state, including vector store metadata, conversation history, and uploaded file records. It also creates the `postgres-credentials` secret required by the OGX server.

> **Note:** Deploy `ogx-metadata-postgres.yaml` before the OGX server. If you use a namespace other than `ogx-sandbox`, update the `POSTGRES_HOST` value accordingly.

## Resources

- [OGX Documentation](https://ogx-ai.github.io/)
- [Red Hat OpenShift AI Docs](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed)
