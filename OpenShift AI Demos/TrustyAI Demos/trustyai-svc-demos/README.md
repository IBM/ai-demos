# Configuring TrustyAI 
This guide provides steps to enable and deploy the TrustyAI component in RHOAI.

### Prerequisite
If the TrustyAI component is not enabled in RHOAI, please follow these steps - [Configure the RHOAI for RawDeployment](../README.md)

Configuring monitoring for user-defined projects:  
```
oc create -f - <<EOF 
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-monitoring-config
  namespace: openshift-monitoring
data:
  config.yaml: |
    enableUserWorkload: true
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: user-workload-monitoring-config
  namespace: openshift-user-workload-monitoring
data:
  config.yaml: |
    prometheus:
      logLevel: debug
      retention: 15d
EOF
```
Reference - [Configure Monitoring](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.25/html/managing_and_monitoring_models/managing_and_monitoring_models_on_the_single_model_serving_platform#configuring-monitoring-for-the-single-model-serving-platform_cluster-admin)

---

### 1. Enable TrustyAI for RawDeployment
Follow this procedure to enable the TrustyAI service in the RawDeployment. This step is not required in serverless mode.

Patch the `inferenceservice-config` configmap to allow updating the `inferenceservice-config`
```
oc patch configmap inferenceservice-config -n redhat-ods-applications \
--type merge -p '{"metadata": {"annotations": {"opendatahub.io/managed": "false"}}}'
```

To enable TrustyAI for RawDeployment follow this steps - [enabling-trustyai-kserve-integration_monitor](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_cloud_service/1/html/monitoring_data_science_models/configuring-trustyai_monitor#enabling-trustyai-kserve-integration_monitor).

### 2. Create a project:
Create a namespace in your OpenShift cluster where TrustyAI will be deployed.
```
oc new-project trustyai-demo
```

### 3. Deploy TrustyAI service
TrustyAI service can be configured with either a PVC or a relational database (e.g., MySQL, MariaDB). Using a database improves scalability, performance, and data management.

#### 3.1 Deploy with PVC
```
oc create -f - <<EOF 
apiVersion: trustyai.opendatahub.io/v1alpha1
kind: TrustyAIService
metadata:
  name: trustyai-service
  annotations:
    opendatahub.io/enable-route: "true"
spec:
  storage:
    format: "PVC"
    folder: "/inputs"
    size: "1Gi"
  data:
    filename: "data.csv"
    format: "CSV"
  metrics:
    schedule: "5s" 
EOF
```

#### 3.2 Deploy with database  
i. Deploy the Database Pod and Service
```
oc run trustyai-mariadb --image=rhel9/mariadb-1011 --env="MYSQL_USER=trustyai" --env="MYSQL_PASSWORD=asd123" --env="MYSQL_DATABASE=trustyai_db"
oc expose pod/trustyai-mariadb --port 3306
```

ii. Create a Secret with Database Credentials
```
oc create -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: mariadb-credentials
type: Opaque
stringData:
  databaseKind: mariadb
  databaseUsername: trustyai
  databasePassword: 'asd123'
  databaseService: trustyai-mariadb
  databasePort: '3306' 
  databaseGeneration: update 
  databaseName: trustyai_db 
EOF
```

iii. Create the TrustyAI Service
```
oc create -f - <<EOF 
apiVersion: trustyai.opendatahub.io/v1alpha1
kind: TrustyAIService
metadata:
  name: trustyai-service
spec:
  storage:
    format: "DATABASE"
    size: "1Gi"
    databaseConfigurations: "mariadb-credentials"
  metrics:
    schedule: "5s"
EOF
```
