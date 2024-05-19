## Set Up A Kubernetes Cluster

First, create a Kubernetes Cluster with Nvidia GPUs. In this example, we will use GKE.

### Make a GKE Cluster with GPUs

First, create the cluster (swap your GCP `PROJECT_ID`):

```bash
export PROJECT_ID=sandbox-377216
export CLUSTER_NAME=my-gpu-cluster
export REGION=us-central1
export ZONE=${REGION}-c
export MACHINE_TYPE=e2-standard-16

gcloud container clusters create ${CLUSTER_NAME} \
    --workload-pool ${PROJECT_ID}.svc.id.goog \
    --machine-type ${MACHINE_TYPE} \
    --location ${ZONE} \
    --num-nodes 1
```

Next, add a `nodepool` with GPUs (in this case, we use L4s):

```bash
export POOL_NAME=my-gpu-nodepool
export NUM_GPUS_PER_NODE=1
export GPU_TYPE=nvidia-l4
export GPU_MACHINE_TYPE=g2-standard-8
export MAX_NODES=3
export NUM_NODES=3
export MIN_NODES=1

gcloud container node-pools create ${POOL_NAME} \
    --cluster ${CLUSTER_NAME} \
    --accelerator type=${GPU_TYPE},count=${NUM_GPUS_PER_NODE},gpu-driver-version=latest \
    --machine-type ${GPU_MACHINE_TYPE} \
    --node-locations ${ZONE} \
    --min-nodes ${MIN_NODES} \
    --max-nodes ${MAX_NODES} \
    --num-nodes ${NUM_NODES}
```

Third, point `kubectl` at your cluster:

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} \
    --zone ${ZONE} \
    --project ${PROJECT_ID}
```

## Install KServe

Install with:

```bash
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.12/hack/quick_install.sh" | bash
```

For more detailed installation instructions, [checkout the docs](https://kserve.github.io/website/0.12/admin/serverless/serverless/).

## Deploy vLLM With KServe

### Create a Model OCI

Install Reqs To Download Model:
```bash
python -m venv download-env
source download-env/bin/activate
pip install -r requirements-download.txt
```

Download Model From Hub:
```bash
export HF_TOKEN={your_token}
bash download_from_hub.sh -m meta-llama/Meta-Llama-3-8B-Instruct
```

Add model to OCI image:
```bash
docker build --build-arg="MODEL_PATH=$PWD/models/meta-llama/Meta-Llama-3-8B-Instruct" -t robertgouldshaw2/llama-3:v0.1 .
```

Push OCI image to docker hub:
```bash
docker push robertgouldshaw2/llama-3:v0.1
```

### Enable OCI in KServe

```bash
bash ./enable_oci.sh
```

### Deploy vLLM

```bash
kubectl apply -f vllm.yml
```

### Query The Server

Install client reqs:
```bash
python -m venv client-env
source client-env/bin/activate
pip install openai
```

Send a request:

```bash
export SERVICE_HOSTNAME=$(kubectl get inferenceservice tinyllama -n default -o jsonpath='{.status.url}' | cut -d "/" -f 3)
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')

python3 sample-client.py --prompt "vLLM is the best inference server for LLMs because"
```

### Configure Prometheus

- https://kserve.github.io/website/0.10/modelserving/observability/prometheus_metrics/
- https://knative.dev/development/serving/observability/metrics/collecting-metrics/
- https://github.com/kserve/kserve/blob/master/qpext/README.md
- https://github.com/knative-extensions/monitoring/tree/main/grafana

Install Prometheus Stack:
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/kube-prometheus-stack -n default -f prom_values.yml
```

Apply ServiceMonitor:
```bash
kubectl create namespace knative-eventing
kubectl apply -f https://raw.githubusercontent.com/knative-extensions/monitoring/main/servicemonitor.yaml
```

Save this file as `qpext_image_patch.yaml`:
```bash
data:
  queue-sidecar-image: kserve/qpext:latest
```
