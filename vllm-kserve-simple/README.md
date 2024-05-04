## vLLM in GKE w/ KServe

This guide shows how to setup a GKE cluster and deploy vLLM in K8s using KServe.

### Make a Cluster

Set region and project id:
```bash
export REGION=us-central1
export PROJECT_ID=sandbox-377216
export CLUSTER_NAME=l4-cluster
```

Launch cluster:
```bash
gcloud container clusters create ${CLUSTER_NAME} --location ${REGION} \
  --workload-pool ${PROJECT_ID}.svc.id.goog \
  --enable-image-streaming \
  --node-locations=$REGION-a \
  --workload-pool=${PROJECT_ID}.svc.id.goog \
  --addons GcsFuseCsiDriver   \
  --machine-type n2d-standard-4 \
  --num-nodes 1 --min-nodes 1 --max-nodes 5 \
  --ephemeral-storage-local-ssd=count=2
```

Add a node pool. GKE Automatically taints the GPU nodes with the following:
- **Key:** `nvidia.com/gpu`
- **Effect:** `NoSchedule`

```bash
export POOL_NAME=nvidia-l4-nodepool
export NUM_GPUS_PER_NODE=1
export GPU_TYPE=nvidia-l4
export MACHINE_TYPE=g2-standard-8
export MAX_NODES=3
export NUM_NODES=2
export MIN_NODES=1

gcloud container node-pools create ${POOL_NAME} \
    --cluster ${CLUSTER_NAME} \
    --accelerator type=${GPU_TYPE},count=${NUM_GPUS_PER_NODE},gpu-driver-version=latest \
    --machine-type $MACHINE_TYPE \
    --node-locations ${REGION}-a --region ${REGION} \
    --min-nodes ${MIN_NODES} --max-nodes ${MAX_NODES} --num-nodes ${NUM_NODES}
```

Point `kubectl` at your cluster.

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION} --project ${PROJECT_ID}
```


### Deploy vLLM To Your Cluser with KServe

Install KServe:
```bash
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.12/hack/quick_install.sh" | bash
```

Set Secret:
```bash
export HF_TOKEN={your_token}
kubectl create secret generic hf-token --from-literal=hf-token="$HF_TOKEN"
```

Create your `ServingRuntime` and `InferenceService`
```bash
kubectl apply -f nm-vllm.yml
```

Install client reqs:
```bash
python -m venv client-env
source client-env/bin/activate
pip install openai
```

Send a request:

```bash
SERVICE_HOSTNAME=$(kubectl get inferenceservice nm-vllm -n default -o jsonpath='{.status.url}' | cut -d "/" -f 3)
INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')

python3 sample-client.py --prompt "vLLM is the best inference server for LLMs because"
```