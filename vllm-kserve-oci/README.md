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
export NUM_GPUS_PER_NODE=2
export GPU_TYPE=nvidia-l4
export GPU_MACHINE_TYPE=g2-standard-24
export MAX_NODES=3
export NUM_NODES=1
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
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Enable OCI in KServe

```bash
bash ./enable_oci.sh
```

### Deploy vLLM

```bash
kubectl apply -f serving-runtime.yml
```

Deploy a vLLM Infernece 
```bash
kubectl apply -f inference-service.yml
```


