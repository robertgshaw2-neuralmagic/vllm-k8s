## vLLM in GKE

This guide shows how to setup a GKE cluster and deploy vLLM in K8s.

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
export POOL_NAME=l4-nodepool
export NUM_GPUS_PER_NODE=2
export GPU_TYPE=nvidia-l4
export MACHINE_TYPE=g2-standard-24
export MAX_NODES=3
export NUM_NODES=2
export MIN_NODES=0

gcloud container node-pools create ${POOL_NAME} \
    --cluster ${CLUSTER_NAME} \
    --accelerator type=${GPU_TYPE},count=${NUM_GPUS_PER_NODE},gpu-driver-version=latest \
    --machine-type $MACHINE_TYPE \
    --node-locations ${REGION}-a --region ${REGION} \
    --enable-autoscaling \
    --min-nodes ${MIN_NODES} --max-nodes ${MAX_NODES} --num-nodes ${NUM_NODES} \
    --ephemeral-storage-local-ssd=count=2
```

Connect to your cluster

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION} --project ${PROJECT_ID}
```






### Make A Container

Next, we will make a container with vLLM to run Hugging Face's

```bash

```
gcloud artifacts repositories create quickstart-docker-repo --repository-format=docker \
--location=us-central1 --description="Docker repository"