# `ReadOnlyMany` PVC

For deploying an LLM which has signifcant network resources associated with downloading a file, we want to add models to a PVC that can be used in the cluster.

Additionally, since we are targeting deploying in a multi-node setup, we need a PVC that can be mounted to multiple nodes. We need:
- `ReadOnlyMany` (https://kubernetes.io/docs/concepts/storage/persistent-volumes/#access-modes)

The following guides were useful in developing this content:
- https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/volume-snapshots#create-snapshotclass
- https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/backup-pd-volume-snapshots
- https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/readonlymany-disks#volume-snapshot


## Create a Persistent Volumne

To mount a `ReadOnlyMany` filesystem, we need to create a persistent file store and snapshot it.

Run the following to make a pod we can use to download the file:
```bash
kubectl apply -f pvc-demo.yaml
kubectl exec -it pod-demo -- /bin/bash
python -m venv env
source env/bin/activate
pip install huggingface_hub[cli]
export MODEL_NAME="TinyLlama-1.1B-Chat-v1.0"
huggingface-cli download $MODEL_NAME --local-dir /mnt/models/$MODEL_NAME --cache-dir /mnt/models/$MODEL_NAME
```

Run the following to snapshot your PVC:
```bash
kubectl apply -f volumesnapshotclass.yaml
kubectl apply -f volumesnapshot.yaml
```

Deploy using the snapshot:
```bash
kubectl apply -f pvc-vllm.yaml
```



We can see that the service is deployed on multiple nodes with the same PVC:

```bash
(client-env) robertgshaw@Roberts-MacBook-Pro pvc-experiments % kubectl get pods -o custom-columns="NAME:.metadata.name,NODE:.spec.nodeName" 
NAME                                                     NODE
vllm-7f687dcb9f-8z6qj                                    gke-my-gpu-cluster-default-pool-320e69e4-rsx9
vllm-7f687dcb9f-994mf                                    gke-my-gpu-cluster-default-pool-320e69e4-bq4g
vllm-7f687dcb9f-jwmfr                                    gke-my-gpu-cluster-default-pool-320e69e4-sh2n
```