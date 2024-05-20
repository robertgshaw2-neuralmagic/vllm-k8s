# vLLM + KServe

In this example, we will demonstrate how to run vLLM with:
- KServe / KNative
- Prometheus/Grafana Telemetry
- Concurrency-based Autoscaling
- `ReadOnlyMany` PVC for low cold-start times

This tutorial assumes you have KServe and Prometheus Operator Installed.

## Add Model To The Cluster

Given the size of we recommend pre-downloading your model artifacts into a PVC that can be 
mounted to all nodes via `ReadOnlyMany`.

### Save Model to a PVC

Create a pod and a `ReadWriteOnce` PVC to download the model:

```bash
kubectl apply -f pvc_download.yaml
```

Exec into the pod and download:
```bash
kubcetl cp download_llama.sh downloader-pod:/home
kubectl exec -it downloader-pod -- bash
```

Download a model:
```bash
bash /home/download_llama.sh
```

### Snapshot the Volume

Next, snapshot the volume so we can load it as a `ReadOnlyMany` PVC.

```bash
kubectl apply -f volumesnapshot.yaml
```

Once the volumesnapshot is `READY`, we can now mount it to each node running in the cluster.

```bash
kubectl get volumesnapshot \
  -o custom-columns='NAME:.metadata.name,READY:.status.readyToUse'

>> NAME                      READY
>> downloader-pvc-snapshot   true
```


### Create a ReadOnlyMany PVC From the Snapshot

Create a PVC with `ReadOnlyMany` based on the snapshot. This allows the same PVC to be mounted to each node.

```bash
kubectl apply -f pvc_runtime.yaml
```

## Deploy vLLM

### Launch InferneceService

Now, we can deploy a vLLM `InferenceService` which uses:
- `ReadOnlyMany` PVC
- `Autoscaling` targeting concurrency of 8

```bash
kubectl apply -f vllm.yaml
```

### Launch Monitoring

```bash
kubectl apply -f vllm_monitoring.yaml
```

Note: this step assumes you already have Prometheus Operator running in your K8s Cluser.

## Query The Server

Install client reqs:
```bash
cd vllm-client
python -m venv client-env
source client-env/bin/activate
pip install -r requirements.txt
```

Get endpoint:
```bash
export SERVICE_HOSTNAME=$(kubectl get inferenceservice llama-3 -n default -o jsonpath='{.status.url}' | cut -d "/" -f 3)
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
```

Hit the server with load:
```bash
python3 client.py --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 0.25 --num-prompts 100
```

Hit the server with more load:
```bash
python3 client.py --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 1.0
```

Observe the metrics in Grafana.
- http://localhost:3000/