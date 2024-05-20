### Load model into a PVC

Create a PVC and pod that we can use to download a model onto the cluster.

```bash
kubectl apply -f downloader-pvc.yaml
```

We can see that the PVC is created where we will store the model:

```bash
kubectl get pvc

>> NAME             STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   AGE
>> downloader-pvc   Bound    pvc-2daf80b8-cb56-4e5d-bcbb-042143300bfc   30Gi       RWO            premium-rwo    92s
```

Exec into the pod and download:

```bash
kubectl exec -it downloader-pod -- bash
```

Download a model:

```bash
export HF_TOKEN=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
python3 -m venv env
source env/bin/activate
pip install huggingface_hub[cli]
export MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
huggingface-cli download $MODEL_ID --local-dir /mnt/$MODEL_ID --cache-dir /mnt/$MODEL_ID --exclude *.pt*
exit
```

Snapshot the volume so we can load it as a `ReadOnlyMany` PVC that can be mounted on multiple nodes in the cluster.

```bash
kubectl apply -f volumesnapshotclass.yaml
kubectl apply -f volumesnapshot.yaml
```

Once the volumesnapshot is `READY`, we can now mount it to each node running in the cluster.

```bash
kubectl get volumesnapshot \
  -o custom-columns='NAME:.metadata.name,READY:.status.readyToUse'

>> NAME                      READY
>> downloader-pvc-snapshot   true
```

Create a PVC with `ReadOnlyMany` based on the snapshot. This will allow us to mount the PVC to each 

```bash
kubectl apply -f runtime_pvc.yaml
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
