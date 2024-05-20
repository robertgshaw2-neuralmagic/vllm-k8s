### Simulate Client Requests

Install requirements for client:
```bash
pip install -r requirements.txt
```

Download some sample data:
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Launch clients:

```bash
export SERVICE_HOSTNAME=$(kubectl get inferenceservice llama-3 -n default -o jsonpath='{.status.url}' | cut -d "/" -f 3)
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
python3 client.py --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 1.0
```
