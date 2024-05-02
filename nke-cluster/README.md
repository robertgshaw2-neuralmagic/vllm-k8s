Save your HF Token as a secret:

```bash
kubectl create secret generic hf-token --from-literal=hf-token=$HF_TOKEN
```

Deploy:

```bash
kubectl apply -f simple.yaml
```