#!/bin/bash

SERVICE_HOSTNAME=$(kubectl get inferenceservice nm-vllm -n default -o jsonpath='{.status.url}' | cut -d "/" -f 3)
INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')

curl -H "Content-Type: application/json" \
     -H "Authorization: Bearer null" \
     -H "Host: ${SERVICE_HOSTNAME}" \
     "http://${INGRESS_HOST}:${INGRESS_PORT}/v1/completions" \
     -d '{"model": "meta-llama/Meta-Llama-3-8B", "prompt":"Hello my name is"}'