#!/bin/bash

REGION="us-central1"
BUCKET_NAME="nm-vllm-models"
URL="gs://$BUCKET_NAME"

gcloud storage buckets create $URL --location $REGION
