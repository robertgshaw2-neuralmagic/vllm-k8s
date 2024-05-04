import argparse
import os
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="Hello my name is")

SERVICE_HOSTNAME = os.environ["SERVICE_HOSTNAME"]
INGRESS_HOST = os.environ["INGRESS_HOST"]
INGRESS_PORT = os.environ["INGRESS_PORT"]

base_url = f"http://{INGRESS_HOST}:{INGRESS_PORT}/v1/"
api_key = "null"

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
    default_headers={
        "Host": SERVICE_HOSTNAME,
    }
)

models = client.models.list()
model = models.data[0].id
prompt = parser.parse_args().prompt

# Completion API
completion = client.completions.create(
    model=model,
    prompt=prompt)

print(f"Prompt: {prompt}")
print(f"Completion: {completion.choices[0].text}")
