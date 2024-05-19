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
python3 client.py --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 1.0
```

