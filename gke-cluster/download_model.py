from huggingface_hub import snapshot_download
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model-id", type=str)
parser.add_argument("--ignore-patterns", type=str, action="append")


if __name__ == "__main__":
    args = parser.parse_args()
    snapshot_download(repo_id=args.model_id, ignore_patterns=args.ignore_patterns)