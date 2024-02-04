import argparse
import asyncio
import json
import random
import time
import os
from typing import AsyncGenerator, List, Tuple

import aiohttp
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase, AutoTokenizer
# from vllm.transformers_utils.tokenizer import get_tokenizer

_GPU_TYPES = ["2xL4"]

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []

def make_synthetic_requests(
    num_input_words: int, 
    num_output_tokens: int,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    prompt = ("Hello_" * num_input_words)[:-1]
    num_input_tokens = len(tokenizer(prompt).input_ids)
    input_requests = [(prompt, num_input_tokens, num_output_tokens)] * num_requests
    return input_requests, num_input_tokens, num_output_tokens

async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(backend: str, model: str, api_url: str, prompt: str,
                       prompt_len: int, output_len: int, best_of: int,
                       use_beam_search: bool, pbar: tqdm) -> None:
    request_start_time = time.perf_counter()

    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        pload = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": False,
        }
        if model is not None:
            pload["model"] = model
    elif backend == "tgi":
        assert not use_beam_search
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": True,
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers,
                                    json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)

            # Re-send the request if it failed.
            if "error" not in output:
                break

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))
    pbar.update(1)

async def benchmark(
    backend: str,
    model: str,
    api_url: str,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []
    pbar = tqdm(total=len(input_requests))
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(
            send_request(backend, model, api_url, prompt, prompt_len,
                         output_len, best_of, use_beam_search, pbar))
        tasks.append(task)
    await asyncio.gather(*tasks)
    pbar.close()


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"{args.protocol}://{args.host}:{args.port}{args.endpoint}"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                              trust_remote_code=args.trust_remote_code)
    
    input_requests, num_input_tokens, num_output_tokens = make_synthetic_requests(
            num_input_words=args.num_input_words,
            num_output_tokens=args.num_output_tokens,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
        )
    print(f"QPS: {args.request_rate}")
    print(f"Num input tokens: {num_input_tokens}")
    print(f"Num output tokens: {num_output_tokens}")

    benchmark_start_time = time.perf_counter()
    asyncio.run(
        benchmark(args.backend, args.model, api_url, input_requests,
                  args.best_of, args.use_beam_search, args.request_rate))
    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {args.num_prompts / benchmark_time:.2f} requests/s")

    # Compute the latency statistics.
    latencies = [latency for _, _, latency in REQUEST_LATENCY]
    
    latency_results = {
        "model_name": args.model,
        "gpu_type": args.gpu_type,
        "input_tokens": num_input_tokens,
        "output_tokens": num_output_tokens,
        "qps": args.request_rate,
        "best_of": args.best_of,
        "use_beam_search": args.use_beam_search,
        "num_prompts": args.num_prompts
    }

    mean_latency = np.mean(latencies)
    print(f"Average latency: {mean_latency:.3f} s")
    latency_results["mean_latency"] = mean_latency

    percentiles = [50, 90, 95, 99, 100]
    p_latencies = np.percentile(latencies, percentiles)
    for p, p_latency in zip(percentiles, p_latencies.tolist()):
        print(f"P{p} latency: {p_latency:.2f} s")
        latency_results[f"p{p}_latency"] = p_latency

    avg_per_token_latency = np.mean([
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency in REQUEST_LATENCY
    ])
    print(f"Average latency per token: {avg_per_token_latency:.3f} s")
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency in REQUEST_LATENCY])
    print("Average latency per output token: "
          f"{avg_per_output_token_latency:.3f} s")

    df = pd.DataFrame.from_dict([latency_results])
    results_dir = f"results_{args.gpu_type}"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    csv_path = f"{results_dir}/results.csv"
    make_header = not os.path.isfile(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=make_header)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--backend",
                        type=str,
                        default="vllm",
                        choices=["vllm", "tgi"])
    parser.add_argument("--protocol",
                        type=str,
                        default="http",
                        choices=["http", "https"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--endpoint", type=str, default="/generate")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num-input-words", type=int, required=False, default=256,
                        help="Number of words to pass for `synthetic` data.")
    parser.add_argument("--num-output-tokens", type=int, required=False, default=256,
                        help="Number of output tokens to generate for `synthetic` data.")
    parser.add_argument("--tokenizer",
                        type=str,
                        required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--best-of",
                        type=int,
                        default=1,
                        help="Generates `best_of` sequences per prompt and "
                        "returns the best one.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--request-rate",
                        type=float,
                        default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                        "then all the requests are sent at time 0. "
                        "Otherwise, we use Poisson process to synthesize "
                        "the request arrival times.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument('--gpu-type',
                        type=str,
                        help='GPU setup you are profiling.',
                        choices=_GPU_TYPES)
    args = parser.parse_args()
    main(args)