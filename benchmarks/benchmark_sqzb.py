import aiohttp
import argparse
import asyncio
import random
import requests
import time
import functools
from dataclasses import dataclass
from typing import AsyncGenerator, Any

import numpy as np
import pandas as pd
import torch
import tqdm
import tqdm.asyncio
from transformers import AutoTokenizer


@dataclass
class RequestResult():
    num_input_tokens: int

    num_generated_tokens: int
    generated_text: str

    arrival_time: float
    first_scheduled_time: float
    first_token_time: float
    finished_time: float
    
    client_side_total_latency: float

results: list[RequestResult] = []


def read_or_create_prompts(
    dataset_path: str,
    vocab_size: int,
    max_input_len: int,
    n: int,
) -> list[list[int]]:
    if dataset_path: 
        file_ext = dataset_path.split(".")[-1]
        match file_ext:
            case "parquet":
                reader = pd.read_parquet
            case "pkl":
                reader = pd.read_pickle
            case "csv":
                reader = pd.read_csv
            case _:
                raise NotImplementedError("UNSUPPORTED_DATASET_TYPE")
        df = reader(dataset_path)

        # team NAVER requested to report benchmark data excluding the input 
        # tokenization thus we tokenize our inputs in advance to exclude it
        assert "tok_inputs" in df.columns
        prompt_tok_ids = df["tok_inputs"][:n].apply(np.ndarray.tolist).to_list()
    else:
        # create list of random tok ids of fixed length when dataset isn't given
        randint_kwargs = dict(
            low=0, 
            high=vocab_size, 
            size=(max_input_len,)
        )
        randint = functools.partial(torch.randint, **randint_kwargs)
        prompt_tok_ids = [randint().tolist() for _ in range(n)]
        assert all(len(tok_ids) <= max_input_len for tok_ids in prompt_tok_ids)
    
    return prompt_tok_ids
    

async def get_prompt(
    prompts: list[list[int]],
    request_rate: float,
) -> AsyncGenerator[list[int], None]:
    for prompt in iter(prompts):
        yield prompt

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def get_model_id(url: str):
    response = requests.get(url).json()
    return response["data"][0]["id"]


async def send_request(
    url: str,
    token_ids: list[int],
    max_output_len: int,
    model_id: str,
    ignore_eos: bool,
    stop_token_ids: list[int],
) -> None:
    input_len = len(token_ids)
    payload = {
        "model": model_id,
        "prompt": token_ids,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_output_len,
        "ignore_eos": ignore_eos,
        "stop_token_ids": stop_token_ids
    }

    request_start_time = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=48 * 3600)
    session = aiohttp.ClientSession(timeout=timeout)
    try:
        async with session.post(url, json=payload) as response:
            result = await response.json()
        await session.close()
    except Exception as e:
        print(response)
        raise e

    request_end_time = time.perf_counter()

    assert input_len == result["usage"]["prompt_tokens"]
    
    results.append(RequestResult(
        num_input_tokens=input_len,
        num_generated_tokens=result["usage"]["completion_tokens"],
        generated_text=result["choices"][0]["text"],
        arrival_time=result["metrics"][0]["arrival_time"],
        first_scheduled_time=result["metrics"][0]["first_scheduled_time"],
        first_token_time=result["metrics"][0]["first_token_time"],
        finished_time=result["metrics"][0]["finished_time"],
        client_side_total_latency=request_end_time - request_start_time
    ))
    

async def benchmark(
    url: str,
    model_id: str,
    prompts: list[list[int]],
    max_output_len: int,
    ignore_eos: bool,
    stop_token_ids: list[int],
    request_rate: float,
) -> None:
    common_payloads: dict[str, Any] = dict(
        model_id=model_id,
        max_output_len=max_output_len,
        ignore_eos=ignore_eos,
        stop_token_ids=stop_token_ids,
    )

    tasks: list[asyncio.Task] = []
    async for tok_ids in get_prompt(prompts, request_rate):
        task = asyncio.create_task(
            send_request(url, tok_ids, **common_payloads))
        tasks.append(task)
    
    [await t for t in tqdm.asyncio.tqdm.as_completed(tasks)]


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    url = f"http://{args.host}:{args.port}"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    stop_token_ids = []
    if args.dataset:
        stop_token_ids = [tokenizer.eos_token_id]
        if (eot_id := tokenizer.get_vocab().get("<|eot_id|>", None)):
            stop_token_ids.append(eot_id) 

    model_id = get_model_id(url + "/v1/models")
    input_prompts = read_or_create_prompts(
        args.dataset, 
        tokenizer.vocab_size,
        args.max_input_len,
        args.num_requests
    )
    
    benchmark_start_time = time.perf_counter()
    asyncio.run(benchmark(url + "/v1/completions", model_id, input_prompts, 
                          args.max_output_len, not args.dataset, stop_token_ids,
                          args.request_rate)) 
    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time
    df = pd.DataFrame(data=results)

    total_input_tokens = df['num_input_tokens'].sum()
    total_generated_tokens = df['num_generated_tokens'].sum()
    print("SUMMARY")
    print(f"\t# requests: {args.num_requests}")
    print(f"\tTotal input tokens: {total_input_tokens}")
    print(f"\tTotal generated tokens: {total_generated_tokens}")
    print(f"\tTotal latency: {benchmark_time} sec")
    
    # team NAVER requested to report TTFT data excluding the queueing time
    # so we use first_scheduled_time instead of arrival_time
    sec_to_msec = 1000
    ttft = (df['first_token_time'] - df['first_scheduled_time']) * sec_to_msec
    print("TTFT")
    print(f"\tmedian: {ttft.median()} msec")
    print(f"\tmean: {ttft.mean()} msec")
    print(f"\tmax: {ttft.max()} msec")

    tpot = (df['finished_time'] - df['first_token_time']) * sec_to_msec
    tpot /= df['num_generated_tokens']
    print("TPOT")
    print(f"\tmedian: {tpot.median()} msec")
    print(f"\tmean: {tpot.mean()} msec")
    print(f"\tmax: {tpot.max()} msec")  

    out_path = model_id.split("/")[-1]
    out_path += f"_qps_{args.request_rate}"
    out_path += f"_total_{benchmark_time}"
    out_path += f"_in_{total_input_tokens}"
    out_path += f"_out_{total_generated_tokens}"
    out_path += f"_{args.dataset.split('/')[-1]}" if args.dataset else "_random"
    out_path += f"_{args.num_requests}"
    out_path += ".pkl"
    
    # df.to_pickle(out_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("-n", "--num-requests", type=int, default=1024)

    parser.add_argument(
        "--max-input-len", type=int, choices=[1024, 2048, 4096, 8192])
    parser.add_argument("--max-output-len", type=int, default=1024)

    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                             "then all the requests are sent at time 0. "
                             "Otherwise, we use Poisson process to synthesize "
                             "the request arrival times.")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    main(args)