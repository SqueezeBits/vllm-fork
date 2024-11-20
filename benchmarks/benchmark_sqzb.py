import aiohttp
import argparse
import asyncio
import random
import requests
import time
import json
import functools
import requests
import functools
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Optional, List, Union, Dict, Tuple, Callable, Awaitable

import numpy as np
import pandas as pd
import torch
import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


RAW_RESULT = Tuple[Dict, int, float]
REQUESTS: List[Callable[[], Awaitable[RAW_RESULT]]] = []

class Entrypoint(Enum):
    OPENAI = 0
    # add entrypoint to benchmark

@dataclass
class RequestResult():
    num_input_tokens: int
    num_generated_tokens: int
    generated_text: str
    arrival_time: float
    first_scheduled_time: float
    first_token_time: float
    finished_time: float
    waiting_time: float
    client_side_total_latency: float


def sample_prompts(
    dataset: pd.DataFrame,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[List[int]]:
    # Only keep the first two turns of each conversation.
    dataset_list = [(row["conversations"][0]["value"],
                     row["conversations"][1]["value"])
                    for _, row in dataset.iterrows()
                    if len(row["conversations"]) >= 2]

    # Shuffle the dataset.
    random.shuffle(dataset_list)

    # Filter out sequences that are too long or too short
    filtered_dataset = []
    for conversation in dataset_list:
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt, completion = conversation
        prompt_token_ids = tokenizer(prompt).input_ids
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt_token_ids))

    return filtered_dataset


def read_or_create_prompts(
    dataset_path: str,
    vocab_size: int,
    max_input_len: int,
    n: int,
    tokenizer: Optional[PreTrainedTokenizerBase],
    mimic_throughput_sample: bool = False,
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
            case "json":
                reader = pd.read_json
            case _:
                raise NotImplementedError("UNSUPPORTED_DATASET_TYPE")
        df = reader(dataset_path)
        # team NAVER requested to report benchmark data excluding the input 
        # tokenization thus we tokenize our inputs in advance to exclude it
        if mimic_throughput_sample:
            assert tokenizer
            prompt_tok_ids = sample_prompts(df, n, tokenizer)
        else:
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
    

def create_request_callables(
    prompts: List[List[int]],  entrypoint: Entrypoint, url: str, model_id: str, max_output_len: int, 
    ignore_eos: bool, stop_token_ids: List[int], lora_pattern: List[Union[str, None]], random_lora: bool,
    json_template: Union[Dict, None]
) -> List[Callable[[], Awaitable[RAW_RESULT]]]:
    def get_model(index: int) -> str:
        if lora_pattern:
            lora = random.choice(lora_pattern) if random_lora else lora_pattern[index % len(lora_pattern)]
            return lora or model_id
        return model_id

    def get_endpoint_and_payload(request_id: int, token_ids: List[int]) -> Tuple[str, Dict[str, Union[str, List[int], float, int, bool]]]:
        model = get_model(request_id)
        if entrypoint == Entrypoint.OPENAI:
            endpoint = "/v1/completions"
            payload = {
                "model": model,
                "prompt": token_ids,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": max_output_len,
                "ignore_eos": ignore_eos,
                "stop_token_ids": stop_token_ids
            }
            if json_template:
                payload["guided_json"] = json_template
        else:
            raise NotImplementedError()

        return endpoint, payload

    async def request(request_id: int, token_ids: List[int]) -> RAW_RESULT:
        endpoint, payload = get_endpoint_and_payload(request_id, token_ids)
        timeout = aiohttp.ClientTimeout(total=48 * 3600)
        session = aiohttp.ClientSession(timeout=timeout)
        request_start_time = time.perf_counter()
        async with session.post(url + endpoint, json=payload) as response:
            result = await response.json()
        await session.close()

        request_end_time = time.perf_counter()    
        return result, len(token_ids), request_end_time - request_start_time

    return [functools.partial(request, i, token_ids) for i, token_ids in enumerate(prompts)]
    

def get_model_id(url: str):
    response = requests.get(url).json()
    return response["data"][0]["id"]


async def get_request(request_rate: float) -> AsyncGenerator[Callable[[], Awaitable[RAW_RESULT]], None]:
    for request in iter(REQUESTS):
        yield request

        if request_rate == float("inf"):
            continue

        interval = np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(interval)


def parse_raw_data(raw_data: RAW_RESULT) -> RequestResult:
    response = raw_data[0]
    input_token_len = raw_data[1]
    request_latency = raw_data[2]

    parsed = RequestResult(
        num_input_tokens=input_token_len,
        num_generated_tokens=response["usage"]["completion_tokens"],
        generated_text=response["choices"][0]["text"],
        arrival_time=response["metrics"][0]["arrival_time"],
        first_scheduled_time=response["metrics"][0]["first_scheduled_time"],
        first_token_time=response["metrics"][0]["first_token_time"],
        finished_time=response["metrics"][0]["finished_time"],
        waiting_time=response["metrics"][0]["time_in_queue"],
        client_side_total_latency=request_latency,
    )

    return parsed


async def benchmark(request_rate: float, concurrency: Union[int, None]) -> List[RAW_RESULT]:
    semaphore = asyncio.Semaphore(concurrency) if concurrency is not None else None
    async def concurrency_wrapper(request: Callable[[], Awaitable[RAW_RESULT]], progress_bar):
        if semaphore:
            async with semaphore:
                raw_result = await request()
        else:
            raw_result = await request()

        progress_bar.update(1)
        return raw_result
    
    with tqdm.tqdm(total=len(REQUESTS)) as progress_bar:
        tasks = []
        async for request in get_request(request_rate):
            tasks.append(asyncio.create_task(concurrency_wrapper(request, progress_bar)))

        outputs = await asyncio.gather(*tasks)
    return outputs


def get_unique_filepath(filepath: str) -> str:
    base, ext = os.path.splitext(filepath)
    counter = 1
    new_filepath = filepath

    while os.path.exists(new_filepath):
        new_filepath = f"{base}_{counter}{ext}"
        counter += 1
    return new_filepath


def main(args: argparse.Namespace):
    global REQUESTS

    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    url = f"http://{args.host}:{args.port}"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    stop_token_ids = []
    if args.dataset:
        stop_token_ids = [tokenizer.eos_token_id, tokenizer.get_vocab().get("<|eot_id|>", None)]
        stop_token_ids = [t for t in stop_token_ids if t is not None]

    prompts = read_or_create_prompts(
        args.dataset, 
        tokenizer.vocab_size,
        args.max_input_len,
        args.num_requests,
        tokenizer,
        args.mimic_throughput_sample,
    )

    model_id = get_model_id(url + "/v1/models")
    if args.json_template:
        with open(args.json_template, 'r') as file:
            json_template = json.load(file)
    else:
        json_template = None

    REQUESTS = create_request_callables(
        prompts, args.entrypoint, url, model_id, args.max_output_len, not args.dataset, 
        stop_token_ids, args.lora_pattern, args.random_lora, json_template
    )
    benchmark_start_time = time.perf_counter()
    raw_results = asyncio.run(benchmark(args.request_rate, args.concurrency))
    benchmark_end_time = time.perf_counter()
    benchmark_duration = benchmark_end_time - benchmark_start_time

    iteration_data = requests.get(url + "/iteration_data").json()
    requests.get(url + "/clear_iteration_data")
    
    total_iteration = iteration_data["num_iteration"]
    mean_bs = sum(bs[1] for bs in iteration_data["batch_sizes"]) / total_iteration

    results = [parse_raw_data(raw) for raw in raw_results]
    df = pd.DataFrame(data=results)

    total_input_tokens = df['num_input_tokens'].sum()
    total_generated_tokens = df['num_generated_tokens'].sum()
    print("SUMMARY")
    print(f"\t# requests: {args.num_requests}")
    print(f"\tTotal input tokens: {total_input_tokens}")
    print(f"\tTotal generated tokens: {total_generated_tokens}")
    print(f"\tTotal latency: {benchmark_duration} sec")
    print(f"\tTotal iteration: {total_iteration}")
    print(f"\tMean running batchsize: {mean_bs}")
    
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

    if args.save_result:
        base_dir = Path(args.result_dir)
        base_dir.mkdir(exist_ok=True)

        file_name = model_id.strip("/").split("/")[-1]
        file_name += f"_qps_{args.request_rate}"
        file_name += f"_concurrency_{args.concurrency}" if args.concurrency else ""
        file_name += f"_total_{benchmark_duration}"
        file_name += f"_iter_{total_iteration}"
        file_name += f"_batch_{mean_bs}"
        file_name += f"_in_{total_input_tokens}"
        file_name += f"_out_{total_generated_tokens}"
        file_name += "_LoRA" if args.lora_pattern else ""
        file_name += "_guided" if args.json_template else ""
        file_name += f"_{args.dataset.split('/')[-1]}" if args.dataset else "_random"
        file_name += f"_{args.num_requests}"
        file_name += ".pkl"

        df.to_pickle(base_dir / file_name)


def parse_entrypoint(value: str) -> Entrypoint: 
    return Entrypoint[value.upper()]


def parse_lora_pattern(value: str) -> List[Union[str, None]]:
    parts = value.split(',')
    return [part if part != '' else None for part in parts]

def parse_qps(value: str) -> float:
    if value == "inf" or float(value) == -1.0:
        return float("inf")
    
    return float(value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the online serving scenario.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--entrypoint", type=Entrypoint, default=Entrypoint.OPENAI)

    parser.add_argument("--max-input-len", type=int, required=True)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("-n", "--num-requests", type=int, default=1024)
    parser.add_argument("--mimic-throughput-sample", action='store_true',
                        help="Mimic request sampling process of "
                             "benchmark_throughput.py script.")

    parser.add_argument("--request-rate", type=parse_qps, default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                             "then all the requests are sent at time 0. "
                             "Otherwise, we use Poisson process to synthesize "
                             "the request arrival times.")
    parser.add_argument("--concurrency", type=int, default=None)

    parser.add_argument("--save-result", action='store_true', help="Save results to pkl file.")
    parser.add_argument("--result-dir", type=str, default=".")

    # guided json
    parser.add_argument("--json-template", type=str, default=None,
                        help="Path to guided json template.")
    
    # lora
    parser.add_argument("--lora-pattern", type=parse_lora_pattern, default=[],
                        help="Multi-batch LoRA ids. Skip LoRA for empty IDs. "
                             "e.g.: ,,sql-lora,sql-lora-2"
                             "LoRAs are applied in round-robin manner "
                             "unless random-lora flag is set.")
    parser.add_argument("--random-lora", action='store_true',
                        help="Shuffle lora-pattern randomly.")

    args = parser.parse_args()

    main(args)
