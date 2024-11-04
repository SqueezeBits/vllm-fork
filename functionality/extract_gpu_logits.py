from io import TextIOWrapper
import os

import json

import argparse
import functools
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser


def read_or_create_prompts(
    dataset_path: str,
    vocab_size: int,
    max_input_len: int,
    n: int,
) -> List[List[int]]:
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

        assert "tok_inputs" in df.columns
        prompt_tok_ids = df["tok_inputs"][:n].apply(np.ndarray.tolist).to_list()
    else:
        randint_kwargs = dict(
            low=0, 
            high=vocab_size, 
            size=(max_input_len,)
        )
        randint = functools.partial(torch.randint, **randint_kwargs)
        prompt_tok_ids = [randint().tolist() for _ in range(n)]

    assert all(len(tok_ids) <= max_input_len for tok_ids in prompt_tok_ids)
    return prompt_tok_ids


offset = None
def record_logit(files: List[TextIOWrapper], k, logits: torch.Tensor, sampling_metadata) -> None:
    global offset
    if offset is None:
        offset = min(seq_groups.seq_ids[0] for seq_groups in sampling_metadata.seq_groups)
    assert offset is not None

    for i, seq_groups in enumerate(sampling_metadata.seq_groups):
        logit = logits[i].flatten()
        topk = logit.topk(k)

        val = topk.values.tolist()
        idx = topk.indices.tolist()

        seq_id = seq_groups.seq_ids[0]
        file_id = seq_id - offset
        file: TextIOWrapper = files[file_id]
        line = json.dumps({
            "logits": val,
            "indices": idx,
            "max": idx[val.index(max(val))]
        })
        file.write(f"{line}\n")

def override_probs(in_files: List[TextIOWrapper], out_files: List[TextIOWrapper], logits: torch.Tensor, sampling_metadata) -> Tuple[torch.Tensor, torch.Tensor]:
    global offset
    if offset is None:
        offset = min(seq_groups.seq_ids[0] for seq_groups in sampling_metadata.seq_groups)
    assert offset is not None

    _, vocab_size = logits.shape

    max_indices = []
    for i, seq_groups in enumerate(sampling_metadata.seq_groups):
        logit = logits[i].flatten()
        
        seq_id = seq_groups.seq_ids[0]
        if seq_id == 0:
            max_indices.append(max_indices[0])
            continue

        file_id = seq_id - offset
        in_file = in_files[file_id]
        out_file = out_files[file_id]
    
        ref = json.loads(in_file.readline())
        ref_idx = ref["indices"]
        ref_max = ref["max"]


        max_indices.append(ref_max)
        val = logit[ref_idx].tolist()

        line = json.dumps({
            "logits": val,
            "indices": ref_idx,
            "max": ref_max
        })
        out_file.write(f"{line}\n")

    probs = torch.nn.functional.one_hot(torch.tensor(max_indices), vocab_size).to(dtype=torch.float)
    log_probs = probs - 1

    return probs, log_probs

def main(args: argparse.Namespace):
    print(args)
    assert args.max_model_len >= args.max_input_len + args.max_output_len

    # set LLM engine
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    
    # set prompts
    prompt_token_ids = read_or_create_prompts(
        args.dataset, 
        tokenizer.vocab_size,
        args.max_input_len,
        args.num_requests,
    )
    inputs = [{
        "prompt_token_ids": prompt_token_id
    } for prompt_token_id in prompt_token_ids]

    # set sampling params
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.get_vocab().get("<|eot_id|>", None)]
    stop_token_ids = [t for t in stop_token_ids if t is not None]
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_output_len,
        ignore_eos=(args.dataset == ""),
        stop_token_ids=stop_token_ids,
    )
    
    # warmup
    llm.generate(inputs[:min(10, len(inputs))], sampling_params=sampling_params, use_tqdm=False)

    # create files to store the logits
    os.makedirs(args.out_path, exist_ok=True)
    logit_files = [open(f"{args.out_path}/{i}.logits", "wt") for i in range(len(inputs))]

    # run inference
    if args.in_path:
        ref_files = [open(f"{args.in_path}/{i}.logits", "rt") for i in range(len(inputs))]
        llm.llm_engine.model_executor.driver_worker.model_runner.model.model.sampler.override_probs = functools.partial(override_probs, ref_files, logit_files)
    else:
        llm.llm_engine.model_executor.driver_worker.model_runner.model.sampler.record_logit = functools.partial(record_logit, logit_files, args.topk)


    outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=True)
    assert len(outputs) == len(logit_files)
    for f in logit_files:
        f.close()
        
    # save results
    for i, output in enumerate(outputs):
        with open(f"{args.out_path}/{i}.tokens", "wt") as f:
            f.write("\n".join(map(str, output.outputs[0].token_ids)))


if __name__ == '__main__':
    parser = FlexibleArgumentParser()
    parser.add_argument('--model', type=str, default="/home/irteamsu/models/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset", type=str, default="/home/irteamsu/datasets/dynamic_sonnet_llama3/dynamic_sonnet_llama_3_prefix_256_max_1024_1024_sampled.parquet")    
    # parser.add_argument('--model', type=str)
    # parser.add_argument("--dataset", type=str)
    parser.add_argument('--max-model-len', type=int, default=2048)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--block-size', type=int, default=16)
    parser.add_argument('--enforce-eager', action='store_true')
    parser.add_argument("--max-input-len", type=int, choices=[1024, 2048, 4096, 8192], default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("-n", "--num-requests", type=int, default=100)

    parser.add_argument('-k', '--topk', type=int, default=5)
    parser.add_argument("--out-path", type=str, default="tmp")    
    parser.add_argument("--in-path", type=str, required=False)

    args = parser.parse_args()
    main(args)
