from io import TextIOWrapper
import os

import argparse
import functools
from typing import List

import numpy as np
import pandas as pd
import torch

from vllm import LLM, SamplingParams
from vllm.inputs import PromptInputs
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
def record_logit(files: List[TextIOWrapper], logits, sampling_metadata) -> None:
    global offset
    if offset is None:
        offset = min(seq_groups.seq_ids[0] for seq_groups in sampling_metadata.seq_groups)
    assert offset is not None

    for i, seq_groups in enumerate(sampling_metadata.seq_groups):
        logit = logits[i].tolist()
        seq_id = seq_groups.seq_ids[0]
        file_id = seq_id - offset
        file: TextIOWrapper = files[file_id]
        file.write(f"{','.join(map(str, logit))}\n")


def override_logit(files: List[TextIOWrapper], logits, sampling_metadata) -> torch.Tensor:
    global offset
    assert offset

    device = logits.device
    dtype = logits.dtype
    overrided = []
    for seq_groups in sampling_metadata.seq_groups:        
        seq_id = seq_groups.seq_ids[0]
        file_id = seq_id - offset
        
        file: TextIOWrapper = files[file_id]
        logit = [float(l) for l in file.readline().strip("\n").split(",")]
        overrided.append(logit)

    return torch.tensor(overrided, device=device, dtype=dtype)



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
    inputs: List[PromptInputs] = [{
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

    # create tmp files to store the logits
    # assert not os.path.exists("tmp")
    os.makedirs("tmp", exist_ok=True)
    output_logit_files = [open(f"tmp/{i}.logits", "wt") for i in range(len(inputs))]
    input_logits_files = [open(f"res/{i}.logits") for i in range(len(inputs))]

    # run inference
    llm.llm_engine.model_executor.driver_worker.model_runner.model.sampler.record_logit = functools.partial(record_logit, output_logit_files)
    llm.llm_engine.model_executor.driver_worker.model_runner.model.sampler.override_logit = functools.partial(override_logit, input_logits_files)
    outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=True)
    assert len(outputs) == len(output_logit_files)
    for f in output_logit_files:
        f.close()
        
    # save results
    for i, output in enumerate(outputs):
        with open(f"tmp/{i}.tokens", "wt") as f:
            f.write("\n".join(map(str, output.outputs[0].token_ids)))


    # save input if needed
    if args.dataset == "":
        random_data = pd.DataFrame([{"tok_inputs": token_ids} for token_ids in prompt_token_ids])
        out_path = f"max-input-len_{args.max_input_len}"
        out_path += f"_max-output-len_{args.max_output_len}"
        out_path += f"_n_{args.num_requests}"
        out_path += ".parquet"
        random_data.to_pickle(out_path)


if __name__ == '__main__':
    parser = FlexibleArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument("--dataset", type=str, default="")    
    parser.add_argument('--max-model-len', type=int)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--block-size', type=int, default=16)
    parser.add_argument('--enforce-eager', action='store_true')
    parser.add_argument("--max-input-len", type=int, choices=[1024, 2048, 4096, 8192], required=True)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("-n", "--num-requests", type=int, default=128)

    args = parser.parse_args()
    main(args)
