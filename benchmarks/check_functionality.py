import textwrap

usage_examples = textwrap.dedent("""\
usage examples:
  extracting reference values to ref dir:
    python check_functionality.py --model <path to model> --dataset <path to dataset> ref

  extracting reference values with block size set to 128 and output dir specified:
    python check_functionality.py --model <path to model> --dataset <path to dataset> --block-size 128 ref --out <output dir>

  evaluating with reference values in ref dir:
    python check_functionality.py --model <path to model> --dataset <path to dataset> eval

  evaluating with reference values in specified dir:
    python check_functionality.py --model <path to model> --dataset <path to dataset> eval --ref <ref dir>

  evaluating with reference values in specified dir with output dir specified:
    python check_functionality.py --model <path to model> --dataset <path to dataset> eval --ref <ref dir> --out <output dir>

  evaluating with previous results (comparing ./ref and ./val):
    python check_functionality.py --model <path to model> --dataset <path to dataset> eval --skip-inference
""")

import argparse
import functools
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from vllm import LLM, SamplingParams
from vllm.model_executor.layers.sampler import (
    Sampler, 
    SamplerOutput, 
    SampleResultArgsType, 
    get_logprobs, 
    _build_sampler_output, 
    _sample, 
)
from vllm.model_executor.sampling_metadata import SamplingMetadata


@dataclass
class TokenStats():
    indices: List[int]
    logits: List[float]
    probs: List[float]
    logprobs: List[float]
    sampled: int


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


def parse_files(path: str) -> Dict[int, List[TokenStats]]:
    ret = defaultdict(list)
    assert os.path.exists(path) and os.path.isdir(path)
    base_dir = Path(path)

    for file in base_dir.glob("*.pkl"):
        req_id = int(file.stem)
        df = pd.read_pickle(file)
        for row in df.itertuples(index=False):
            ret[req_id].append(TokenStats(*row))

    return ret


def calc_per_token_logit_mse(ref: Dict[int, List[TokenStats]], val: Dict[int, List[TokenStats]]) -> List[np.ndarray]:
    offset = min(val.keys())
    assert len(ref) == len(val)
    assert all((k + offset) in val for k in ref.keys())

    ret: List[np.ndarray] = []
    for req_id, ref_stats in ref.items():
        val_stats = val[req_id + offset]

        mse_per_token = [sum((r - v)**2 for r, v in zip(rts.logits, vts.logits)) / len(rts.logits) for (rts, vts) in zip(ref_stats, val_stats)]
        ret.append(np.array(mse_per_token))

    return ret


def calc_per_token_prob_mae(ref: Dict[int, List[TokenStats]], val: Dict[int, List[TokenStats]]) -> List[np.ndarray]:
    offset = min(val.keys())
    assert len(ref) == len(val)
    assert all((k + offset) in val for k in ref.keys())

    ret: List[np.ndarray] = []
    for req_id, ref_stats in ref.items():
        val_stats = val[req_id + offset]

        mae_per_token = [sum(abs((r - v)) for r, v in zip(rts.probs, vts.probs)) / len(rts.logits) for (rts, vts) in zip(ref_stats, val_stats)]
        ret.append(np.array(mae_per_token))

    return ret


def main(args: argparse.Namespace):
    print(args)

    recorded: Dict[int, List[TokenStats]] = defaultdict(list)
    reference: Optional[Dict[int, List[TokenStats]]] = None
    if not (args.mode == "eval" and args.skip_inference):
        # monkey patch sampler
        def patched_forward(        
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata
        ) -> Optional[SamplerOutput]:
            if len(recorded):
                offset = min(recorded.keys())
            else:
                offset = min(sg.seq_ids[0] for sg in sampling_metadata.seq_groups)

            assert logits is not None
            batch_size, vocab_size = logits.shape

            if not sampling_metadata.reuse_sampling_tensors:
                self._init_sampling_tensors(logits, sampling_metadata)
            elif self._do_penalties:
                self._init_sampling_tensors(logits, sampling_metadata)

            assert self._sampling_tensors is not None
            sampling_tensors = self._sampling_tensors

            logits = logits.to(torch.float)
            logits.div_(sampling_tensors.temperatures.unsqueeze(dim=1))

            probs = torch.softmax(logits, dim=-1, dtype=torch.float)
            logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
            sampled = logprobs.argmax(dim=-1)
            
            num_pads = len([sg for sg in sampling_metadata.seq_groups if sg.seq_ids[0] == 0 and sg.sampling_params.max_tokens==16])
            overridden = []
            for i in range(batch_size - num_pads):
                seq_group = sampling_metadata.seq_groups[i]
                seq_id = seq_group.seq_ids[0]

                logit = logits[i]
                prob = probs[i]
                logprob = logprobs[i]
                if reference is not None:
                    if len(reference[seq_id - offset]) == 0:
                        overridden.append(sampled[i].item())
                        continue

                    ref = reference[seq_id - offset].pop(0)
                    idx = torch.tensor(ref.indices, device=logit.device)
                    stats = TokenStats(
                        idx.tolist(),
                        logit[idx].tolist(),
                        prob[idx].tolist(),
                        logprob[idx].tolist(),
                        sampled[i].item(), # type: ignore
                    )
                    recorded[seq_id].append(stats)

                    overridden.append(ref.sampled)
                else:
                    topk = prob.topk(args.topk)
                    stats = TokenStats(
                        topk.indices.tolist(),
                        logit[topk.indices].tolist(),
                        topk.values.tolist(),
                        logprob[topk.indices].tolist(),
                        -1,
                    )
                    recorded[seq_id].append(stats)

            if reference is not None:
                overridden = overridden + [overridden[0]] * num_pads
                probs = torch.nn.functional.one_hot(torch.tensor(overridden), vocab_size).to(device=logits.device, dtype=torch.float)
                logprobs = probs - 1

            maybe_deferred_sample_results, _ = _sample(
                probs,
                logprobs,
                sampling_metadata,
                sampling_tensors,
                include_gpu_probs_tensor=self.include_gpu_probs_tensor,
                modify_greedy_probs=self._should_modify_greedy_probs_inplace,
            )

            prompt_logprobs = None
            sample_logprobs = None
            if not sampling_metadata.skip_sampler_cpu_output:
                assert not isinstance(maybe_deferred_sample_results,
                                    SampleResultArgsType)
                prompt_logprobs, sample_logprobs = get_logprobs(
                    logprobs, sampling_metadata, maybe_deferred_sample_results)

            sampler_output = _build_sampler_output(
                maybe_deferred_sample_results,
                sampling_metadata,
                prompt_logprobs,
                sample_logprobs,
                on_device_tensors=None,
                skip_sampler_cpu_output=sampling_metadata.skip_sampler_cpu_output)
                            
            if reference is None:
                for o in sampler_output.outputs:
                    seq_id = o.samples[0].parent_seq_id
                    if seq_id not in recorded:
                        continue
                    recorded[seq_id][-1].sampled = o.samples[0].output_token

            return sampler_output

        Sampler.forward = patched_forward

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
        llm.generate(inputs[:min(10, len(inputs))], sampling_params=sampling_params, use_tqdm=False) # type: ignore
        recorded.clear()

        # record logits
        reference = parse_files(args.ref) if args.mode == "eval" else None
        outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=True) # type: ignore

        # validate results
        offset = min(recorded.keys())
        reference = parse_files(args.ref) if args.mode == "eval" else None
        for i, output in enumerate(outputs):
            o = output.outputs[0].token_ids
            r = reference[i] if args.mode == "eval" else recorded[i + offset] # type: ignore
            assert all(rt.sampled == ot for rt, ot in zip(r, o))
            recorded[i + offset] = recorded[i + offset][:len(o)]

        # save results
        Path(args.out).mkdir(exist_ok=True)
        for i, output in enumerate(outputs):
            pd.DataFrame(recorded[i + offset]).to_pickle(f"{args.out}/{i}.pkl")

    else: # evaluating difference with previously recorded values
        reference = parse_files(args.ref)
        recorded = parse_files(args.out)

    # do the math
    if args.mode == "eval":
        assert reference
        per_token_logit_mse = calc_per_token_logit_mse(reference, recorded) 
        per_token_prob_mae = calc_per_token_prob_mae(reference, recorded)
        
        per_request_logit_mse = np.array([mse.mean() for mse in per_token_logit_mse])
        per_request_prob_mae = np.array([mae.mean() for mae in per_token_prob_mae])

        print("REQUEST-WISE LOGITS DIFFERENCE(MSE):")
        print(f"\tMIN: {per_request_logit_mse.min()}")
        print(f"\tMEDIAN: {np.median(per_request_logit_mse)}")
        print(f"\tAVERAGE: {per_request_logit_mse.mean()}")
        print(f"\t90%: {np.percentile(per_request_logit_mse, 90)}")
        print(f"\t99%: {np.percentile(per_request_logit_mse, 99)}")
        print(f"\tMAX: {per_request_logit_mse.max()}")
        
        print("REQUEST-WISE PROBABILITY DIFFERENCE(MAE):")
        print(f"\tMIN: {per_request_prob_mae.min()}")
        print(f"\tMEDIAN: {np.median(per_request_prob_mae)}")
        print(f"\tAVERAGE: {per_request_prob_mae.mean()}")
        print(f"\t90%: {np.percentile(per_request_prob_mae, 90)}")
        print(f"\t99%: {np.percentile(per_request_prob_mae, 99)}")
        print(f"\tMAX: {per_request_prob_mae.max()}")



if __name__ == "__main__":
    class CustomFormatter(
        argparse.ArgumentDefaultsHelpFormatter, 
        argparse.RawDescriptionHelpFormatter):
        pass

    main_parser = argparse.ArgumentParser(formatter_class=CustomFormatter, epilog=usage_examples)
    main_parser.add_argument("--model", type=str, required=True, help="Name or path of the huggingface model to use.")
    main_parser.add_argument("--max-model-len", type=int, default=2048, help="Model context length.")
    main_parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="The fraction of device memory to be used for the model executor.")
    main_parser.add_argument("--block-size", type=int, default=16, help="Token block size for contiguous chunks of tokens.")
    main_parser.add_argument("--enforce-eager", action="store_true", help="Whether to disable graph mode execution.")

    main_parser.add_argument("--dataset", type=str, required=True, help="Path of the dataset to use.")    
    main_parser.add_argument("--max-input-len", type=int, default=1024, help="Maximum token length of samples in dataset.")
    main_parser.add_argument("--max-output-len", type=int, default=1024, help="Maximum number of tokens to generate.")
    main_parser.add_argument("-n", "--num-requests", type=int, default=100, help="Number of samples to use.")
    
    main_parser.add_argument("-k", "--topk", type=int, default=5, help="Number of top tokens to compare logits and probabilities.")

    subparsers = main_parser.add_subparsers(dest="mode", required=True, help="Mode to run the script.")
    ref_mode_parser = subparsers.add_parser("ref")
    ref_mode_parser.add_argument("--out", type=str, default="ref", help="Path of directory to store reference values.")        

    eval_mode_parser = subparsers.add_parser("eval")
    eval_mode_parser.add_argument("--ref", type=str, default="ref", help="Path to the directory containing reference values for comparison.")
    eval_mode_parser.add_argument("--out", type=str, default="val", help="Path to the directory to store current outputs which is being evaluated.")
    eval_mode_parser.add_argument("--skip-inference", action="store_true", help="Whether to skip inference and reuse the previously stored values.")

    args = main_parser.parse_args()
    main(args)
