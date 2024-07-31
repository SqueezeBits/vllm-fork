# vLLM with Chunked Prefill for Gaudi® 2 AI Accelerators

## About Chunked Prefill
[[paper](https://arxiv.org/pdf/2308.16369)] | [[github](https://github.com/vllm-project/vllm/issues/3130)]

Chunked-prefill with piggybacking was introduced in **SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills**. It employs *prefill-chunking*, which splits a prefill request into
equal sized chunks, and *piggybacking*, which constructs a batch using a single prefill chunk and populates the
remaining slots with decodes. Since prefill requests are compute-bound and decoding requests are memory-bound, it can improve efficiency by balancing between the two.

## Benefits on Gaudi® 2
### Possible advantages
- Reduced the number of traced graphs, allowing more memory for KV cache and fewer preemptions, which can lead to larger batch size and longer sequences.
- Improved decoding throughput by piggybacking decoding tokens.

### Possible disadvantages
- Increased number of graph launches per single input sequence.
- Attention overhead due to piggybacking.

## Usage
Both prefill chunking and piggybacking can be enabled with the `enable_chunked_prefill` parameter when initializing an `LLM` instance.
```python
llm = LLM(
        model=model,
        ...
        enable_chunked_prefill=True,
        ...
    )
```
For benchmarks, it is included in [benchmark_throughput.py](./benchmarks/benchmark_throughput.py) script.
- Example command for running with 1024 fixed-length inputs of 8K and outputs of 1K
```bash
python benchmarks/benchmark_throughput.py --model /models/Meta-Llama-3-8B-Instruct/ --trust-remote-code --enforce-eager --num-prompts 1024  --input-len 8192 --output-len 1024 --max-num-batched-tokens 4096 --max-num-seqs 128 --enable-chunked-prefill
```

## Implementation Details
The following features were implemented from the [base commit](https://github.com/SqueezeBits/vllm-fork/commit/60df2350b9455fa1c87b319a300840df7267f49e) from habana_main branch on vllm-fork.
- Modified the shape of inputs from 2D to 1D.
- Implemented chunked prefill scheduler.
- Implemented paged-prompt-attention for the prefill sequences having KV caches from the previous chunks.
- Implemented attention mask to properly handle different sequences and past contexts.
- Implemented piggybacking on scheduler and attention.
- Optimized HPU graphs behaviors.

## Future Works
- Rebase to the latest habana_main branch. (after habana_next being merged).
- Minimize internal paddings and tensor manipulations such as slicing and concatenating for further optimization.
