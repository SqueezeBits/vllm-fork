import pandas as pd

splits = {
    "1k": "dynamic_sonnet_llama_3_prefix_256_max_1024_1024_sampled.parquet", 
    "2k": "dynamic_sonnet_llama_3_prefix_512_max_2048_1024_sampled.parquet", 
    "4k": "dynamic_sonnet_llama_3_prefix_1024_max_4096_1024_sampled.parquet", 
    "8k": "dynamic_sonnet_llama_3_prefix_2048_max_8192_1024_sampled.parquet"
}
for s in splits.values():
    df = pd.read_parquet("hf://datasets/squeezebits/dynamic_sonnet_llama3/" + s)
    df.to_parquet(s)
