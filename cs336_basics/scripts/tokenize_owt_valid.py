import os
from cs336_basics.tokenizer.bpe import BPE_Tokenizer

input_file = "data/TinyStoriesV2-GPT4-train.txt"
output_dir = "data/tokenized"
tokenizer = BPE_Tokenizer(
    vocab="data/tokenizer/TinyStoriesV2-GPT4-vocab.pkl",
    merges="data/tokenizer/TinyStoriesV2-GPT4-merges.pkl",
    special_tokens=["<|endoftext|>"],
    verbose=10
)

max_length = 1024

tokenizer.tokenize_and_save_parallel(
    input_file=input_file,
    output_dir=output_dir,
    num_processes=40,
    num_buffer_tokens=500_000_000 # Approx 2MB buffer for token IDs (assuming np.int32)
)
