"""
Analyzing the vocabulary after training a BPE tokenizer on TinyStories and OpenWebText.
"""

import pickle
from cs336_basics.utils import load_with_pickle

vocab_path = "data/tinystories_vocab.pkl"
merges_path = "data/tinystories_merges.pkl"


vocab, merges = load_with_pickle(vocab_path, merges_path)

# Find the 10 longest tokens in the vocabulary
longest_tokens = sorted(vocab.items(), key=lambda x: len(x[1]), reverse=True)[:10]
print("10 longest tokens in the vocabulary:")
for token, bytes_value in longest_tokens:
    print(f"Token: {token}, Bytes: {bytes_value}, Length: {len(bytes_value)}")
    
    
vocab_path = "data/owt_vocab.pkl"
merges_path = "data/owt_merges.pkl"


vocab, merges = load_with_pickle(vocab_path, merges_path)

# Find the 10 longest tokens in the vocabulary
longest_tokens = sorted(vocab.items(), key=lambda x: len(x[1]), reverse=True)[:10]
print("10 longest tokens in the vocabulary:")
for token, bytes_value in longest_tokens:
    print(f"Token: {token}, Bytes: {bytes_value}, Length: {len(bytes_value)}")