from cs336_basics.bpe import BPE_Tokenizer
from cs336_basics.utils import save_with_pickle
import json
import pickle

"""
    Train a BPE tokenizer on the TinyStories dataset.
    Serialize the vocabulary and merges to disk.
"""
    

if __name__ == "__main__":
    
    bpe_tokenizer = BPE_Tokenizer(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )
    
    bpe_tokenizer.train()
    
    # Save the vocabulary and merges to disk
    print(f"Vocabulary size: {len(bpe_tokenizer.vocab)}")
    print(f"Merges size: {len(bpe_tokenizer.merges)}")
    save_with_pickle(
        "data/tinystories_vocab.pkl",
        "data/tinystories_merges.pkl",
        bpe_tokenizer.vocab,
        bpe_tokenizer.merges
    )
