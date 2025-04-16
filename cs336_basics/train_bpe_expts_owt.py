from cs336_basics.bpe import BPE_Tokenizer
from cs336_basics.utils import save_with_pickle
import wandb
"""
    Train a BPE tokenizer on the TinyStories dataset.
    Serialize the vocabulary and merges to disk.
"""
    

if __name__ == "__main__":
    
    kwargs = {
        "wandb": True,
    }
    
    wandb.login(
        host="http://wandb-local:8080",
        key="local-457a9e8c8b72f707c6097ca5ed30cf734f3af223"
    )
    
    bpe_tokenizer = BPE_Tokenizer()
    bpe_tokenizer.prepare_training_data(
        input_path="data/owt_train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
        **kwargs
    )
    bpe_tokenizer.train(parallel=False)

    # Save the vocabulary and merges to disk
    print(f"Vocabulary size: {len(bpe_tokenizer.vocab)}")
    print(f"Merges size: {len(bpe_tokenizer.merges)}")
    save_with_pickle(
        "data/owt_vocab.pkl",
        "data/owt_merges.pkl",
        bpe_tokenizer.vocab,
        bpe_tokenizer.merges
    )
