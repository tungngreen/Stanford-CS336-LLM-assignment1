import torch
import wandb
from cs336_basics.transformer.trainer import TransformerTrainer
from cs336_basics.tokenizer.bpe import BPE_Tokenizer


wandb.login(
    host="http://crystal:5000",
    key="local-457a9e8c8b72f707c6097ca5ed30cf734f3af223"
)

BPE_tokenizer = BPE_Tokenizer()
BPE_tokenizer.from_files(
    vocab_path="data/tokenizer/TinyStoriesV2-GPT4-vocab.pkl",
    merges_path="data/tokenizer/TinyStoriesV2-GPT4-merges.pkl",
    special_tokens=("<|endoftext|>")
)
data_path = {
    "train": "data/tokenized/TinyStoriesV2-GPT4-train_tokenized.npy",
    "valid": "data/tokenized/TinyStoriesV2-GPT4-valid_tokenized.npy",
}
# data_path = {
#     "train": "data/tokenized/test_tokenized.npy",
#     "valid": "data/tokenized/test_tokenized.npy"
# }
kwargs = {
    "wandb": True
}
# Example usage
trainer = TransformerTrainer(
    vocab_size=BPE_tokenizer.vocab_size,
    context_length=512,
    d_model=512,
    num_layers=6,
    num_heads=16,
    d_ff=1344,
    rope=None, # Rope will be initialized later if needed
    device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float32,
    # Data
    dataset="tinystories",
    data_path=data_path,
    # Optimizer training hyperparameters
    batch_size=32,
    optimizer="adamw",
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    eps=1e-8,
    # Learning rate scheduler hyperparameters
    lr_scheduler="cosine annealing",
    warmup_iters=1000,
    max_lr=0.0015,
    min_lr=0.00015,
    cosine_cycle_iters=18000,
    # Gradient clipping
    max_l2_norm=1.0,
    # Tokenizer
    tokenizer=BPE_tokenizer,
    # Checkpointing
    checkpoint_path="checkpoints",
    # Logger
    logger=None,  # Logger will be initialized in the class
    **kwargs   
)

trainer.train(num_steps=50_000, current_step=0)
