"""
Train a transformer model on a text dataset.
"""

import os
import torch
from tqdm import tqdm
from einops import rearrange
import time

from cs336_basics.transformer.model import Transformer
from cs336_basics.transformer.data import TokenizedDataset
from cs336_basics.transformer.utils import save_checkpoint, load_checkpoint
from cs336_basics.transformer.embedding import RotaryPositionalEmbedding
from cs336_basics.transformer.optimizers import AdamW, learning_rate_scheduler, gradient_clipping
from cs336_basics.transformer.loss import cross_entropy
from cs336_basics.common import Logger
from cs336_basics.tokenizer.bpe import BPE_Tokenizer
import wandb

class TransformerTrainer:
    """TransformerTraining is a class for training a transformer model.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        context_length: int = 32,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        rope: RotaryPositionalEmbedding = None,
        theta: float = 10000.0,
        device: torch.device = None,
        dtype: torch.dtype = None,
        # data
        dataset: str = "tinystories",
        data_path: str | dict[str, str] = "data/tokenized/TinyStoriesV2-GPT4-train.pkl",
        # Optimizer training hyperparameters
        batch_size: int = 32,
        optimizer: str = "adamw",
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        # Learning rate scheduler hyperparameters
        lr_scheduler: str = "cosine annealing",
        warmup_iters: int = 1000,
        max_lr: float = 0.001,
        min_lr: float = 1e-5,
        cosine_cycle_iters: int = 10000,
        # gradient clipping
        max_l2_norm: float = 1.0,
        #Tokenizer
        tokenizer: BPE_Tokenizer = None,
        # Logger
        logger: Logger = None,
        seed: int = 42,
        **kwargs
    ):
        
        run_config = {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "rope": rope is not None,
            "theta": theta,
            "device": str(device),
            "dtype": str(dtype),
            "dataset": dataset,
            "data_path": data_path,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
            "lr_scheduler": lr_scheduler,
            "warmup_iters": warmup_iters,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "cosine_cycle_iters": cosine_cycle_iters,
            "seed": seed,
            "gradient_clipping": max_l2_norm
        }
        
        timenow = time.strftime("%Y%m%d-%H%M%S")
        run_name = dataset + "_" + str(vocab_size) + "_" + str(context_length) + "_" + \
                    str(d_model) + "_" + str(num_layers) + "_" + str(num_heads) + \
                    "_" + str(d_ff) + "_" + str(max_lr) + "_" + str(min_lr) + "_" + timenow
        
        run_dir = os.path.join("runs", run_name)
        os.makedirs(run_dir, exist_ok=True)
        log_dir = os.path.join(run_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "training.log")
        
        if logger is None:
            logger = Logger(
                name="Transformer",
                log_file=log_file,
                level="INFO"  # default logging level
            )
            logger.info("Logger initialized for TransformerTraining.")

        if tokenizer is None:
            error_msg = "Tokenizer must be provided for TransformerTraining."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.checkpoint_frequency = getattr(kwargs, "checkpoint_frequency", 5000)
        run_config["checkpoint_frequency"] = self.checkpoint_frequency
        logger.info(f"Checkpoint frequency set to: {self.checkpoint_frequency}")
            
        self.validation_frequency = getattr(kwargs, "validation_frequency", 5000)
        run_config["validation_frequency"] = self.validation_frequency
        
        if kwargs.get("wandb") is not None:
            self.wandb = wandb.init(
                project="TransformerTraining",
                name=run_name,
                dir=run_dir,
                config=run_config,
            )

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope
        self.theta = theta
        self.device = device
        self.dtype = dtype
        self.tokenizer = tokenizer
        self.checkpoint_path = checkpoint_dir
        self.logger = logger
        self.max_l2_norm = max_l2_norm
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters
        
        self.model = Transformer(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope=rope,
            theta=theta,
            device=device,
            dtype=dtype
        )
        
        if optimizer == "adamw":
            self.optimizer = AdamW(
                params=self.model.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
                eps=eps
            )
        else:
            error_msg = f"Unsupported optimizer: {optimizer}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.train_dataset = None
        self.val_dataset = None
        if type(data_path) is str:
            data_path = {
                "train": data_path,
                "valid": None  # No validation set provided
            }
            self.logger.info(f"Validation dataset not provided, only training dataset will be used.")
        elif type(data_path) is dict:
            if "train" not in data_path:
                error_msg = "Training data path must be provided in the data_path dictionary."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            if "valid" not in data_path:
                self.logger.warning("Validation data path not provided, only training dataset will be used.")
                data_path["valid"] = None
        else:
            error_msg = "data_path must be a string or a dictionary with 'train' and 'valid' keys."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Data paths set: {data_path}")
    
        self.train_dataset = TokenizedDataset(
            tokenized_data_path=data_path["train"],
            batch_size=batch_size,
            context_length=context_length,
            total_num_steps=1000000,
            device=device,
            seed=seed
        )
        logger.info(f"Training dataset loaded from {data_path['train']}.")
        
        if data_path["valid"] is not None:
            self.val_dataset = TokenizedDataset(
                tokenized_data_path=data_path["valid"],
                batch_size=batch_size,
                context_length=context_length,
                total_num_steps=1000000,
                device=device,
                seed=seed
            )
            logger.info(f"Validation dataset loaded from {data_path['valid']}.")
            
    def train(self, num_steps: int = 100000, current_step: int = 0):
        """Train the transformer model for a specified number of steps.
        
        Parameters
        ----------
        num_steps : int, optional
            Number of training steps to run, by default 100000.
        """
        self.train_dataset.generate_seeds(total_num_steps=num_steps)
        self.model.train()
        
        self.logger.info(f"Starting training for {num_steps} steps from step {current_step+1}.")
        
        min_train_loss = float('inf')
        min_val_loss = float('inf')
        
        token_positions = torch.arange(self.context_length, device=self.device)
        torch.autograd.set_detect_anomaly(True)
        start_time = time.time()
        for step in tqdm(range(current_step+1, num_steps+1), desc="Training Steps"):
            step_start_time = time.time()
            # Get a batch of data
            lr_t = learning_rate_scheduler(
                step=step,
                max_lr=self.max_lr,
                min_lr=self.min_lr,
                warmup_iters=self.warmup_iters,
                cosine_cycle_iters=self.cosine_cycle_iters
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_t
            
            # Get a batch of data
            # Step is provided to ensure that we get the same batch across multiple training runs.
            # This is useful for debugging and reproducibility.
            input_batched_seqs, output_batched_seqs = self.train_dataset(step=step)
            
            # Forward pass
            logits = self.model(
                in_features=input_batched_seqs,
                token_positions=token_positions
            )
            
            # Compute loss
            logits_2d = rearrange(logits, 'b s v -> (b s) v')
            output_batched_seqs_1d = rearrange(output_batched_seqs, 'b s -> (b s)')
            loss = cross_entropy(
                logits=logits_2d,
                targets=output_batched_seqs_1d
            )
            
            # Gradient descent step
            self.optimizer.zero_grad()
            loss.backward()
            
            gradient_clipping(
                parameters=self.model.parameters(),
                max_l2_norm=self.max_l2_norm
            )
            self.optimizer.step()
            
            # Log the loss and learning rate
            self.logger.info(f"Step {step}: Loss = {loss.item():.4f}, Learning Rate = {lr_t:.6f}")
            wandb_log = {
                "train_loss": loss.item(),
                "learning_rate": lr_t,
                "step": step
            }
                
            if loss.item() < min_train_loss:
                min_train_loss = loss.item()
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    iteration=step,
                    checkpoint_path=os.path.join(self.checkpoint_path, "best_train_loss.pth"),
                )
                self.logger.info(f"New minimum training loss: {min_train_loss:.4f} at step {step}.")
                
            if step % self.checkpoint_frequency == 0:
                # Save the model checkpoint
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    iteration=step,
                    checkpoint_path=os.path.join(self.checkpoint_path, f"checkpoint_{step}.pth"),
                )
                self.logger.info(f"Checkpoint saved at step {step} to {self.checkpoint_path}.")
            
            if self.val_dataset is not None and step % self.validation_frequency == 0:
                # Validation step
                self.model.eval()
                with torch.no_grad():
                    val_input_batched_seqs, val_output_batched_seqs = self.val_dataset(step=step)
                    val_logits = self.model(
                        in_features=val_input_batched_seqs,
                        token_positions=token_positions
                    )
                    val_logits_2d = rearrange(val_logits, 'b s v -> (b s) v')
                    val_output_batched_seqs_1d = rearrange(val_output_batched_seqs, 'b s -> (b s)')
                    val_loss = cross_entropy(
                        logits=val_logits_2d,
                        targets=val_output_batched_seqs_1d
                    )
                    
                    
                    if val_loss.item() < min_val_loss:
                        min_val_loss = val_loss.item()
                        save_checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            iteration=step,
                            checkpoint_path=os.path.join(self.checkpoint_path, "best_val_loss.pth"),
                        )
                        self.logger.info(f"New minimum validation loss: {min_val_loss:.4f} at step {step}.")
                        
                    self.logger.info(f"Validation Step {step}: Loss = {val_loss.item():.4f}")
                
                    wandb_log["val_loss"] = val_loss.item()
                self.model.train()

            step_end_time = time.time()
            elapsed_time = step_end_time - step_start_time
            wandb_log["step_time"] = elapsed_time
            wandb_log["wall_time"] = step_end_time - start_time
            self.logger.info(f"Step {step} took {elapsed_time:.2f} seconds.")
            if self.wandb:
                wandb.log(wandb_log)
            
        self.logger.info(f"Training completed after {num_steps} steps.")
            
        
                
        
if __name__ == "__main__":
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
        context_length=256,
        d_model=512,
        num_layers=6,
        num_heads=16,
        d_ff=1344,
        rope=None, # Rope will be initialized later if needed
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float32,
        # Data
        dataset="tinystories",
        data_path=data_path,
        # Optimizer training hyperparameters
        batch_size=64,
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
    