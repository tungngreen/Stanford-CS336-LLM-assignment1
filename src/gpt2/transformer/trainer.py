"""
Train a transformer model on a text dataset.
"""

import os
import torch
from tqdm import tqdm
from einops import rearrange
import time


from gpt2.transformer.model import Transformer
from gpt2.transformer.data import TokenizedDataset
from gpt2.transformer.utils import save_checkpoint, load_checkpoint, run_step
from gpt2.transformer.embedding import RotaryPositionalEmbedding
from gpt2.transformer.optimizers import AdamW, learning_rate_scheduler, gradient_clipping
from gpt2.transformer.loss import cross_entropy
from gpt2.common import Logger
from gpt2.tokenizer.bpe import BPE_Tokenizer
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

        self.log_dir = log_dir
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
            
            # Get a batch of data
            # Step is provided to ensure that we get the same batch across multiple training runs.
            # This is useful for debugging and reproducibility.
            input_batched_seqs, output_batched_seqs = self.train_dataset(step=step)
            
            kwargs = {
                "learning_rate": lr_t,
                "token_positions": token_positions,
                "max_l2_norm": self.max_l2_norm
            }
            
            loss = run_step(
                model=self.model,
                inputs=input_batched_seqs,
                outputs=output_batched_seqs,
                optimizer=self.optimizer,
                enable_backward=True,
                **kwargs
            )
            
            torch.cuda.synchronize()
            
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
            
            # Only run validation if:
            # 1. Validation dataset is provided
            # 2. Step is a multiple of validation frequency
            # 3. Profiling is not enabled (prof is None)
            if self.val_dataset is not None and step % self.validation_frequency == 0 and prof is None:
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
            
    def train_profiler(self, num_steps: int = 100000, current_step: int = 0, with_stack=False, enable_backward=False, compiled=False):
        """Train the transformer model with profiling for a specified number of steps.
        
        Parameters
        ----------
        num_steps : int, optional
            Number of training steps to run, by default 100000.
        """
        if compiled:
            self.model = torch.compile(self.model)
        
        from torch.profiler import profile, record_function, ProfilerActivity
        
        # 10 is the number of warmup steps to stabilize the profiler
        # We use `max_step` to initialize the dataset with enough steps
        max_step = max(num_steps, 10)
        
        self.train_dataset.generate_seeds(total_num_steps=max_step+1)
        self.model.train()
        
        self.logger.info(f"Starting training with profiler for {num_steps} steps from step {current_step+1}.")
        
        memory_snapshot = os.path.join(self.log_dir, "memory_snapshot.pickle")
        
        
        token_positions = torch.arange(self.context_length, device=self.device)
        torch.autograd.set_detect_anomaly(True)
        
        for warmup_step in range(10):
            # Warm-up steps to stabilize the profiler
            lr_t = learning_rate_scheduler(
                step=warmup_step,
                max_lr=self.max_lr,
                min_lr=self.min_lr,
                warmup_iters=self.warmup_iters,
                cosine_cycle_iters=self.cosine_cycle_iters
            )
            
            input_batched_seqs, output_batched_seqs = self.train_dataset(step=warmup_step)
            
            kwargs = {
                "learning_rate": lr_t,
                "token_positions": token_positions,
                "max_l2_norm": self.max_l2_norm
            }
            
            run_step(
                model=self.model,
                inputs=input_batched_seqs,
                outputs=output_batched_seqs,
                optimizer=self.optimizer,
                enable_backward=True,
                **kwargs
            )
            
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
            record_shapes=True,
            profile_memory=True,
            with_stack=with_stack
        ) as prof:
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
                
                # Get a batch of data
                input_batched_seqs, output_batched_seqs = self.train_dataset(step=step)
                
                kwargs = {
                    "learning_rate": lr_t,
                    "token_positions": token_positions,
                    "max_l2_norm": self.max_l2_norm
                }
                
                loss = run_step(
                    model=self.model,
                    inputs=input_batched_seqs,
                    outputs=output_batched_seqs,
                    optimizer=self.optimizer,
                    enable_backward=enable_backward,
                    **kwargs
                )
                
                prof.step()
                
                torch.cuda.synchronize()
                step_end_time = time.time()

                self.logger.info(f"Step {step}")
                wandb_log = {
                    "step": step,
                    "step_time": step_end_time - step_start_time,
                    "wall_time": step_end_time - start_time
                }
        prof.export_chrome_trace(
            os.path.join(self.log_dir, "profiler_trace.json"),
            # profile_memory=True,
            # with_stack=True   
        )
        
        torch.cuda.memory._dump_snapshot(
            memory_snapshot
        )
        torch.cuda.memory._record_memory_history(enabled=None)
        
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        self.logger.info(f"Profiler trace exported to {self.log_dir}/profiler_trace.json")
                
        
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
    