"""
Implementation (from scratch) of the data module for the Transformer model.
"""

import torch
from typing import Tuple
from jaxtyping import Int
import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging
from torch.utils.data import DataLoader, Dataset
import logging

def data_loading(
    token_ids: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str = 'cpu',
    seed: int = None,
    **kwargs
) -> Tuple[Int[torch.Tensor, 'batch_size context_length'], Int[torch.Tensor, 'batch_size context_length']]:
    """data_loading loads the data for the Transformer model.
    Parameters
    ----------
    token_ids : np.ndarray
        Array of token IDs to be used for training.
    batch_size : int
        Number of samples per batch.
    context_length : int
        Length of the context window for each sample.
    device : str, optional
        Device to load the data onto, by default 'cpu'.
    seed : int, optional
        Seed for random number generation, by default None.
    Returns
    -------
    Tuple[Int[torch.Tensor, 'batch_size, context_length'], Int[torch.Tensor, 'batch_size, context_length']]
        Tuple containing input and target tensors for the Transformer model.
    """
    
    # Ensure token_ids is a 1D array
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be a 1D array")

    # Calculate the number of possible starting indices
    num_possible_starting_indices = len(token_ids) - context_length
    if num_possible_starting_indices <= 0:
        raise ValueError("token_ids must be longer than context_length")
    
    if seed is not None:
        # Randomly sample starting indices for each batch
        np.random.seed(seed)
    starting_indices = np.random.randint(0, num_possible_starting_indices, size=batch_size)
    x = np.array([token_ids[i:i + context_length] for i in starting_indices])
    y = np.array([token_ids[i + 1:i + context_length + 1] for i in starting_indices])
    # Convert to PyTorch tensors and move to the specified device
    x_tensor = torch.tensor(x, device=device)
    y_tensor = torch.tensor(y, device=device)
    return x_tensor, y_tensor


class TokenizedDataset(Dataset):
    """ TokenizedDataset is a custom dataset to efficiently load tokenized text data for the Transformer model.
    """
    
    def __init__(
        self,
        tokenized_data_path: str,
        batch_size: int,
        context_length: int,
        total_num_steps: int = 100_000,
        device: str = 'cpu',
        seed: int = 42,
    ) -> None:
        """__init__ Read tokenized data into (virtual) memory.
        
        Parameters
        ----------
        tokenized_data_path : str
            Path to the file containing tokenized data.
            The data is either in mmmapped numpy format (.mmap) or a pickle file (.pkl).
        batch_size : int
            Number of samples per batch.
        context_length : int
            Length of the context window for each sample.
        total_num_steps : int, optional
            Total number of steps to run the training for, by default 100_000.
            This is not used in this class but can be useful for other purposes.
        device : str, optional
            Device to load the data onto, by default 'cpu'.
        
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Loading tokenized data from {tokenized_data_path} into memory.")
        
        file_extension = tokenized_data_path.split('.')[-1]
        
        # Load into virtual memory for memory efficiency if the file extension is a supported format.
        self.token_ids = None
        if file_extension == 'npy': # 
            logger.info("Loading tokenized data into virtual memory using numpy's memmap.")
            self.token_ids = np.load(tokenized_data_path, mmap_mode='r')
        elif file_extension == 'pkl':
            import pickle
            logger.info("Loading tokenized data fully into memory using pickle.")
            with open(tokenized_data_path, 'rb') as f:
                self.token_ids = pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported formats are .mmap, .npy, and .pkl.")
        
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        
        # If one doesn't want the final truth, not just yet, one can provide a different seed.
        # But if one is indeed ready to see the answer to the Ultimate Question of Life, the Universe, and Everything,
        # one can set the seed to 42.
        self.seed = seed
        # Randomly sample starting indices for each batch
        np.random.seed(seed)
        self.seeds = list(np.random.choice(range(0, total_num_steps*100), size=total_num_steps*100, replace=False))
        
    def __call__(self, step: int):
        return data_loading(
            token_ids=self.token_ids,
            batch_size=self.batch_size,
            context_length=self.context_length,
            device=self.device,
            seed=self.seeds[step]
        )
        
    def generate_seeds(self, total_num_steps: int = 100_000):
        """generate_seeds Generate seeds for random sampling.
        
        Parameters
        ----------
        total_num_steps : int, optional
            Total number of steps to generate seeds for, by default 100_000.
        
        Returns
        -------
        list
            List of seeds for random sampling.
        """
        self.seeds = list(np.random.choice(range(0, total_num_steps), size=total_num_steps, replace=False))
        
if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    dataset = TokenizedDataset(
        tokenized_data_path='/home/hihi/code/courses/stanford-cs336/Stanford-CS336-LLM-assignment1/data/tokenized/TinyStoriesV2-GPT4-train_tokenized.npy',
        batch_size=32,
        context_length=128,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    from gpt2.tokenizer.bpe import BPE_Tokenizer
    tokenizer = BPE_Tokenizer(verbose=40)
    tokenizer.from_files(
        vocab_path="data/tokenizer/TinyStoriesV2-GPT4-vocab.pkl",
        merges_path="data/tokenizer/TinyStoriesV2-GPT4-merges.pkl",
        special_tokens=["<|endoftext|>"]
    )
    
    for i in range(5):
        token_ids, _ = dataset(i)

        decoded = tokenizer.decode(token_ids[0].cpu().numpy().tolist())
        print(decoded)
