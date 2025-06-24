"""
Utility functions for the Transformer model.
"""

import torch
from torch.nn import Module
from torch.optim import Optimizer
import os
from typing import BinaryIO, IO

def save_checkpoint(
    model: Module,
    optimizer: Optimizer,
    iteration: int,
    checkpoint_path: str | os.PathLike | BinaryIO | IO[bytes]
) -> None:
    """
    Save the model and optimizer state to a checkpoint file.

    Parameters
    ----------
    model : Module
        The PyTorch model to save.
    optimizer : Optimizer
        The optimizer state to save.
    iteration : int
        The current iteration number.
    out : str | os.PathLike | BinaryIO | IO[bytes]
        The output file path or file-like object to save the checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    
    if isinstance(checkpoint_path, (str, os.PathLike)):
        torch.save(checkpoint, checkpoint_path)
    elif isinstance(checkpoint_path, (BinaryIO, IO)):
        torch.save(checkpoint, checkpoint_path)
    else:
        raise TypeError("Output must be a file path or a file-like object.")

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: Module,
    optimizer: Optimizer
) -> int:
    """
    Load the model and optimizer state from a checkpoint file.
    Parameters
    ----------
    src : str | os.PathLike | BinaryIO | IO[bytes]
        The source file path or file-like object to load the checkpoint from.
    model : Module
        The PyTorch model to load the state into.
    optimizer : Optimizer
        The optimizer to load the state into.
    Returns
    -------
    int
        The iteration number from the checkpoint.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']