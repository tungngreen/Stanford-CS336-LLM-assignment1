"""
Utility functions for the Transformer model.
"""

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
import os
from typing import BinaryIO, IO
from jaxtyping import Float

from einops import rearrange

from gpt2.transformer.optimizers import AdamW, gradient_clipping
from gpt2.transformer.loss import cross_entropy
from gpt2.transformer.model import Transformer
from torch.profiler import record_function


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

def run_step(
    model: Transformer,
    inputs: Float[Tensor, 'b s'],
    outputs: Float[Tensor, 'b s'],
    optimizer: AdamW,
    enable_backward: bool = True,
    **kwargs
):
    """Run a single training step.
    
    Parameters
    ----------
    model : Transformer
        The transformer model to train.
    inputs : Float[Tensor, 'b s']
        The input sequences for the model, where `b` is the batch size and `s` is the sequence length.
    outputs : Float[Tensor, 'b s']
        The target sequences for the model, where `b` is the batch size and `s` is the sequence length.
    optimizer : AdamW
        The optimizer to use for training.
    enable_backward : bool, optional
        Whether to enable the backward pass and optimization step. If False, only the forward pass is run.
        Defaults to True.
    **kwargs : dict
        Additional keyword arguments that can include:
        - token_positions: Optional tensor of token positions for the model.
        - max_l2_norm: Maximum L2 norm for gradient clipping.
    
    Returns
    -------
    loss : torch.Tensor
        The loss value for the training step.
    """
    
    lr_t = kwargs.get("learning_rate")
    
    token_positions = kwargs.get("token_positions", None)
    
    with record_function("forward_pass"):
        logits = model(inputs, token_positions=token_positions)
    
    if enable_backward:
        with record_function("backward_pass"):
            # Compute the loss
            logits_2d = rearrange(logits, 'b s v -> (b s) v')
            output_batched_seqs_1d = rearrange(outputs, 'b s -> (b s)')
            loss = cross_entropy(logits_2d, output_batched_seqs_1d)
            loss.backward()
        
        with record_function("optimizer"):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_t
            # Perform the optimization step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
        gradient_clipping(
            parameters=model.parameters(),
            max_l2_norm=kwargs.get("max_l2_norm", 1.0)
        )
            
        return loss
    
    return None