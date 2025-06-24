"""
Implementation of the loss function for the Transformer model.
"""

import torch
import torch.nn as nn
from jaxtyping import Float
from einops import rearrange

def cross_entropy(
    logits: Float[torch.Tensor, 'batch_size vocab_size'],
    targets: Float[torch.Tensor, 'batch_size']
):
    """cross_entropy calculates the cross-entropy loss between logits and targets.

    Parameters
    ----------
    logits : Float[torch.Tensor, &#39;... vocab_size&#39;]
        Logits from the model for the next token prediction, shape (..., vocab_size).
        These logits are the raw, unnormalized scores for each token in the vocabulary.
    targets : Float[torch.Tensor, 'batch_size']
        True class indices (token IDs) for each item in the batch, shape (batch_size).
        
    Returns
    -------
    Float[torch.Tensor, '']
        The mean cross-entropy loss over the batch. This is a scalar value representing the average loss.

    """
    
    # Create a one-hot encoded matrix for targets
    batch_size, vocab_size = logits.shape
    targets_matrix = torch.zeros(batch_size, vocab_size, device=logits.device)
    targets_matrix.scatter_(1, rearrange(targets, 'b -> b 1'), 1.0)  # One-hot encoding
    
    
    # Subtract the maximum logit for numerical stability
    logits_stabilized = logits - torch.max(logits, dim=-1, keepdim=True).values
    
    # Compute the log probabilities
    # Canceling the log of exponential of the numerator
    log_probs = logits_stabilized - torch.log(torch.sum(torch.exp(logits_stabilized), dim=-1, keepdim=True))
    
    # Compute the cross-entropy loss
    # loss = -torch.sum(targets_matrix * log_probs, dim=-1)
    loss = torch.einsum('bv,bv->b', targets_matrix, log_probs)
    loss = rearrange(loss, 'b -> b 1')  # Convert to scalar
    loss_batch = -torch.sum(loss, dim=-1)
    return loss_batch.mean()  # Return the mean loss over the batch
    

    