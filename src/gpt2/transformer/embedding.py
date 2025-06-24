"""
Install embedding module that's based on nn.Module."""

import torch
from torch import nn
from torch import Tensor
import math
from jaxtyping import Float, Int
import einops

class Embedding(nn.Module):
    """Embedding module that maps token IDs to dense vectors.
    
    This module is a simple wrapper around a learnable weight matrix that
    serves as an embedding lookup table. It is initialized with a normal
    distribution truncated to prevent extreme values, which is a common
    practice in deep learning to ensure stable training.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None, **kwargs):
        """__init_ Construct an embedding module.

        Parameters
        ----------
        num_embeddings : int
            Size of the vocab
        embedding_dim : int
            Dimension of the embedding vectors, i.e., d_model
        device : _type_, optional
            Device to run on, by default None (CPU)
        dtype : _type_, optional
            Type of data, by default torch.fp32
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(torch.zeros(num_embeddings, embedding_dim, device=device, dtype=dtype))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initializes the parameters of the linear layer.
        The weight is initialized using a variantion of Xavier initialization with
        Normal(μ = 0, σ^2 = 2 / (d_out + d_in)) and truncated to [-3 * σ, 3 * σ] to prevent extreme values.
        Embedding layers are initialized with a Normal distribution truncated to [-3, 3]
            Normal(μ = 0, σ^2 = 1) truncated to [-3, 3]
        RMSNorm's gamma is initialized to 1.0.
        """
        
        stdv = math.sqrt(2.0 / (self.num_embeddings + self.embedding_dim))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=stdv * stdv, a=-3 * stdv, b=3 * stdv)
    
    def forward(self, token_ids: Int[torch.Tensor, 'batch_size seq_len']) -> Float[torch.Tensor, 'batch_size seq_len d_model']:
        """forward 

        Parameters
        ----------
        token_ids : Int[torch.Tensor, 'batch_size seq_len']
            The token IDs to be embedded, shape (batch_size, seq_len).

        Returns
        -------
        Float[torch.Tensor, 'batch_size seq_len d_model']
            The embedded vectors for the input token IDs, shape (batch_size, seq_len, d_model).
        """
        embedded = self.weight[token_ids]
        return embedded
    
    def extra_repr(self):
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, device={self.weight.device}, dtype={self.weight.dtype}"
        
        
import torch
import torch.nn as nn
from typing import Optional # For Python < 3.10 type hints

# Assuming Float and other jaxtyping aliases are defined elsewhere if used
# from jaxtyping import Float

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: Optional[torch.device] = None, # Made Optional for clarity
        dtype: Optional[torch.dtype] = None,   # Made Optional for clarity
        *args, **kwargs
    ) -> None: # __init__ returns None
        """
        Initializes the Rotary Positional Embedding module.

        Parameters
        ----------
        theta : float
            The base period for the rotary embeddings.
        d_k : int
            Dimension of the query and key vectors.
        max_seq_len : int
            Maximum sequence length that this module will support.
        device : Optional[torch.device], optional
            Device to store the precomputed embeddings on, by default None (uses default device).
        dtype : Optional[torch.dtype], optional
            Data type for the precomputed embeddings, by default None (uses default tensor dtype).
        """
        super().__init__(*args, **kwargs)

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Corrected frequency calculation
        # Exponents: (2*i / d_k) where i is the index of the pair
        # torch.arange(0, d_k, 2) -> [0, 2, 4, ..., d_k-2]
        arange_tensor = torch.arange(0, d_k, 2, device=device, dtype=dtype)
        freqs = 1.0 / (theta ** (arange_tensor / d_k)) # shape: (d_k // 2)

        positions = torch.arange(max_seq_len, device=device, dtype=dtype) # shape: (max_seq_len)

        # Outer product to get (max_seq_len, d_k // 2)
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)

        cos_values = torch.cos(angles)
        sin_values = torch.sin(angles)

        # If the initial dtype was specific (e.g. bfloat16), cast back after math ops
        if dtype is not None:
            cos_values = cos_values.to(dtype)
            sin_values = sin_values.to(dtype)

        # Use persistent=False to avoid saving in state_dict, as these are not trainable parameters
        self.register_buffer('cos_vals_cached', cos_values, persistent=False)
        self.register_buffer('sin_vals_cached', sin_values, persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, '... seq_len d_k'],
        token_positions: Optional[Float[torch.Tensor, '... seq_len']]
    ) -> torch.Tensor: # Float[torch.Tensor, '... seq_len d_k']
        """
        Applies Rotary Positional Embeddings to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (e.g., query or key). Expected shape: (..., current_seq_len, d_k).
        token_positions : Optional[torch.Tensor], optional
            Positions of the tokens in the sequence, shape (..., seq_len).
            If None, assumes x represents a contiguous sequence starting from 0 up to current_seq_len - 1.
            Defaults to None.

        Returns
        -------
        torch.Tensor
            The input tensor with RoPE applied, shape (..., seq_len, d_k).
        """
        indices = token_positions.to(device=self.cos_vals_cached.device, dtype=torch.long)

        # Ensure indices are within bounds of cached values
        if torch.max(indices) >= self.max_seq_len or torch.min(indices) < 0:
            raise ValueError(
                f"Token positions are out of the precomputed range [0, {self.max_seq_len-1}]. "
                f"Max position: {torch.max(indices)}, Min position: {torch.min(indices)}"
            )

        # Gather the cosine and sine values for the specific token positions
        # self.cos_vals_cached has shape (max_seq_len, d_k // 2)
        # indices shape e.g., (batch_size, current_seq_len) or (current_seq_len,)
        # cos_selected/sin_selected shape will be (..., current_seq_len, d_k // 2)
        cos_selected = self.cos_vals_cached[indices].to(x.dtype)
        sin_selected = self.sin_vals_cached[indices].to(x.dtype)

        # The Rotary Positional Embedding applies a rotation to pairs of input features.
        # For each pair of features (q_{2j}, q_{2j+1}) in the input vector q,
        # and a corresponding rotation angle theta_j (which depends on position and pair index),
        # the transformation is equivalent to multiplying by the 2D rotation matrix:
        # R_j = [[cos(theta_j), -sin(theta_j)],
        #        [sin(theta_j),  cos(theta_j)]]
        #
        # So, the rotated pair (q'_{2j}, q'_{2j+1}) is:
        # q'_{2j}   = q_{2j} * cos(theta_j) - q_{2j+1} * sin(theta_j)
        # q'_{2j+1} = q_{2j} * sin(theta_j) + q_{2j+1} * cos(theta_j)
        #
        # If we were to represent this for the entire d_k dimensional vector q,
        # it would be a block-diagonal matrix R where each 2x2 block on the diagonal is R_j.
        # R = [[cos(theta_0), -sin(theta_0), 0,            0, ...],
        #      [sin(theta_0),  cos(theta_0), 0,            0, ...],
        #      [0,            0,             cos(theta_1), -sin(theta_1), ...],
        #      [0,            0,             sin(theta_1),  cos(theta_1), ...],
        #                                  ...
        #                                  ...
        #                                  ...
        #      [0,            0,             0,            0, ... cos(theta_{d_k//2-1}), -sin(theta_{d_k//2-1})],
        #      [0,            0,             0,            0, ... sin(theta_{d_k//2-1}),  cos(theta_{d_k//2-1})]]
        #
        # Multiplying by this full matrix R (q' = R * q) is computationally inefficient.
        # Instead, we can achieve the same result more efficiently:
        # 1. Split the input tensor q into its even-indexed components (q_even = [q_0, q_2, ...])
        #    and odd-indexed components (q_odd = [q_1, q_3, ...]).
        # 2. The `cos_selected` variable will hold [cos(theta_0), cos(theta_1), ...]
        #    and `sin_selected` will hold [sin(theta_0), sin(theta_1), ...]
        #
        # Then the operations:
        #   rotated_even = q_even * cos_selected - q_odd * sin_selected
        #   rotated_odd  = q_even * sin_selected + q_odd * cos_selected
        #
        # directly compute all the q'_{2j} components (in `rotated_even`) and all the q'_{2j+1}
        # components (in `rotated_odd`) simultaneously, achieving the same result as the
        # block-diagonal matrix multiplication without forming the matrix.

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Apply RoPE formula:
        # x_rotated_even = x_even * cos - x_odd * sin
        # x_rotated_odd  = x_even * sin + x_odd * cos
        rotated_even = x_even * cos_selected - x_odd * sin_selected
        rotated_odd  = x_even * sin_selected + x_odd * cos_selected

        # Interleave the rotated halves back
        # Stack along a new dimension, then flatten that dimension with the previous one
        # result shape: (..., current_seq_len, d_k // 2, 2) -> (..., current_seq_len, d_k)
        x_rotated = torch.stack((rotated_even, rotated_odd), dim=-1).flatten(start_dim=-2)

        return x_rotated
    
    def extra_repr(self) -> str:
        return (f"theta={self.theta}, d_k={self.d_k}, max_seq_len={self.max_seq_len}, "
                f"device={self.cos_vals_cached.device}, dtype={self.cos_vals_cached.dtype}")