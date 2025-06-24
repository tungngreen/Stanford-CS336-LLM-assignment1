"""
Implementation of of the pre-transformer block
"""

import torch
from torch import nn
from jaxtyping import Float, Int
from typing import Optional
from gpt2.transformer.linear import Linear
from gpt2.transformer.activation import SiLU, SwigLU
from gpt2.transformer.normalization import RMSNorm
from gpt2.transformer.attention import MultiheadSelfAttention, softmax
from gpt2.transformer.embedding import RotaryPositionalEmbedding, Embedding
from gpt2.tokenizer.bpe import BPE_Tokenizer

class TransformerBlock(nn.Module):
    """
    Transformer Block implementation with multi-head self-attention, feed-forward network, and layer normalization.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
        rope: RotaryPositionalEmbedding = None,
        theta: float = 10000.0,
        max_seq_len: int = 32,
        *args,
        **kwargs):
        """__init__ Initialization of Pre-Norm transformer block

        Parameters
        ----------
        d_model : int
            Dimensionality of Transformer block inputs
        num_heads : int
            Number of heads to use in multihead self-attention
        d_ff : int
            Dimensionality of the position-wise feed-forward inner layer
        """
        super().__init__(*args, **kwargs)
        
        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )
        
        if rope is not None:
            self.rope = rope
        else:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=d_model // num_heads,
                max_seq_len=max_seq_len,
                device=device,
                dtype=dtype
            )
        
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=self.rope,
            device=device,
            dtype=dtype
        )
        
        self.ln2  = RMSNorm(
            d_model=d_model,
            dtype=dtype,
            device=device
        )
        
        self.ffn = SwigLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )
        
    def forward(self, in_features: Float[torch.Tensor, 'batch_size seq_len d_model'], token_positions: Int[torch.Tensor, 'batch_size seq_len'] = None) -> Float[torch.Tensor, 'batch_size seq_len d_model']:

        norm_1 = self.ln1(in_features)
        multihead_self_attention = self.attn(norm_1, token_positions)

        add_1 = in_features + multihead_self_attention
        
        norm_2 = self.ln2(add_1)
        ffn = self.ffn(norm_2)
        
        add_2 = add_1 + ffn
        
        return add_2
    
    def extra_repr(self) -> str:
        """
        Returns a string representation of the TransformerBlock parameters.
        
        Returns
        -------
        str
            A string representation of the TransformerBlock parameters.
        """
        return f'd_model={self.attn.d_model}, num_heads={self.attn.num_heads}, d_ff={self.ffn.d_ff}, device={self.attn.Wq.weight.device}, dtype={self.attn.Wq.weight.dtype}'
    
    
class Transformer(nn.Module):
    """
    Transformer model implementation with multiple transformer blocks.
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
        *args, **kwargs):
        """__init__ Initialization of Transformer model
        
        Parameters
        ----------
        vocab_size: int, optional
            The size of the vocabulary, necessary for determining the dimensionality of the token
            embedding matrix., by default 50000
        context_length (aka max_seq_len): int, optional
            The maximum context length, necessary for determining the dimensionality of
            the position embedding matrix, by default 32
        d_model : int, optional
            Dimensionality of Transformer block inputs
        num_layers : int, optional
            Number of transformer blocks, by default 6
        num_heads : int, optional
            Number of heads to use in multihead self-attention, by default 8
        d_ff : int, optional
            Dimensionality of the position-wise feed-forward inner layer, by default 2048
        num_layers : int, optional
            Number of transformer blocks, by default 6
        rope : RotaryPositionalEmbedding, optional
            Rotary positional embedding to use, by default None (creates a new one).
        theta : float, optional
            Theta value for rotary positional embedding, by default 10000.0
        device : torch.device, optional
            Device to run the model on, by default None (uses default device).
        dtype : torch.dtype, optional           
            Data type of the model parameters, by default None (uses default dtype).

        """
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.device = device
        self.dtype = dtype
        
        self.token_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )
        
        self.rope = rope if rope is not None else RotaryPositionalEmbedding(
            theta=theta,
            d_k=d_model // num_heads,
            max_seq_len=context_length,
            device=device,
            dtype=dtype
        )
        
        for layer in range(num_layers):
            setattr(self, f'transformer_block_{layer}', TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                device=device,
                dtype=dtype,
                rope=self.rope
            ))
        
        self.norm = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )
        self.linear = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype
        )
        
    def forward(
        self,
        in_features: Float[torch.Tensor, 'batch_size seq_len'],
        token_positions: Int[torch.Tensor, '... seq_len'] = None
    ) -> Float[torch.Tensor, 'batch_size seq_len d_model']:
        
        """
        Forward pass of the Transformer model.
        
        Parameters
        ----------
        in_features : Float[torch.Tensor, 'batch_size seq_len']
            Input tensor of shape (batch_size, seq_len) containing several batches, each with a sequence of token IDs.
        token_positions : Int[torch.Tensor, '... seq_len'], optional
            Token positions for rotary positional embedding, by default None.
        
        Returns
        -------
        Float[torch.Tensor, 'batch_size seq_len d_model']
            Output tensor after passing through all transformer blocks and normalization.
        """
        
        x = self.token_embedding(in_features)
        
        for layer in range(self.num_layers):
            transformer_block = getattr(self, f'transformer_block_{layer}')
            x = transformer_block(x, token_positions)
        
        x = self.norm(x)
        x = self.linear(x)
        
        return x
    
    def extra_repr(self) -> str:
        """
        Returns a string representation of the Transformer parameters.
        
        Returns
        -------
        str
            A string representation of the Transformer parameters.
        """
        return f'd_model={self.d_model}, num_heads={self.num_heads}, d_ff={self.d_ff}, num_layers={self.num_layers}, vocab_size={self.vocab_size}, context_length={self.context_length}, device={self.device}, dtype={self.dtype}'