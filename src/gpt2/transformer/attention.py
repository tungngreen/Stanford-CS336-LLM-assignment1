"""
Scaled Dot-Product Attention mechanism for transformer models.
"""

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from typing import Optional
from einops import einsum, rearrange

from gpt2.transformer.linear import Linear
from gpt2.transformer.embedding import RotaryPositionalEmbedding

def softmax(
    in_features: Float[torch.Tensor, '... d_model'],
    dim: int = -1
) -> Float[torch.Tensor, '... d_model']:
    """
    Computes the softmax of the input tensor along the specified dimension.

    Parameters
    ----------
    x : Float[torch.Tensor, '... d_model']
        Input tensor to calculate softmax
    dim : int, optional
        The dimension along which to compute the softmax, by default -2 (last dimension).

    Returns
    -------
    Float[torch.Tensor, '... d_model']
        Output tensor after applying softmax.
    """
    # Since exp(v_i) can become very large, we subtract the maximum value in the tensor
    # to prevent overflow and ensure numerical stability.
    # This is a common technique to stabilize softmax calculations.
    # Subtract the maximum value along the specified dimension for numerical stability
    # shape of max_val will be (..., 1) if dim=-1
    max_val = torch.max(in_features, dim=dim, keepdim=True).values

    # shaped as (..., d_model) after subtraction
    # This stabilizes the input to the exponential function
    in_features_stabilized = in_features - max_val
    
    # Compute the exponentials of the stabilized input
    # shaped as (..., d_model)
    exp_x = torch.exp(in_features_stabilized)
    
    # Sum the exponentiated values along the specified dimension
    # shaped as (..., 1) if dim=-1
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    
    # Divide to get the softmax probabilities
    # shaped as (..., d_model)
    sm = exp_x / sum_exp_x

    return sm


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Computes the scaled dot-product attention.

    Parameters
    ----------
    Q : Float[Tensor, " ... queries d_k"]
        Query tensor of shape (..., queries, d_k).
    K : Float[Tensor, " ... keys d_k"]
        Key tensor of shape (..., keys, d_k).
    V : Float[Tensor, " ... values d_v"]
        Value tensor of shape (..., values, d_v).
        keys = values for self-attention.
    mask : Float[Tensor, " ... queries keys"] | None, optional
        Optional mask tensor to apply to the attention scores.

    Returns
    -------
    Float[Tensor, " ... queries d_v"]
        Output tensor after applying scaled dot-product attention.
    """
    
    """
    The "What" vs. the "How Much"
        - Queries (Q): Represents the current word or token that is "looking" for information. 
            It's asking a question: "What other parts of the input are relevant to me?"
        - Keys (K): Represents all the words in the input sequence that can be "looked at."
            They provide a "label" or "identifier" for each word's content. 
            The interaction between Q and K determines the attention scores.
        - Values (V): Also represents all the words in the input sequence. 
            Crucially, the Value matrix contains the actual information or 
            meaning we want to extract from each word.
        - Probabilities (from softmax): These are the "how much" a query should pay
            attention to each key. They are the attention weights.
    """
    
    d_k = Q.shape[-1]
    # Calculate the attention scores using the matrix multiplication of Q and K.
    # The scores indicate how much focus each query should have on each key.
    # scores shape: (..., queries, keys)
    scores = torch.einsum('... q d, ... k d -> ... q k', Q, K) / (d_k ** 0.5)
    
    # Apply mask to scores (logits) BEFORE softmax
    if mask is not None:
        # Use a large negative number for masked positions.
        # This ensures that after softmax, these positions will have probabilities close to 0.
        scores = torch.masked_fill(scores, mask==0, float('-inf'))
        
    # Apply softmax to get attention weights
    # Softmax is applied along the 'keys' dimension (-1),
    # It normalizes them into a probability distribution, ensuring all the attention weights
    # for a given query sum to 1. 
    # These probabilities, often called attention weights, determine the proportion of each
    # value vector that should be considered for the output.
    # shaped as (..., queries, keys)
    attn_weights = softmax(scores, dim=-1)
    
    # Multiply the attention weights with the value tensor V.
    # This step combines the values based on the attention weights.
    # if the value is the same as the key, then this is self-attention.
    # the output will have shape (..., queries, d_v)
    
    # By doing this, we a creating a new vector for each query that is a blend of the meanings
    # of the other tokens in the sequence, weighted by how relevant they are to the query.
    #   - The output is no longer just the original meaning of the query token in isolation.
    #   - Instead, it is contextualized with respect to the other tokens in the sequence.
    output = torch.einsum('... q k, ... k d -> ... q d', attn_weights, V)
    
    return output

class MultiheadSelfAttention(nn.Module):
    """MultiheadSelfAttention 
    Instead of calculating attention just once, we do it multiple times in parallel and then combine the results.
    Each parallel instance is called an "attention head."

    Think of it like forming a committee of experts. Rather than asking one general expert (single-head attention) 
    to analyze a sentence, you ask a committee of, say, eight specialists (multi-head attention). Each specialist 
    might focus on a different aspect of the sentence.
    
    "The cat sat on the mat." A multi-head attention mechanism might learn:

    Head 1 (Syntactic Focus): Might learn to connect verbs to their subjects. When processing "sat," its
        probabilities would be highest for "cat." It answers the "who did the action?" question.
    Head 2 (Positional Focus): Might learn to focus on adjacent words. When processing "sat," it might give high
        probabilities to "cat" and "on."
    Head 3 (Semantic/Long-Range Focus): In a longer sentence like "The cat, which was old and tired, sat on the mat,"
        this head might learn to link "cat" directly to "mat," skipping over the descriptive clause. It might be
        looking for "agent-location" relationships.
    Head 4 (Pronoun Resolution): In "The cat saw the dog, and it ran away," a head could learn to link "it" back to
        "dog" (or "cat," depending on context), resolving the pronoun's antecedent.

    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        *args, **kwargs
    ) -> None:
        """
        Initializes the Multihead Self-Attention module.
        Parameters
        ----------
        d_model : int
            Dimension of the model (input and output).
        num_heads : int
            Number of attention heads.
        device : Optional[torch.device], optional
            Device to run the module on, by default None (uses default device).
        dtype : Optional[torch.dtype], optional
            Data type of the weights, by default None (uses default tensor dtype).
        """
        super().__init__(*args, **kwargs)
        
        d_k = d_model // num_heads
        d_v = d_k
        
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        
        self.Wq = Linear(in_features=d_model, out_features=num_heads * d_k, device=device, dtype=dtype)
        self.Wk = Linear(in_features=d_model, out_features=num_heads * d_k, device=device, dtype=dtype)
        self.Wv = Linear(in_features=d_model, out_features=d_v * num_heads, device=device, dtype=dtype)
        self.Wo = Linear(in_features=d_v * num_heads, out_features=d_model, device=device, dtype=dtype)
        
        self.rope = rope
        
    def forward(
        self,
        in_features: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        """
        Forward pass of the Multihead Self-Attention module.
        
        Parameters
        ----------
        in_embeddings : Float[Tensor, "... seq_len d_model"]
            Input embeddings of shape (..., seq, d_model).
        
        Returns
        -------
        Float[Tensor, "... seq_len d_model"]
            Output tensor after applying multihead self-attention.
        """
        *outter_dims, seq_len, _ = in_features.shape
        
        # Project inputs to query, key, value spaces
        # Q and K will have shape (..., seq_len, d_k)
        # V will have shape (..., seq_len, d_v)
        # Actually its num_heads * d_k and num_heads * d_v because we have multiple heads
        Q = einsum(in_features, self.Wq.weight, '... seq_len d_model, d_k d_model -> ... seq_len d_k')
        # Q = self.Wq(in_features)
        K = einsum(in_features, self.Wk.weight, '... seq_len d_model, d_k d_model -> ... seq_len d_k')
        # K = self.Wk(in_features)
        V = einsum(in_features, self.Wv.weight, '... seq_len d_model, d_v d_model -> ... seq_len d_v')
        # V = self.Wv(in_features)
        # print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
        
        # Bring the head dimension to the front to have a similar effect as Concat(head_1, head_2, ..., head_n)
        # with head_i = (Q_i, K_i, V_i)
        # This is to prepare for the multi-head attention mechanism.
        Q = rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        # Q = Q.view(*outter_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K = rearrange(K, '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', num_heads=self.num_heads)
        # K = K.view(*outter_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = rearrange(V, '... seq_len (num_heads d_v) -> ... num_heads seq_len d_v', num_heads=self.num_heads)
        # V = V.view(*outter_dims, seq_len, self.num_heads, self.d_v).transpose(-3, -2)
        # print(f"Q shape after view: {Q.shape}, K shape after view: {K.shape}, V shape after view: {V.shape}")
        
        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        

        # Generate causal mask for self-attention by using a lower triangular matrix
        # This mask ensures that each position can only attend to previous positions and itself.
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device), diagonal=0).bool()

        # Compute scaled dot-product attention
        # Shape of output will be (..., num_heads, seq_len, d_v)
        attn_output = scaled_dot_product_attention(Q=Q, K=K, V=V, mask=causal_mask)
        
        # Concatenate heads and project back to d_model
        attn_output = rearrange(attn_output, '... num_heads seq_len d_v -> ... seq_len (num_heads d_v)')
        
        output = einsum(attn_output, self.Wo.weight, '... seq_len hd_v, d_model hd_v -> ... seq_len d_model')
        
        return output
    
    def extra_repr(self) -> str:
        """
        Returns a string representation of the MultiheadSelfAttention parameters.
        
        Returns
        -------
        str
            A string representation of the MultiheadSelfAttention parameters.
        """
        return f'd_model={self.d_model}, num_heads={self.num_heads}, d_k={self.d_k}, d_v={self.d_v}, device={self.Wq.weight.device}, dtype={self.Wq.weight.dtype}'