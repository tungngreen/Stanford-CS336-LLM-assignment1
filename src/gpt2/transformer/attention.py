"""
Scaled Dot-Product Attention mechanism for transformer models.
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from torch import Tensor
from jaxtyping import Float, Int
from typing import Optional
from einops import einsum, rearrange
from math import ceil, sqrt

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

class FlashAttentionTorchFunctionClass(Function):
    """
    Custom autograd function for Flash Attention.
    """
    
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx,
                Q: Float[Tensor, " ... queries d_k"],
                K: Float[Tensor, " ... keys d_k"],
                V: Float[Tensor, " ... values d_v"],
                is_causal: bool) -> Float[Tensor, " ... queries d_v"]:
        """forward method for Flash Attention.
        This method implements the Flash Attention algorithm, which is a memory-efficient
        and fast attention mechanism that computes attention in tiles.
        
        The forward pass computes the attention output and saves necessary tensors for the backward pass.
        
        The algorithm works by:
        1. Splitting the query, key, and value tensors into smaller tiles.
        2. Computing attention scores for each tile.
        3. Applying softmax to the scores to get attention weights.
        4. Combining the weighted values to produce the final output.
        5. Saving the log-sum-exp tensor for the backward pass.
        
        This implementation is designed to be efficient in terms of memory usage and computation speed,
        especially for long sequences, by processing the attention in smaller chunks (tiles).

        Parameters
        -------
        ctx : Float[Tensor, " ... queries d_v"]
            Context object to save tensors for backward pass.
        Q : Float[Tensor, " ... queries d_k"]
            Query tensor of shape (..., queries, d_k).
        K : Float[Tensor, " ... keys d_k"]
            Key tensor of shape (..., keys, d_k).
        V : Float[Tensor, " ... values d_v"]
            Value tensor of shape (..., values, d_v).
        is_causal : bool
            Whether the attention is causal or not.
            
        Returns
        -------
        Float[Tensor, " ... queries d_v"]
            Output tensor after applying Flash Attention.
            
        """
        # Hidden dimensions
        d_k = Q.shape[-1]
        d_v = V.shape[-1]
        
        # The size of tiles 16-64
        Bq = 32
        Bk = 32

        # Number of tiles in Q and K
        Tq = ceil(Q.shape[-2] / Bq)
        Tk = ceil(K.shape[-2] / Bk)
        
        # Split Q into tiles of size Bq, shape will be (..., Tq, Bq, d_k)
        Q_tiles = rearrange(Q, '... (tq Bq) d_k -> ... tq Bq d_k', tq=Tq, Bq=Bq)
        # Split K into tiles of size Bk
        K_tiles = rearrange(K, '... (tk Bk) d_k -> ... tk Bk d_k', tk=Tk, Bk=Bk)
        # Split V into tiles of size Bk
        V_tiles = rearrange(V, "... (tk Bk) d_v -> ... tk Bk d_v", tk=Tk, Bk=Bk)
        
        # Initialize lists to hold the output tensors for each tile
        # O will hold the softmax-weighted output for each query tile, L will hold the log-sum-exp values
        # O_i will be of shape (..., Bq, d_v) and L_i will be of shape (..., Bq)
        out_seq = []
        L = []
        for i in range(1, Tq+1):
            # Load Q_i from global memory, shape will be (..., Bq, d_k)
            Q_i = Q_tiles[..., i-1, :, :]

            # Initialize O_i as a list of tensors to hold the output for each tile
            # O_i will hold the output for the current query tile, shape will be (..., Bq, d_v)
            # Initialize with zeros, this is important because we will accumulate the output for each tile
            # This is similar to initializing a running sum, where we will add the contributions from each key tile.
            # Q_i_0 is initialized as 0.
            O_i = []
            O_i.append(torch.zeros(Q_i.shape[:-1] + (d_v,), device=Q_i.device, dtype=Q_i.dtype))
            
            # Inititalize l_i, running proxies for the softmax denominator and will be updated using the unnormalized
            # softmax values
            # When we finally write the ouput, we will need to finish normalizing the output by dividing by l_i_Tk,
            # which is the final value of l_i after processing all Tk tiles.
            # Shape will be (..., Bq)
            # l_i will hold the log-sum-exp values for the current query tile, shape will be (..., Bq)
            # l_i_0 is initialized as 0.
            l_i = []
            l_i.append(torch.zeros_like(Q_i[..., 0]))

            # Initialize m_i, row-wise running maximum until the current K-tile as we move right-ward through the tiles
            # of K^T
            # We will update m_i_j each new row-wise tile of S as we move wright-ward through the K-tiles.
            # Using the maximum, we can compute the
            # unnormalized softmax values. Shape Bq with original value of -inf
            # m_i will hold the row-wise running maximum for the current query tile, shape will be (..., Bq)
            m_i = []
            m_i.append(torch.full(l_i[0].shape, float('-inf'), device=Q_i.device, dtype=Q_i.dtype))

            for j in range(1, Tk+1):
                # Load K_j and V_j from global memory, shape will be (..., Bk, d_k) and (..., Bk, d_v)
                K_j = K_tiles[..., j-1, :, :]
                V_j = V_tiles[..., j-1, :, :]
                
                # Compute presoftmax attention scores, shape will be (..., Bq, Bk)
                S_i_j = torch.einsum('... q d, ... k d -> ... q k', Q_i, K_j) / sqrt(d_k)

                # Compute m_i_j = max(S_i_j, m_i_j-1), shape will be Bq
                # Each element of m_i[j] will be the maximum of the corresponding ROW in S_i_j and the previous maximum
                # m_i[j-1]
                m_i.append(torch.maximum(m_i[j-1], torch.max(S_i_j, dim=-1).values))

                # Compute the unnormalized softmax values, shape will be (..., Bq, Bk)
                S_i_j_unnorm = torch.exp(S_i_j - rearrange(m_i[j], '... -> ... 1'))
                
                # Compute the scale factor for the log-sum-exp, shape will be (..., Bq)
                # Because we previously shifted the S_i_j by m_i[j-1] so the previous output sequence and L are on a
                # different scale, we need to compute the scale factor for the current tile.
                # For instance, S1 = [4.0, 6.0], max is 6.0, exponential is exp(x-6.0). then the new max is 8.0
                # We must scale the old values down by exp(6.0 - 8.0)
                scale = torch.exp(m_i[j-1] - m_i[j])
                
                # Compute the sum of the unnormalized softmax as we move right-ward through the K tiles,
                # shape will be (..., Bq)
                S_i_rowsum = torch.sum(S_i_j_unnorm, dim=-1)
                
                # Update the log-sum-exp value for the current tile, shape will be (..., Bq)
                l_i.append(S_i_rowsum + l_i[j-1] * scale)
                
                # Rearrange the scale to match the shape of S_i_j_unnorm for broadcasting
                # This is necessary to ensure that the scale factor can be applied correctly to the output of the
                # unnormalized softmax.
                # The shape will be (..., Bq, 1) so it can be broadcasted to the shape of S_i_j_unnorm
                scale = rearrange(scale, '... -> ... 1')
            
                # rescale the previous output O_i[j-1] by the scale factor
                O_i_prev_scale = scale * O_i[j-1]
                
                # Compute the attention output for this tile, shape will be (..., Bq, d_v)
                O_i_j = torch.einsum('... q k, ... k d -> ... q d', S_i_j_unnorm, V_j) + O_i_prev_scale
                O_i.append(O_i_j)

            # After processing all K tiles, we are at the right end of the K tiles, we have the final output.
            # will be (..., Bq, d_v)
            O_i_Tk_unnormalized = O_i[-1]
            
            # l_i_Tk is the last log-sum-exp value for the current query tile, shape will be (..., Bq)
            l_i_Tk = l_i[-1]

            # Mormalize the output by dividing by the log-sum-exp value for the current tile
            # This is the final step to ensure that the output is properly normalized.
            # The output will be of shape (..., Bq, d_v)
            O_i_Tk_normalized = O_i_Tk_unnormalized / l_i_Tk.unsqueeze(-1)
            
            # Calculate the final log-sum-exp value for the current tile
            # This is the final value that will be used in the backward pass to compute gradients.
            # The shape will be (..., Bq)
            # L_i_out is the final log-sum-exp value for the current query tile, shape will be (..., Bq)
            L_i_out = m_i[-1] + torch.log(l_i_Tk)
            
            # Append the output and log-sum-exp values for the current tile to the lists
            out_seq.append(O_i_Tk_normalized)
            L.append(L_i_out)

        # Concatenate the output tiles along the queries dimension
        out_seq = torch.cat(out_seq, dim=-2)
        L = torch.cat(L, dim=-1)

        # Reshape the output to match the original query shape
        # O = rearrange(O, '... tq Bq d_v -> ... (tq Bq) d_v', tq=Tq, Bq=Bq)
        
        # Save the log-sum-exp tensor for backward pass
        ctx.save_for_backward(L, Q, K, V, out_seq)

        return out_seq


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
        return f'd_model={self.d_model}, num_heads={self.num_heads}, d_k={self.d_k}, d_v={self.d_v}' + \
               f'device={self.Wq.weight.device}, dtype={self.Wq.weight.dtype}'