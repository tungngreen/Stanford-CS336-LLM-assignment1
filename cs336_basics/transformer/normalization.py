"""
Implementation of Layer Normalization with RMS
"""

import torch
import torch.nn as nn
from jaxtyping import Float, Int

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device = None,
        dtype: torch.dtype = None) -> nn.Module:
        """__init__ Initialization a RMSNorm layer

        Parameters
        ----------
        d_model : int
            Hidden dimension of the model
        eps : int
            Epsilon value for numerical stability
        device : torch.device, optional
            Device to store the parameters on, by default None
        dtype : torch.dtype, optional
            Data type of the parameters, by default None

        Returns
        -------
        nn.Module
            Normalized results
        """
        super().__init__()
        
        # The hidden dimension of model
        self.d_model = d_model
        # Epsilon value
        self.eps = eps
        
        # Gain parameters initialized as 1
        """
        - Acting as identity function at Initialization
            - 0 kills gradients, large values may explode.
            - 1 just scale identically.
        - Good for initial stability.
            - Preserving signals.
            - If the flow is good at the beginning then 1 simply maintains that scale.
        - Allowing the Network to Learn Optimal Scaling by maintaining its representational power and find the best scale for each feature.
        """
        self.weight = nn.Parameter(torch.ones(self.d_model, device=device, dtype=dtype))
        
    def forward(self, x: Float[torch.Tensor, 'batch_size seq_len d_model']) -> Float[torch.Tensor, 'batch_size seq_len d_model']:
        """forward Forward pass of the RMSNorm layer
        Parameters
        ----------
        x : Float[torch.Tensor, 'batch_size seq_len d_model']
            Input tensor to be normalized
        Returns
        -------
        Float[torch.Tensor, 'batch_size seq_len d_model']
            Normalized tensor
        """
        
        in_dtype = x.dtype
        
        x = x.to(dtype=torch.float32)
        
        RMS = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        normalized_x = x / RMS
        normalized_x = normalized_x * self.weight
        
        return normalized_x.to(dtype=in_dtype)
        

    
    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}, weight={self.weight.shape}, dtype={self.weight.dtype}, device={self.weight.device}"
        
    