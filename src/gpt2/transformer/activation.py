""" Activation functions for Transformer models.
"""
import torch
from torch import Tensor
from jaxtyping import Float

from gpt2.transformer.linear import Linear
import math

class SiLU(torch.nn.Module):
    def __init__(self, device: torch.device = None, dtype: torch.dtype = None, *args, **kwargs) -> torch.nn.Module:
        """__init__ Implementation of the SiLU (Sigmoid Linear Unit) activation function.

        Parameters
        ----------
        device : torch.device, optional
            Device to run on, by default None (CPU)
        dtype : torch.dtype, optional
            Data type of the weights, by default None (float32)

        Returns
        -------
        torch.nn.Module
            SiLU activation module
        """
        super().__init__(*args, **kwargs)
        
    def forward(self, x: Float[Tensor, '... d_model']) -> Float[Tensor, '... d_model']:
        """Forward pass of the SiLU activation function.
        Parameters
        ----------
        x : Float[Tensor, '... d_model']
            Input tensor of shape (..., d_model)
        Returns
        -------
        Float[Tensor, '... d_model']
            Output tensor of the same shape as input, after applying SiLU activation
        """
        # SiLU is defined as x * sigmoid(x)
        # where sigmoid(x) = 1 / (1 + exp(-x))
        # This can be computed as x * torch.sigmoid(x)
        # torch.sigmoid has shape (..., d_model) as well
        return x * torch.sigmoid(x)

class SwigLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
        *args, **kwargs) -> torch.nn.Module:
        """__init__ _summary_

        _extended_summary_

        Parameters
        ----------
        d_model : int
            _description_

        Returns
        -------
        torch.nn.Module
            _description_
        """
        super().__init__(*args, **kwargs)
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # The first projection's gated path
        # Remember that the first projection is d_model -> d_ff
        # The linear layer will initialize the weights as (d_ff, d_model)
        # Because of row-major order, the weight matrix is of shape (d_ff, d_model)
        # or (out_features, in_features)
        # shped as (d_model, d_ff)
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        # LInear path, shaped as (d_model, d_ff)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        # second projection
        # shaped as (d_ff, d_model)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)

        self.activation = SiLU(device=device, dtype=dtype)
        self.reset_parameters()
        
        
    def reset_parameters(self):
        """
        Kaiming Uniform initialization for weights
        For ReLU-like activations (like GELU, Swish/SiLU), `a=0` for the gain is typical.
        This is `math.sqrt(2.0 / fan_in)` or `math.sqrt(2.0 / (fan_in + fan_out))` based on mode.

        For w1 (d_model -> d_ff)
        Using `fan_in` mode, which is common for Kaiming initialization
        gain = nn.init.calculate_gain('relu') # Or 'leaky_relu' or specific value
        For standard Kaiming, it's often `math.sqrt(2 / fan_in)`
        
        Weights
        Kaiming Uniform (He Uniform)
        Formula: U(-bound, bound) where bound = sqrt(6 / fan_in)
        """        
        # Kaiming Uniform initialization for w1
        torch.nn.init.kaiming_uniform_(self.w1.weight, a=0, mode='fan_in', nonlinearity='relu')
        # Kaiming Uniform initialization for w2
        torch.nn.init.kaiming_uniform_(self.w2.weight, a=0, mode='fan_in', nonlinearity='relu')
        # Kaiming Uniform initialization for w3
        torch.nn.init.kaiming_uniform_(self.w3.weight, a=0, mode='fan_in', nonlinearity='relu')

    # def forward(self, x: Float[Tensor, '... d_model']) -> Float[Tensor, '... d_model']:
    #     """forward Implementation of FFN SwigLu forward using einsum

    #     Parameters
    #     ----------
    #     x : Float[Tensor, '... d_model']
    #         Input tensor (after RMSNorm) of shape [..., d_model]

    #     Returns
    #     -------
    #     Float[Tensor, '... d_model']
    #         Output of FFN which will be added into the residual connection
    #     """
    
    #     w1x = torch.einsum('...i, oi -> ...o', x, self.w1)
    #     silu = w1x * torch.sigmoid(w1x)
    #     linear_1 = torch.einsum('...i, oi -> ...o', x, self.w3)

    #     swiglu = torch.einsum('...i, oi -> ...o', silu * linear_1, self.w2)

    #     return swiglu
    
    def forward(self, x: Float[Tensor, '... d_model']) -> Float[Tensor, '... d_model']:
        """forward forward Implementation using Linear layer implemented previously
        
        Parameters
        ----------
        x : Float[Tensor, '... d_model']
            Input tensor (after RMSNorm) of shape [..., d_model]

        Returns
        -------
        Float[Tensor, '... d_model']
            Output of FFN which will be added into the residual connection
        """
        
        # w1x is the first projection of x into d_ff space
        # w1x shape is (..., d_ff)
        w1x = self.w1(x)
        
        # silu is the activation applied to w1x in d_ff space
        # silu shape is (..., d_ff)
        silu = self.activation(w1x)
        
        # linear_1 is the linear path applied to x in d_ff space
        # that's not gated to preverse the linearity
        # shape is (..., d_ff)
        linear_1 = self.w3(x)
        
        # swiglu is the final output after applying the second projection
        # projecting back to d_model space
        # swiglu shape is (..., d_model)
        swiglu = self.w2(silu * linear_1)
        return swiglu
        
    
    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}, w1.shape={self.w1.weight.shape}, w2.shape={self.w2.weight.shape}, w3.shape={self.w3.weight.shape}"