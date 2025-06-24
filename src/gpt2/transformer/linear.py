import torch
import math

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None, bias=False, *args, **kwargs):
        """
        A simple linear layer that performs a linear transformation.
        
        Parameters:
        -----------
        in_features : int
            Dimension of input features.
        out_features : int
            Dimension of output features.
        device : torch.device, optional
            The device on which to create the layer. Defaults to None, which uses the CPU.
        dtype : torch.dtype, optional
            The data type of the layer's parameters. Defaults to None, which uses the default dtype (usually float32).
        """ 
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.bias = None if not bias else torch.nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))

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
        
        stdv = math.sqrt(2.0 / (self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=stdv * stdv, a=-3 * stdv, b=3 * stdv)
        # Check if bias is enabled
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear layer.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features).
        
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, out_features).
        """
        result = torch.einsum('...i, oi->...o', x, self.weight)
        result += self.bias if self.bias is not None else 0
        
        return result
    
    def extra_repr(self) -> str:
        """
        Returns a string representation of the layer's parameters.
        
        Returns:
        --------
        str
            A string representation of the layer's parameters.
        """
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, weight_shape={self.weight.shape}, bias_shape={self.bias.shape if self.bias is not None else None}, device={self.weight.device}, dtype={self.weight.dtype}'