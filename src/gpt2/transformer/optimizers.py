"""
This module contains various optimizer classes for training neural networks.
"""

import torch
from torch.optim import Optimizer
import math
from typing import Iterable


def learning_rate_scheduler(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> float:
    """learning_rate_scheduler

    Parameters
    ----------
    step : int
        Current training step.
    max_lr : float
        Maximum learning rate.
    min_lr : float
        Minimum learning rate.
    warmup_steps : int
        Number of warmup steps for the learning rate, T_W
    cosine_cycle_iters : int
        The number of cosine annealing iterations, T_c

    Returns
    -------
    float
        Adjusted learning rate for the current step.
    """
    
    lr = 0
    if step < warmup_iters:
        lr = (step / warmup_iters) * max_lr
    elif (warmup_iters <= step) and (step <= cosine_cycle_iters):
        lr = min_lr + 0.5 * (1 + math.cos(
            math.pi * (step - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        )) * (max_lr - min_lr)
        
    else:
        lr = min_lr
        
    return lr

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float
) -> None:
    """gradient_clipping

    Parameters
    ----------
    params : Iterable[torch.nn.Parameter]
        Iterable of model parameters to clip gradients for.
    max_l2_norm : float
        Maximum L2 norm for the gradients.
    """
    
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            
    total_norm = total_norm ** 0.5
    
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)


class AdamW(Optimizer):
    """Implementation of AdamW optimizer on a base `torch.optim.Optimizer` class
    """
    
    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float],
        weight_decay: float,
        eps: float = 1e-8,
    ):
        """__init__

        Parameters
        ----------
        params : iterable
            Parameters or Parameter groups to optimize.
        lr : float
            Learning rate for the optimizer.
        betas : tuple[float, float]
            Hyperparameters to control the first and second moment estimates
        weight_decay : float
            Weight decay rate to pull parameters towards 0.
        eps : float, optional
            Hyperparameter to improve numerical stability, by default 1e-8

        """

        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        # self.lr = lr
        # self.beta_1 = betas[0]
        # self.beta_2 = betas[1]
        # self.weight_decay = weight_decay
        # self.eps = eps
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        # super() initializes self.param_groups
        # each group in self.param_groups is a dictionary
        # containing the parameters and their associated hyperparameters.
        # dict_keys(['params', 'lr', 'betas', 'weight_decay', 'eps'])
        super().__init__(params, defaults)

        # Initialize m, v, and state for each parameter within each param_group
        # The state is managed by the optimizer itself.
        for group in self.param_groups:
            for p in group['params']:
                # The state for each parameter is stored in self.state[p]
                # self.state is a dictionary managed by torch.optim.Optimizer
                # and automatically handles device transfers if params move.
                if p not in self.state: # Initialize state for new parameters if needed
                    self.state[p] = dict(m=torch.zeros_like(p, memory_format=torch.preserve_format),
                                         v=torch.zeros_like(p, memory_format=torch.preserve_format),
                                         t=1)
        
    def step(self):
        """step Makes 1 update to the weights base on their gradients
        """
        
        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                # Get the state if state doesn't exist, it should be initialized at 1
                t = state.get("t", 1)
                
                grad = p.grad.data
                
                # First moment estimate
                # mul_ functions for in place calculations
                m = state["m"]                
                m.mul_(beta_1).add_(grad, alpha=1 - beta_1)
                
                # Second moment estimate
                v = state["v"]
                v.mul_(beta_2).addcmul_(grad, grad, value=1 - beta_2)
                
                lr_t = lr * math.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t)
                # Update the parameter
                # addcdiv_ performs inplace operations value * tensor1 / tensor2
                p.data.addcdiv_(m, v.sqrt() + eps, value=-lr_t)
                # Apply weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * lr)
                # Increment the time step
                state["t"] = t + 1
        return self
                
                