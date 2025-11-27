"""
Exponential Moving Average (EMA) implementation for model training.

This module provides EMA functionality to maintain a moving average of model parameters,
which typically leads to better test-time performance.
"""

import copy
import torch
import torch.nn as nn
from typing import Optional


class ExponentialMovingAverage:
    """
    Exponential Moving Average of model parameters.
    
    Maintains a moving average of model parameters using an exponential decay.
    This is commonly used to improve model stability and final performance.
    
    Args:
        model: The model to track
        decay: Decay rate for the moving average (0.9999 means very slow decay)
        update_norm: Only update parameters when gradient norm exceeds this threshold
        device: Device to store EMA parameters on
    
    Example:
        >>> model = MyModel()
        >>> ema = ExponentialMovingAverage(model, decay=0.9999)
        >>> # During training
        >>> for batch in dataloader:
        >>>     loss = model(batch)
        >>>     loss.backward()
        >>>     optimizer.step()
        >>>     ema.update()  # Update EMA after optimizer step
        >>> # For evaluation
        >>> with ema.average_parameters():
        >>>     predictions = model(inputs)
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_norm: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.decay = decay
        self.update_norm = update_norm
        self.device = device or next(model.parameters()).device
        
        # Initialize EMA parameters as a copy of current parameters on the same device
        self.ema_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.ema_params[name] = param.data.clone().to(self.device)
        
        # Track number of updates for adaptive decay
        self.num_updates = 0
    
    def update(self, model: Optional[nn.Module] = None):
        """
        Update the EMA parameters using current model parameters.
        
        Args:
            model: Optional model to update from. If None, uses self.model
        """
        model = model or self.model
        self.num_updates += 1
        
        # Adaptive decay (optional)
        # Start with a lower decay rate for the first few updates
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        for name, param in model.named_parameters():
            if name not in self.ema_params:
                continue
            
            if not param.requires_grad:
                continue
            
            if param.data.dtype.is_floating_point:
                # Update EMA parameter - ensure both tensors are on the same device
                ema_param = self.ema_params[name]
                param_data = param.data
                
                # Ensure ema_param is on the same device as param_data
                if ema_param.device != param_data.device:
                    self.ema_params[name] = ema_param.to(param_data.device)
                    ema_param = self.ema_params[name]
                
                # Update EMA parameter
                self.ema_params[name] = (
                    decay * ema_param + (1.0 - decay) * param_data
                )
    
    def apply(self, model: Optional[nn.Module] = None):
        """
        Apply EMA parameters to the model.
        
        Args:
            model: Model to apply EMA to. If None, uses self.model
        """
        model = model or self.model
        
        for name, param in model.named_parameters():
            if name in self.ema_params:
                param.data.copy_(self.ema_params[name])
    
    def store(self):
        """
        Store current model parameters (typically before applying EMA).
        """
        model_params = {}
        for name, param in self.model.named_parameters():
            model_params[name] = param.data.clone()
        return model_params
    
    def restore(self, model_params: dict):
        """
        Restore model parameters from stored state.
        
        Args:
            model_params: Dictionary of saved model parameters
        """
        for name, param in self.model.named_parameters():
            if name in model_params:
                param.data.copy_(model_params[name])
    
    def copy_to(self, model: nn.Module):
        """
        Copy EMA parameters to another model.
        
        Args:
            model: Target model to copy EMA parameters to
        """
        for name, param in model.named_parameters():
            if name in self.ema_params:
                param.data.copy_(self.ema_params[name])
    
    def state_dict(self):
        """
        Return state dict of EMA parameters.
        """
        return {
            'ema_params': self.ema_params,
            'num_updates': self.num_updates,
        }
    
    def load_state_dict(self, state_dict):
        """
        Load EMA state from state dict.
        
        Args:
            state_dict: State dictionary containing EMA parameters
        """
        self.ema_params = state_dict.get('ema_params', {})
        self.num_updates = state_dict.get('num_updates', 0)
