"""
Multi-loss manager for combining multiple loss functions with configurable weights.

This module provides a flexible way to combine multiple loss functions
with individual weights, eliminating the need for separate combined loss files.
"""

import torch
from typing import Dict, List, Any
import importlib


class MultiLossManager:
    """
    Manager for combining multiple loss functions with weights.
    """
    
    def __init__(self, loss_configs: List[Dict[str, Any]]):
        """
        Initialize multi-loss manager.
        
        Args:
            loss_configs: List of loss configurations, each containing:
                - module: Module path (e.g., 'src.training.loss.l1')
                - main: Function name (e.g., 'get_loss_func')
                - args: Arguments for the loss function
                - weight: Weight for this loss in the final combination
                - name: Optional custom name for this loss (defaults to module name)
        """
        self.loss_functions = []
        self.weights = []
        self.loss_names = []
        
        for config in loss_configs:
            # Import the loss module
            module = importlib.import_module(config["module"])
            
            # Get the loss function
            if 'args' in config:
                loss_func = getattr(module, config["main"])(**config["args"])
            else:
                loss_func = getattr(module, config["main"])()
            
            self.loss_functions.append(loss_func)
            self.weights.append(config["weight"])
            
            # Generate descriptive name for this loss
            if 'name' in config:
                loss_name = config['name']
            else:
                # Extract loss name from module path (e.g., 'src.training.loss.l1' -> 'l1')
                module_parts = config["module"].split('.')
                loss_name = module_parts[-1]  # Get the last part (e.g., 'l1')
            
            self.loss_names.append(loss_name)
    
    def __call__(self, output_dict: Dict[str, torch.Tensor], target_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss from multiple loss functions.
        
        Args:
            output_dict: Model output dictionary
            target_dict: Target dictionary
            
        Returns:
            Dictionary containing combined loss and individual loss components
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Compute each loss function
        for i, (loss_func, weight, loss_name) in enumerate(zip(self.loss_functions, self.weights, self.loss_names)):
            individual_loss_dict = loss_func(output_dict, target_dict)
            
            # Extract the main loss value
            individual_loss = individual_loss_dict['loss']
            
            # Add weighted loss to total
            weighted_loss = weight * individual_loss
            total_loss += weighted_loss
            
            # Store individual loss components with descriptive names
            loss_dict[f"{loss_name}_loss"] = individual_loss
            loss_dict[f"{loss_name}_loss_weighted"] = weighted_loss
        
        # Add total combined loss
        loss_dict['loss'] = total_loss
        
        return loss_dict


def get_loss_func(loss_configs: List[Dict[str, Any]]):
    """
    Factory function for multi-loss manager.
    
    Args:
        loss_configs: List of loss configurations
        
    Returns:
        MultiLossManager instance
    """
    return MultiLossManager(loss_configs)


if __name__ == '__main__':
    # Test the multi-loss manager
    import torch
    
    # Create test data
    batch_size, n_sources, n_channels, n_samples = 2, 4, 2, 48000
    pred = torch.randn(batch_size, n_sources, n_channels, n_samples)
    target = torch.randn(batch_size, n_sources, n_channels, n_samples)
    
    output_dict = {'waveform': pred}
    target_dict = {'waveform': target}
    
    # Test with multiple loss functions
    loss_configs = [
        {
            "module": "src.training.loss.l1",
            "main": "get_loss_func",
            "args": {},
            "weight": 0.5
        },
        {
            "module": "src.training.loss.si_snr",
            "main": "get_loss_func", 
            "args": {},
            "weight": 1.0
        }
    ]
    
    # Test loss function
    loss_fn = get_loss_func(loss_configs)
    loss_dict = loss_fn(output_dict, target_dict)
    
    print("Multi-Loss Results:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.item():.4f}")
        else:
            print(f"{key}: {value:.4f}")
