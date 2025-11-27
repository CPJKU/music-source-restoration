"""
L1 loss implementation for audio source separation.

This module provides a simple L1 (Mean Absolute Error) loss between predicted and target waveforms.
"""

import torch
import torch.nn.functional as F


def get_loss_func():
    """
    Get L1 loss function for training.
    
    This function returns a loss function that uses L1 loss between
    predicted and target waveforms.
    
    Returns:
        Loss function that takes (output_dict, target_dict) and returns loss_dict
    """
    def loss_func(output, target):
        """
        Calculate L1 loss for training.
        
        Args:
            output: Dictionary containing 'waveform' key with model predictions
                   [batch_size, n_sources, n_channels, n_samples]
            target: Dictionary containing 'waveform' key with target sources
                   [batch_size, n_sources, n_channels, n_samples]
        
        Returns:
            Dictionary containing 'loss' key with L1 loss value
        """
        pred = output['waveform']  # [B, S, C, T]
        target_wav = target['waveform']  # [B, S, C, T]
        
        # Calculate L1 loss between prediction and target
        l1_loss = F.l1_loss(pred, target_wav)
        
        loss_dict = {
            'loss': l1_loss,
        }
        return loss_dict
    
    return loss_func


if __name__ == '__main__':
    # Test the L1 loss function
    import torch
    
    # Create test data
    batch_size, n_sources, n_channels, n_samples = 2, 4, 2, 48000
    pred = torch.randn(batch_size, n_sources, n_channels, n_samples)
    target = torch.randn(batch_size, n_sources, n_channels, n_samples)
    
    output_dict = {'waveform': pred}
    target_dict = {'waveform': target}
    
    # Test loss function
    loss_fn = get_loss_func()
    loss_dict = loss_fn(output_dict, target_dict)
    
    print(f"L1 Loss: {loss_dict['loss'].item():.4f}")
