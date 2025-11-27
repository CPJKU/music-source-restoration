from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr


def get_loss_func():
    """
    Get SI-SNR loss function for training.
    
    This function returns a loss function that uses negated SI-SNR as the loss.
    SI-SNR is more stable than regular SNR and provides better gradient flow.
    
    Returns:
        Loss function that takes (output_dict, target_dict) and returns loss_dict
    """
    def loss_func(output, target):
        """
        Calculate SI-SNR loss for training.
        
        Args:
            output: Dictionary containing 'waveform' key with model predictions
                   [batch_size, n_sources, n_channels, n_samples]
            target: Dictionary containing 'waveform' key with target sources
                   [batch_size, n_sources, n_channels, n_samples]
        
        Returns:
            Dictionary containing 'loss' key with negated SI-SNR value
        """
        output_wav = output['waveform']  # [B, S, C, T]
        target_wav = target['waveform']  # [B, S, C, T]

        # 🔥 VECTORIZED: Reshape to [B*S, C, T] → single si_snr call
        B, S, C, T = output_wav.shape
        output_flat = output_wav.view(B * S, C, T)  # [B*S, C, T]
        target_flat = target_wav.view(B * S, C, T)  # [B*S, C, T]

        # Calculate SI-SNR for all sources
        batch_si_snrs = si_snr(output_flat, target_flat)  # [B*S]
        
        # Use negated SI-SNR as loss (higher SI-SNR = lower loss)
        si_snr_loss = -batch_si_snrs.mean()
        
        loss_dict = {
            'loss': si_snr_loss,  # Main loss for backpropagation
        }
        return loss_dict
    
    return loss_func