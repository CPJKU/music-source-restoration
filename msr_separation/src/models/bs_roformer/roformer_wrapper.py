"""
Wrapper for BS-Roformer to make it compatible with the existing training pipeline.

This wrapper adapts the BS-Roformer model to work with the same interface as ResUNet,
allowing it to be used in the existing training pipeline.
"""

import re
import torch
import torch.nn as nn
from typing import Dict, Optional

from .mel_band_roformer import MelBandRoformer
from .bs_roformer import BSRoformer


class RoformerWrapper(nn.Module):
    """
    Wrapper for BS-Roformer or Mel-Band-Roformer to make it compatible with the training pipeline.

    This wrapper adapts the BS-Roformer / Mel-Band-Roformer interface to match the expected input/output
    format of the training pipeline.
    """
    
    def __init__(
        self,
        dim: int = 512,
        depth: int = 12,
        stereo: bool = True,
        num_stems: int = 4,
        time_transformer_depth: int = 1,
        freq_transformer_depth: int = 1,
        freqs_per_bands: Optional[tuple | list] = None,
        freqs_per_bands_with_extra_bands: Optional[tuple | list] = None,
        freq_range: tuple = None,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        flash_attn: bool = True,
        num_residual_streams: int = 4,
        num_residual_fracs: int = 1,
        dim_freqs_in: int = 1025,
        stft_n_fft: int = 2048,
        stft_hop_length: int = 441,
        stft_win_length: int = 2048,
        stft_normalized: bool = False,
        zero_dc: bool = False,
        stft_window_fn = None,
        mask_estimator_depth: int = 2,
        mlp_expansion_factor: int = 4,
        use_mel_band_roformer: bool = False,
        load_pretrained: bool = False,
        lora_config: dict = None,
        use_dual_stft: bool = False,
        hop_length_48k: int = 480,
        win_length_48k: int = 2229,
        time_domain_extra_merge = False,
        resample_to_44100_for_model = False,
        **kwargs
    ):
        """
        Initialize Roformer wrapper.

        Args:
            dim: Model dimension
            depth: Number of transformer layers
            stereo: Whether to use stereo input/output
            num_stems: Number of output sources
            time_transformer_depth: Depth of time transformer
            freq_transformer_depth: Depth of frequency transformer
            freqs_per_bands: Frequency bands configuration
            freq_range: Frequency range to process
            dim_head: Dimension per attention head
            heads: Number of attention heads
            attn_dropout: Attention dropout rate
            ff_dropout: Feed-forward dropout rate
            flash_attn: Whether to use flash attention
            num_residual_streams: Number of residual streams
            num_residual_fracs: Number of residual fractions
            dim_freqs_in: Input frequency dimension
            stft_n_fft: STFT FFT size
            stft_hop_length: STFT hop length
            stft_win_length: STFT window length
            stft_normalized: Whether to normalize STFT
            zero_dc: Whether to zero DC component
            stft_window_fn: STFT window function
            mask_estimator_depth: Mask estimator depth
            mlp_expansion_factor: MLP expansion factor for mask estimator (controls memory usage)
            use_mel_band_roformer: Whether to use Mel-Band Roformer
            load_pretrained: Whether to load pretrained weights
        """
        super().__init__()

        if not use_dual_stft:
            if freq_range is not None:
                print(f"Using freq_range={freq_range} instead of (0, 1025).")
            else:
                freq_range = (0, 1025)  # Limit to first 1025 bins (bins 0-1024)
                print(f"Using freq_range=(0, 1025) to limit to pretrained frequency range")

            # Use the default 62-band configuration (without the extra 3 bands that would cover bins 1025-1114)
            # This is necessary because the model will only process bins 0-1024, so we need bands that sum to 1025
            from .bs_roformer import DEFAULT_FREQS_PER_BANDS
            if freqs_per_bands is not None:
                default_sum = sum(DEFAULT_FREQS_PER_BANDS)
                provided_sum = sum(freqs_per_bands) if isinstance(freqs_per_bands, (list, tuple)) else None
                if provided_sum and provided_sum != default_sum:
                    print(f"Warning: freqs_per_bands sums to {provided_sum} "
                          f"instead of {default_sum}. Overriding to use DEFAULT_FREQS_PER_BANDS (62 bands).")
            freqs_per_bands = DEFAULT_FREQS_PER_BANDS
            print(f"Using default 62-band configuration (DEFAULT_FREQS_PER_BANDS) instead of extended bands")

        # Initialize Roformer model
        roformer_kwargs = {
            'dim': dim,
            'depth': depth,
            'stereo': stereo,
            'num_stems': num_stems,
            'time_transformer_depth': time_transformer_depth,
            'freq_transformer_depth': freq_transformer_depth,
            'dim_head': dim_head,
            'heads': heads,
            'attn_dropout': attn_dropout,
            'ff_dropout': ff_dropout,
            'flash_attn': flash_attn,
            'num_residual_streams': num_residual_streams,
            'num_residual_fracs': num_residual_fracs,
            'dim_freqs_in': dim_freqs_in,
            'stft_n_fft': stft_n_fft,
            'stft_hop_length': stft_hop_length,
            'stft_win_length': stft_win_length,
            'stft_normalized': stft_normalized,
            'zero_dc': zero_dc,
            'mask_estimator_depth': mask_estimator_depth,
            'mlp_expansion_factor': mlp_expansion_factor,
            'load_pretrained': load_pretrained,
            'lora_config': lora_config,
            'hop_length_48k': hop_length_48k,
            'win_length_48k': win_length_48k,
            'time_domain_extra_merge': time_domain_extra_merge,
            'use_dual_stft': use_dual_stft,
            'resample_to_44100_for_model': resample_to_44100_for_model
        }

        # Only add optional parameters if they are not None
        if freqs_per_bands is not None:
            roformer_kwargs['freqs_per_bands'] = tuple(freqs_per_bands)
        if freqs_per_bands_with_extra_bands is not None:
            roformer_kwargs['freqs_per_bands_with_extra_bands'] = tuple(freqs_per_bands_with_extra_bands)
        if freq_range is not None:
            roformer_kwargs['freq_range'] = freq_range
        if stft_window_fn is not None:
            roformer_kwargs['stft_window_fn'] = stft_window_fn

        # Pass phased fine-tuning parameters from kwargs
        if 'phased_fine_tuning' in kwargs:
            roformer_kwargs['phased_fine_tuning'] = kwargs['phased_fine_tuning']
        if 'phased_fine_tuning_phase1_epochs' in kwargs:
            roformer_kwargs['phased_fine_tuning_phase1_epochs'] = kwargs['phased_fine_tuning_phase1_epochs']
        if 'phased_fine_tuning_crossover_bin' in kwargs:
            roformer_kwargs['phased_fine_tuning_crossover_bin'] = kwargs['phased_fine_tuning_crossover_bin']

        # New naming convention (preferred)
        if 'sr_48k' in kwargs:
            roformer_kwargs['sr_48k'] = kwargs['sr_48k']
        if 'n_fft_48k' in kwargs:
            roformer_kwargs['n_fft_48k'] = kwargs['n_fft_48k']

        # Old naming convention (for backward compatibility)
        if 'dual_stft_source_sr' in kwargs:
            roformer_kwargs['sr_48k'] = kwargs['dual_stft_source_sr']
        if 'dual_stft_target_sr' in kwargs:
            roformer_kwargs['sr_pretrained'] = kwargs['dual_stft_target_sr']
        if 'dual_stft_target_n_fft' in kwargs:
            roformer_kwargs['n_fft_pretrained'] = kwargs['dual_stft_target_n_fft']
        if 'dual_stft_source_n_fft' in kwargs:
            roformer_kwargs['n_fft_48k'] = kwargs['dual_stft_source_n_fft']

        if use_mel_band_roformer:
            self.roformer = MelBandRoformer(**roformer_kwargs)
        else:
            self.roformer = BSRoformer(**roformer_kwargs)

        # Store lora_config for later use in freezing logic
        self.lora_config = lora_config

        # If LoRA is enabled, freeze all non-LoRA parameters
        if lora_config is not None and lora_config.get('enabled', False):
            self._freeze_non_lora_parameters()

        # Store configuration
        self.num_stems = num_stems
        self.stereo = stereo
        self.input_channels = 2 if stereo else 1
        self.output_channels = 2 if stereo else 1
    
    def _freeze_non_lora_parameters(self):
        """
        Freeze all parameters that are not LoRA parameters.
        Only LoRA parameters (lora_A_* and lora_B_*) should have requires_grad=True.
        
        Additionally, allows fine-tuning LoRA, BandSplit, and MaskEstimator modules
        based on lora_config flags:
        - finetune_lora: If True (default), LoRA parameters are trainable. If False, LoRA parameters are frozen.
        - finetune_bandsplit: If True, BandSplit module parameters are trainable.
        - finetune_mask_estimator: If True, mask estimator parameters are trainable. If False, all mask estimators are frozen.
        - finetune_only_new_mask_estimators: If True (default when finetune_mask_estimator=True), only NEW mask estimators
          (indices 3-7) are unfrozen, while PRETRAINED mask estimators (indices 0, 1, 2, 8) remain frozen.
          If False, ALL mask estimators (both new and pretrained) are unfrozen when finetune_mask_estimator=True.
        
        This ensures memory efficiency during LoRA fine-tuning while allowing
        task-specific adaptation of input/output layers.
        """
        frozen_count = 0
        trainable_count = 0
        bandsplit_count = 0
        mask_estimator_count = 0
        
        # Get fine-tuning flags from config (default to False for backwards compatibility)
        # finetune_lora defaults to True to maintain backward compatibility (LoRA params were always trainable)
        finetune_lora = self.lora_config.get('finetune_lora', True) if self.lora_config else True
        finetune_bandsplit = self.lora_config.get('finetune_bandsplit', False) if self.lora_config else False
        finetune_mask_estimator = self.lora_config.get('finetune_mask_estimator', False) if self.lora_config else False
        # finetune_only_new_mask_estimators defaults to True to maintain backward compatibility
        # (previous behavior was to only fine-tune new mask estimators)
        finetune_only_new_mask_estimators = self.lora_config.get('finetune_only_new_mask_estimators', True) if self.lora_config else True

        print("LoRA fine-tuning enabled:" if finetune_lora else "LoRA fine-tuning disabled (LoRA parameters frozen).")
        print("BandSplit fine-tuning enabled:" if finetune_bandsplit else "BandSplit fine-tuning disabled.")
        print("MaskEstimator fine-tuning enabled:" if finetune_mask_estimator else "MaskEstimator fine-tuning disabled.")
        if finetune_mask_estimator:
            print(f"  -> Only new mask estimators (3-7) will be fine-tuned: {finetune_only_new_mask_estimators}")
        
        # Pretrained mask estimators: 0=vocals, 1=bass, 2=drums, 8=other
        # New mask estimators: 3=guitars, 4=keyboards, 5=synthesizers, 6=percussions, 7=orchestral
        pretrained_mask_indices = {0, 1, 2, 8}
        new_mask_indices = {3, 4, 5, 6, 7}
        
        for name, param in self.named_parameters():
            # LoRA parameters have 'lora' in their name (lora_A_0, lora_B_0, etc.)
            if 'lora' in name.lower():
                # Control LoRA parameter trainability based on finetune_lora flag
                if finetune_lora:
                    # Ensure LoRA parameters are trainable (explicitly set to True)
                    if not param.requires_grad:
                        param.requires_grad = True
                    trainable_count += 1
                else:
                    # Freeze LoRA parameters
                    param.requires_grad = False
                    frozen_count += 1
            elif finetune_bandsplit and 'band_split' in name.lower():
                # Allow fine-tuning BandSplit if enabled
                if not param.requires_grad:
                    param.requires_grad = True
                bandsplit_count += 1
            elif finetune_mask_estimator and 'mask_estimator' in name.lower():
                # Fine-tune mask estimators based on finetune_only_new_mask_estimators flag
                match = re.search(r'mask_estimators\.(\d+)', name)
                if match:
                    estimator_idx = int(match.group(1))
                    if estimator_idx in new_mask_indices:
                        # Always unfreeze new mask estimators (indices 3-7) when finetune_mask_estimator=True
                        if not param.requires_grad:
                            param.requires_grad = True
                        mask_estimator_count += 1
                    elif estimator_idx in pretrained_mask_indices:
                        # Pretrained mask estimators (indices 0,1,2,8): unfreeze only if finetune_only_new_mask_estimators=False
                        if finetune_only_new_mask_estimators:
                            # Keep pretrained mask estimators frozen
                            param.requires_grad = False
                            frozen_count += 1
                        else:
                            # Unfreeze pretrained mask estimators too
                            if not param.requires_grad:
                                param.requires_grad = True
                            mask_estimator_count += 1
                    else:
                        # Unknown mask estimator index - freeze it to be safe
                        param.requires_grad = False
                        frozen_count += 1
                else:
                    # Could not extract index - freeze to be safe
                    param.requires_grad = False
                    frozen_count += 1
            else:
                # Freeze all other non-LoRA parameters
                param.requires_grad = False
                frozen_count += 1
        
        # Print summary
        trainable_desc = []
        if trainable_count > 0:
            trainable_desc.append(f"{trainable_count} LoRA")
        if bandsplit_count > 0:
            trainable_desc.append(f"{bandsplit_count} BandSplit")
        if mask_estimator_count > 0:
            trainable_desc.append(f"{mask_estimator_count} MaskEstimator")
        
        print(f"LoRA enabled: Frozen {frozen_count} parameters, {' + '.join(trainable_desc)} parameters remain trainable")
        
        # Verify that optimizer will only receive intended trainable parameters
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Roformer wrapper.
        
        Args:
            input_dict: Dictionary containing:
                       - 'mixture': input audio [batch_size, n_channels, n_samples] (48kHz for dual-STFT)
                       - 'mixture_44100': optional pre-resampled 44.1kHz audio [batch_size, n_channels, n_samples]
                                         (only used when use_dual_stft=True with Lightning resamplers)
        
        Returns:
            Dictionary containing 'waveform' key with separated sources
            [batch_size, n_sources, n_channels, n_samples]
        """
        # Extract mixture from input dictionary
        mixture = input_dict['mixture']  # [B, C, T]
        
        # Ensure correct input format for BS-Roformer
        if mixture.dim() == 2:
            # Add channel dimension if mono
            mixture = mixture.unsqueeze(1)  # [B, 1, T]
        
        # Roformer expects [B, C, T] format
        # Forward through Roformer
        separated_audio = self.roformer(mixture)  # [B, S, C, T] or [B, C, T] if num_stems=1
        
        # Ensure output has correct shape
        if self.num_stems == 1:
            # Single source case - add source dimension
            separated_audio = separated_audio.unsqueeze(1)  # [B, 1, C, T]
        
        # Create output dictionary
        output_dict = {
            'waveform': separated_audio,
            'source_mask': torch.ones(separated_audio.shape[0], self.num_stems, 
                                    device=separated_audio.device, dtype=separated_audio.dtype)
        }
        
        # Add phase information for phased fine-tuning
        if hasattr(self.roformer, 'current_phase'):
            output_dict['phase'] = self.roformer.current_phase
        
        return output_dict
    
    def forward_inference(self, input_dict: Dict[str, torch.Tensor], nsources: int = 4, label_len: int = 18) -> Dict[str, torch.Tensor]:
        """
        Forward pass for inference (compatibility with ResUNet interface).
        
        Args:
            input_dict: Dictionary containing 'mixture' and 'label_vector' keys
            nsources: Number of sources (unused, kept for compatibility)
            label_len: Label length (unused, kept for compatibility)
        
        Returns:
            Dictionary containing separated sources
        """
        # For Roformer, we don't use label vectors in the same way
        # Just use the standard forward pass
        return self.forward(input_dict)
    
    def chunk_inference(self, input_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Chunk-based inference (compatibility with ResUNet interface).
        
        Args:
            input_dict: Dictionary containing 'mixture' and 'condition' keys
        
        Returns:
            Separated audio tensor
        """
        # For Roformer, we don't use conditions in the same way
        # Just use the standard forward pass
        output_dict = self.forward(input_dict)
        return output_dict['waveform']
    
    def parameter_groups(self, lr: float, lr_mask_pretrained: float, lr_mask_new: float, lr_decay_factor: float, weight_decay: float):
        """
        Parameter groups for optimizer (compatibility with ResUNet interface).
        
        Args:
            lr: Learning rate for pretrained parameters (unused)
            lr_mask_pretrained: Learning rate for pretrained mask estimator parameters
            lr_mask_new: Learning rate for new mask estimator parameters
            lr_decay_factor: Learning rate decay factor (unused)
            weight_decay: Weight decay
        
        Returns:
            List of parameter groups
        """
        roformer_params = []
        mask_pretrained_params = []
        mask_new_params = []

        for name, p in self.named_parameters():
            # Only include parameters that require gradients (are trainable)
            if not p.requires_grad:
                continue
                
            if (name.startswith('roformer.layers') or name.startswith('roformer.band_split')
                    or name.startswith('roformer.final_norm')):
                roformer_params.append(p)
            elif name.startswith('roformer.mask_estimators'):
                # Extract mask estimator index using regex to handle multi-digit indices
                match = re.search(r'roformer.mask_estimators\.(\d+)', name)
                if match:
                    estimator_idx = int(match.group(1))
                    # Pretrained mask estimators: 0=vocals, 1=bass, 2=drums, 8=other
                    # New mask estimators: 3=guitars, 4=keyboards, 5=synthesizers, 6=percussions, 7=orchestral
                    if estimator_idx in (0, 1, 2, 8):
                        mask_pretrained_params.append(p)
                    else:
                        mask_new_params.append(p)
                else:
                    raise ValueError(f"Could not extract mask_estimator index from parameter name: {name}")
            elif 'lora' in name.lower():
                # LoRA parameters - add to roformer_params group (they modify roformer layers)
                # Use the same learning rate as roformer layers
                roformer_params.append(p)
            else:
                raise ValueError(f"Unexpected key in model: {name}")

        param_groups = [
            {'params': roformer_params, 'lr': lr, 'name': 'roformer'},  # model head (besides base model and seq model)
            {'params': mask_pretrained_params, 'lr': lr_mask_pretrained, 'name': 'mask_pretrained'},
            {'params': mask_new_params, 'lr': lr_mask_new, 'name': 'mask_new'},
        ]

        print(f"Separate_params. Roformer lr={param_groups[0]['lr']}, mask_pretrained lr={param_groups[1]['lr']}, " \
              + f"mask_new lr={param_groups[2]['lr']},")
        return param_groups
    
    def separate_params(self, lr: float, lr_mask_pretrained: float, lr_mask_new: float, lr_decay_factor: float, weight_decay: float):
        """
        Separate parameters for optimizer (compatibility with ResUNet interface).
        
        Args:
            lr: Learning rate
            lr_mask_pretrained: Learning rate for pretrained mask estimator parameters
            lr_mask_new: Learning rate for new mask estimator parameters
            lr_decay_factor: Learning rate decay factor (unused)
            weight_decay: Weight decay
        
        Returns:
            List of parameter groups
        """
        return self.parameter_groups(lr, lr_mask_pretrained, lr_mask_new, lr_decay_factor, weight_decay)


# Factory function for easy instantiation
def get_bs_roformer_wrapper(**kwargs):
    """
    Factory function to create BS-Roformer wrapper.
    
    Args:
        **kwargs: Arguments to pass to BSRoformerWrapper
    
    Returns:
        BSRoformerWrapper instance
    """
    return RoformerWrapper(**kwargs)
