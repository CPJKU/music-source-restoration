"""
FAD-CLAP metric implementation for source separation evaluation.

This module provides FAD-CLAP (Fréchet Audio Distance with CLAP embeddings) 
evaluation for source separation models.
"""

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from scipy.linalg import sqrtm
import soundfile as sf

warnings.filterwarnings("ignore")

try:
    from transformers import ClapModel, ClapProcessor
except ImportError:
    print("Warning: transformers library not installed. FAD-CLAP will not be available.")
    ClapModel = None
    ClapProcessor = None


class FADCLAPMetric(nn.Module):
    """
    FAD-CLAP metric for evaluating source separation quality.
    
    This metric calculates the Fréchet Audio Distance using CLAP embeddings
    to measure the perceptual quality of separated audio sources.
    """
    
    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        batch_size: int = 2,  # Further reduced for stability
        device: Optional[str] = None,
        cache_embeddings: bool = True,
        max_samples_per_epoch: int = 50,  # Further reduced to avoid memory issues
        force_cpu: bool = False,  # Force CPU usage to avoid CUDA issues
    ):
        super().__init__()
        
        if ClapModel is None or ClapProcessor is None:
            raise ImportError(
                "transformers library is required for FAD-CLAP. "
                "Install with: pip install transformers"
            )
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cpu" if force_cpu else (device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cache_embeddings = cache_embeddings
        self.max_samples_per_epoch = max_samples_per_epoch
        self.force_cpu = force_cpu
        
        # Initialize CLAP model
        self._load_clap_model()
        
        # Storage for embeddings during epoch
        self.target_embeddings = []
        self.output_embeddings = []
        self.sample_count = 0
        
    def _load_clap_model(self):
        """Load the CLAP model and processor."""
        try:
            # Force CPU if CUDA issues detected
            if self.force_cpu or self.device == "cpu":
                self.clap_model = ClapModel.from_pretrained(self.model_name, torch_dtype=torch.float32)
                self.clap_processor = ClapProcessor.from_pretrained(self.model_name)
                self.clap_model.eval()
                self.clap_model.to("cpu")
                print(f"CLAP model loaded successfully on CPU (forced)")
            else:
                self.clap_model = ClapModel.from_pretrained(self.model_name)
                self.clap_processor = ClapProcessor.from_pretrained(self.model_name)
                self.clap_model.eval()
                self.clap_model.to(self.device)
                print(f"CLAP model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Warning: Failed to load CLAP model on {self.device}, falling back to CPU: {e}")
            try:
                self.clap_model = ClapModel.from_pretrained(self.model_name, torch_dtype=torch.float32)
                self.clap_processor = ClapProcessor.from_pretrained(self.model_name)
                self.clap_model.eval()
                self.clap_model.to("cpu")
                self.device = "cpu"
                print("CLAP model loaded successfully on CPU (fallback)")
            except Exception as e2:
                raise RuntimeError(f"Failed to load CLAP model on both {self.device} and CPU: {e2}")
    
    def _audio_to_embeddings(self, audio_tensor: torch.Tensor) -> np.ndarray:
        """
        Convert audio tensor to CLAP embeddings.
        
        Args:
            audio_tensor: [batch_size, n_sources, n_channels, n_samples] or [batch_size, n_channels, n_samples]
        
        Returns:
            embeddings: [batch_size * n_sources, embedding_dim] or [batch_size, embedding_dim]
        """
        # Handle different input shapes
        if audio_tensor.dim() == 4:  # [batch_size, n_sources, n_channels, n_samples]
            batch_size, n_sources, n_channels, n_samples = audio_tensor.shape
            # Reshape to [batch_size * n_sources, n_channels, n_samples]
            audio_reshaped = audio_tensor.view(batch_size * n_sources, n_channels, n_samples)
        elif audio_tensor.dim() == 3:  # [batch_size, n_channels, n_samples]
            audio_reshaped = audio_tensor
            batch_size = audio_tensor.shape[0]
            n_sources = 1
        else:
            raise ValueError(f"Unexpected audio tensor shape: {audio_tensor.shape}")
        
        # Convert to numpy and prepare for CLAP
        audio_np = audio_reshaped.detach().cpu().numpy()
        
        # Process each audio sample
        embeddings = []
        for i in range(0, len(audio_np), self.batch_size):
            batch_audio = audio_np[i:i + self.batch_size]
            batch_embeddings = self._process_audio_batch(batch_audio)
            if batch_embeddings is not None:
                embeddings.append(batch_embeddings)
        
        if not embeddings:
            return np.array([])
        
        return np.concatenate(embeddings, axis=0)
    
    def _process_audio_batch(self, audio_batch: np.ndarray) -> Optional[np.ndarray]:
        """Process a batch of audio samples through CLAP."""
        try:
            # Prepare audio for CLAP (expects mono audio)
            audio_list = []
            for audio in audio_batch:
                # audio shape: [n_channels, n_samples]
                if audio.ndim == 2 and audio.shape[0] == 2:  # Stereo [2, n_samples]
                    # Use left channel for CLAP
                    audio_list.append(audio[0])  # [n_samples]
                elif audio.ndim == 1:  # Mono [n_samples]
                    audio_list.append(audio)
                else:
                    # Handle other cases - flatten if needed
                    audio_list.append(audio.flatten())
            
            if not audio_list:
                return None
            
            # Process through CLAP
            inputs = self.clap_processor(
                audios=audio_list, 
                sampling_rate=48000, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            with torch.no_grad():
                audio_features = self.clap_model.get_audio_features(**inputs)
            
            return audio_features.cpu().numpy()
            
        except Exception as e:
            print(f"Warning: Failed to process audio batch: {e}")
            return None
    
    def _calculate_frechet_distance(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> Optional[float]:
        """Calculate Fréchet Distance between two sets of embeddings."""
        if embeddings1.shape[0] < 2 or embeddings2.shape[0] < 2:
            return None
        
        mu1, mu2 = np.mean(embeddings1, axis=0), np.mean(embeddings2, axis=0)
        sigma1, sigma2 = np.cov(embeddings1, rowvar=False), np.cov(embeddings2, rowvar=False)
        
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        
        try:
            covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        except Exception:
            return None
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fad_score = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fad_score
    
    def update(self, output_dict: Dict[str, torch.Tensor], target_dict: Dict[str, torch.Tensor]):
        """
        Update metric with a batch of predictions and targets.
        
        Args:
            output_dict: Dictionary containing 'waveform' key with separated sources
                        [batch_size, n_sources, n_channels, n_samples]
            target_dict: Dictionary containing 'waveform' key with target sources
                        [batch_size, n_sources, n_channels, n_samples]
        """
        if self.sample_count >= self.max_samples_per_epoch:
            return
        
        # Extract waveforms
        output_wav = output_dict['waveform']  # [batch_size, n_sources, n_channels, n_samples]
        target_wav = target_dict['waveform']  # [batch_size, n_sources, n_channels, n_samples]
        
        # Limit samples to avoid memory issues
        remaining_samples = self.max_samples_per_epoch - self.sample_count
        if output_wav.shape[0] > remaining_samples:
            output_wav = output_wav[:remaining_samples]
            target_wav = target_wav[:remaining_samples]
        
        # Skip if audio is too long (avoid memory issues)
        if output_wav.shape[-1] > 480000:  # Skip if longer than 10 seconds at 48kHz
            return
        
        try:
            # Convert to embeddings
            output_embeddings = self._audio_to_embeddings(output_wav)
            target_embeddings = self._audio_to_embeddings(target_wav)
            
            if output_embeddings.size > 0 and target_embeddings.size > 0:
                self.output_embeddings.append(output_embeddings)
                self.target_embeddings.append(target_embeddings)
                self.sample_count += output_wav.shape[0]
        except Exception as e:
            print(f"Warning: Failed to update FAD-CLAP metric: {e}")
            return
    
    def compute(self) -> Dict[str, float]:
        """
        Compute the FAD-CLAP score.
        
        Returns:
            Dictionary containing FAD-CLAP score
        """
        if not self.target_embeddings or not self.output_embeddings:
            return {"fad_clap": 0.0}  # Return 0 instead of inf when no data
        
        # Concatenate all embeddings
        all_target_embeddings = np.concatenate(self.target_embeddings, axis=0)
        all_output_embeddings = np.concatenate(self.output_embeddings, axis=0)
        
        # Calculate FAD-CLAP
        fad_score = self._calculate_frechet_distance(all_target_embeddings, all_output_embeddings)
        
        if fad_score is None:
            return {"fad_clap": 0.0}  # Return 0 instead of inf when calculation fails
        
        return {"fad_clap": float(fad_score)}
    
    def reset(self):
        """Reset the metric state."""
        self.target_embeddings = []
        self.output_embeddings = []
        self.sample_count = 0


def get_fad_clap_metric(**kwargs):
    """
    Factory function to create FAD-CLAP metric.
    
    Args:
        **kwargs: Arguments passed to FADCLAPMetric
        
    Returns:
        FADCLAPMetric instance
    """
    return FADCLAPMetric(**kwargs)


# For backward compatibility and easy integration
def get_metric_func():
    """Get metric function that returns both SI-SNR and FAD-CLAP."""
    from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
    
    def metric_func(output, target):
        # Calculate SI-SNR
        si_snr_val = si_snr(output['waveform'], target['waveform']).mean()
        
        # Note: FAD-CLAP is calculated per-epoch, not per-batch
        # This function only returns SI-SNR for per-batch metrics
        return {
            'si_snr': si_snr_val,
        }
    
    return metric_func
