import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Subset, Dataset
from torchmetrics.functional.audio import signal_noise_ratio as snr
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
from typing import List, Dict, Any

from src.config_updates import IR_PATHS
from src.training.lightningmodule.base_lightningmodule import BaseLightningModule
from src.training.metrics.fad_clap import FADCLAPMetric
from src.training.metrics.si_snr import SI_SNRMetric
from src.training.metrics.multi_mel_snr import MultiMelSNRMetric, multi_mel_snr
from src.utils import LABELS
from src.training.lightningmodule.helper import map_4stem_to_9stem_checkpoint

import pdb
import re

from torch.cuda.amp import autocast

from src.dsp import MusicEffectPipeline, build_mastering_compose, batch_has_signal_lufs


class SubsetWithCollateFn(Subset):
    """
    A Subset wrapper that forwards the collate_fn attribute to the underlying dataset.
    
    This is needed because PyTorch's Subset class doesn't forward custom attributes
    like collate_fn, which are needed by the DataLoader.
    """
    def __init__(self, dataset: Dataset, indices: List[int]):
        super().__init__(dataset, indices)
        self._dataset = dataset
    
    def __getattr__(self, name):
        # Forward collate_fn and any other attributes to the underlying dataset
        if name == 'collate_fn' or hasattr(self._dataset, name):
            return getattr(self._dataset, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class MultiSourceSeparationLightning(BaseLightningModule):
    """
    Lightning module for multi-source separation.
    
    The model outputs all target sources simultaneously.
    For music source separation, this outputs: bass, drums, vocals, other
    
    Args:
        use_film_conditioning: If True, uses LABELS for FiLM conditioning. 
                              If False (default), runs unconditional separation.
        n_sources: Number of sources to separate (default: 4)
        progress_bar_metrics: List of full metric names to show in progress bar.
                             If None (default), all metrics appear in progress bar.
                             If list, only specified metrics appear in progress bar.
                             Example: ['step_train/loss', 'epoch_val/si_snr'] to show only specific metrics.
        resample_to_44100_for_model: If True, down-samples mixture from 48kHz to 44.1kHz before model,
                                     and up-samples predictions back to 48kHz. All metrics calculated at 48kHz.
                                     Default: False
        test_chunk_overlap: Overlap ratio for chunked inference during testing (0.0 = no overlap, 0.5 = 50% overlap).
                           Overlapping chunks are blended using a Hanning window to avoid boundary artifacts.
                           Default: 0.0 (no overlap)
    """
    def __init__(
            self,
            model: dict,
            loss: dict,
            optimizer: dict,
            lr_scheduler: dict = None,
            is_validation: bool = False,
            metric: dict = None,
            n_sources: int = 4,
            use_film_conditioning: bool = False,
            use_fad_clap: bool = True,
            fad_clap_batch_size: int = 4,
            fad_clap_max_samples: int = 100,
            fad_clap_force_cpu: bool = False,
            use_si_snr: bool = True,
            use_multi_mel_snr: bool = False,
            sampling_rate: int = 48000,
            columns: list = None,
            progress_bar_metrics: list = None,
            use_ema: bool = False,
            ema_decay: float = 0.9999,
            resample_to_44100_for_model: bool = False,
            test_chunk_overlap: float = 0.0,
            augment: bool = False,
            target_key: str = "waveform",
            debug: bool = False,
            msr_bench_validation: bool = False,
            msr_bench_max_files: int = None,
            msr_bench_window_duration: float = 10.0,
            msr_bench_seed: int = 42
    ):

        model["args"]["resample_to_44100_for_model"] = resample_to_44100_for_model

        super().__init__(
            model=model,
            loss=loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_validation=is_validation,
            metric=metric,
            use_ema=use_ema,
            ema_decay=ema_decay,
        )

        self.resample_AND_use_dual_stft = False
        if resample_to_44100_for_model and model.get('args', {}).get('use_dual_stft', False):
            self.resample_AND_use_dual_stft = True

        self.debug = debug
        self.n_sources = n_sources
        self.use_film_conditioning = use_film_conditioning
        self.use_fad_clap = use_fad_clap
        self.use_si_snr = use_si_snr
        self.use_multi_mel_snr = use_multi_mel_snr
        self.columns = columns
        self.sampling_rate = sampling_rate
        self.resample_to_44100_for_model = resample_to_44100_for_model
        self.test_chunk_overlap = float(test_chunk_overlap)  # Overlap ratio (0.0 = no overlap, 0.5 = 50% overlap)
        if self.test_chunk_overlap > 0:
            print(f"Test chunk overlap enabled: {self.test_chunk_overlap*100:.1f}% overlap")
        
        # Configuration for which metrics appear in progress bar
        # If None, use default behavior (all metrics in progress bar)
        # If list, only metrics in the list will appear in progress bar
        self.progress_bar_metrics = progress_bar_metrics
        # Parameters for frame-level activity detection
        self.activity_window_ms = 20.0  # moving RMS window
        self.activity_relative_threshold = 0.01 
        self.activity_threshold = 1e-5  # RMS threshold
        self.activity_min_frames = 0    # require at least N active samples (0 to disable)
        self.augment = augment
        self.target_key = target_key
        if self.target_key == "aug_waveform" and not self.augment:
            raise Exception("Warning: target_key is 'aug_waveform' but augment is False. No augmentation will be applied.")

        self.apply_augmentation = MusicEffectPipeline(sample_rate=sampling_rate, ir_paths=IR_PATHS)
        self.mastering_chain = build_mastering_compose(sample_rate=sampling_rate)
        
        # Initialize stem presence tracking counters
        # Will be reset each epoch in on_train_epoch_start
        self.stem_presence_counts = None

        if self.use_film_conditioning:
            # Create one-hot encodings for each source label
            assert len(LABELS) == n_sources, f"LABELS length ({len(LABELS)}) must match n_sources ({n_sources})"
            self.label_vectors = torch.eye(len(LABELS), dtype=torch.float32)  # [n_sources, n_sources]
            print(f"FiLM conditioning enabled with labels: {LABELS}")
        else:
            print("FiLM conditioning disabled (unconditional separation)")
        
        # Initialize FAD-CLAP metrics if enabled (separate instances for train/val)
        if self.use_fad_clap:
            try:
                self.train_fad_clap_metric = FADCLAPMetric(
                    batch_size=fad_clap_batch_size,
                    max_samples_per_epoch=fad_clap_max_samples,
                    force_cpu=fad_clap_force_cpu
                )
                self.val_fad_clap_metric = FADCLAPMetric(
                    batch_size=fad_clap_batch_size,
                    max_samples_per_epoch=fad_clap_max_samples,
                    force_cpu=fad_clap_force_cpu
                )
                print("FAD-CLAP metric enabled (separate instances for train/val)")
            except Exception as e:
                print(f"Warning: Failed to initialize FAD-CLAP metric: {e}")
                self.use_fad_clap = False
        else:
            self.train_fad_clap_metric = None
            self.val_fad_clap_metric = None
        
        # Initialize SI-SNR metrics if enabled (separate instances for train/val)
        if self.use_si_snr:
            self.train_si_snr_metric = SI_SNRMetric()
            self.val_si_snr_metric = SI_SNRMetric()
            print("SI-SNR metric enabled (separate instances for train/val)")
        else:
            self.train_si_snr_metric = None
            self.val_si_snr_metric = None
        
        # Initialize Multi-Mel-SNR metrics if enabled (separate instances for train/val/test)
        if self.use_multi_mel_snr:
            self.train_multi_mel_snr_metric = MultiMelSNRMetric(sample_rate=sampling_rate)
            self.val_multi_mel_snr_metric = MultiMelSNRMetric(sample_rate=sampling_rate)
            self.test_multi_mel_snr_metric = MultiMelSNRMetric(sample_rate=sampling_rate)
            # Accumulators for original multi_mel_snr function (non-parallelized)
            self.train_multi_mel_snr_original_values = []
            self.val_multi_mel_snr_original_values = []
            print("Multi-Mel-SNR metric enabled (separate instances for train/val/test)")
        else:
            self.train_multi_mel_snr_metric = None
            self.val_multi_mel_snr_metric = None
            self.test_multi_mel_snr_metric = None
            self.train_multi_mel_snr_original_values = []
            self.val_multi_mel_snr_original_values = []
        
        # Phased fine-tuning configuration
        # Check if model supports phased fine-tuning
        if hasattr(self.model, 'roformer') and hasattr(self.model.roformer, 'phased_fine_tuning'):
            self.phased_fine_tuning_enabled = self.model.roformer.phased_fine_tuning
            self.phased_fine_tuning_phase1_epochs = self.model.roformer.phased_fine_tuning_phase1_epochs
            if self.phased_fine_tuning_enabled:
                print(f"Phased fine-tuning enabled: Phase 1 = epochs 0-{self.phased_fine_tuning_phase1_epochs-1}, Phase 2 = epoch {self.phased_fine_tuning_phase1_epochs}+")
        else:
            self.phased_fine_tuning_enabled = False
            self.phased_fine_tuning_phase1_epochs = None

        self.msr_bench_validation = msr_bench_validation
        self.msr_bench_max_files = msr_bench_max_files
        self.msr_bench_window_duration = msr_bench_window_duration
        self.msr_bench_seed = msr_bench_seed
        
        # Initialize MSRBench validation accumulators
        if self.msr_bench_validation:
            # Per-source metric accumulators
            self.msr_bench_si_snr_values = {label: [] for label in self.columns} if self.columns else {}
            self.msr_bench_multi_mel_snr_values = {label: [] for label in self.columns} if self.columns else {}
            self.msr_bench_fad_clap_targets = {label: [] for label in self.columns} if self.columns else {}
            self.msr_bench_fad_clap_predictions = {label: [] for label in self.columns} if self.columns else {}
            # Overall accumulators
            self.msr_bench_all_si_snr_values = []
            self.msr_bench_all_multi_mel_snr_values = []
            # Loss accumulators for logging
            self.msr_bench_l1_loss_values = []
            self.msr_bench_mr_stft_loss_values = []
            self.msr_bench_mean_amp_values = []
            self.msr_bench_loss_values = []
            self.msr_bench_snr_values = []
            # Track selected file indices for deterministic selection
            self.msr_bench_selected_indices = None
            # Store original validation dataset to restore after creating subset
            self.msr_bench_original_val_dataset = None
            # Store original val_dataloader method for monkey-patching
            self.msr_bench_original_val_dataloader = None

    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to handle 4-stem to 9-stem checkpoint mapping.
        
        This is called when PyTorch Lightning loads a checkpoint during training resume.
        """
        # Check if we need to map 4-stem to 9-stem
        if self.n_sources == 9:
            # Detect if checkpoint has 4 mask_estimators and model has 9
            ckpt_mask_estimators = set()
            model_mask_estimators = set()
            
            for key in state_dict.keys():
                # Match both 'model.roformer.mask_estimators.X' and 'roformer.mask_estimators.X'
                match = re.search(r'(?:^|\.)mask_estimators\.(\d+)', key)
                if match:
                    ckpt_mask_estimators.add(int(match.group(1)))
            
            model_state = self.state_dict()
            for key in model_state.keys():
                match = re.search(r'(?:^|\.)mask_estimators\.(\d+)', key)
                if match:
                    model_mask_estimators.add(int(match.group(1)))
            
            if len(ckpt_mask_estimators) == 4 and len(model_mask_estimators) == 9:
                print("Detected 4-stem checkpoint loading into 9-stem model. Applying stem mapping...")
                print("  Mapping: vocals->vocals, bass->bass, drums->drums, other->other (9th stem)")
                print("  Initializing new stems (guitars, keyboards, synthesizers, percussions, orchestral elements) from 'other' stem")
                
                # Map the checkpoint
                # First, remove 'model.' prefix from state_dict keys for mapping function
                ckpt_no_prefix = {}
                prefix_map = {}  # Track prefix for each key
                for key, value in state_dict.items():
                    if key.startswith('model.'):
                        key_no_prefix = key[6:]  # Remove 'model.' prefix
                        ckpt_no_prefix[key_no_prefix] = value
                        prefix_map[key_no_prefix] = 'model.'
                    else:
                        ckpt_no_prefix[key] = value
                        prefix_map[key] = ''
                
                # Get model state without prefix
                model_state_no_prefix = {}
                for key, value in model_state.items():
                    if key.startswith('model.'):
                        model_state_no_prefix[key[6:]] = value
                    else:
                        model_state_no_prefix[key] = value
                
                # Apply mapping (only maps mask_estimator keys)
                mapped_ckpt_no_prefix = map_4stem_to_9stem_checkpoint(ckpt_no_prefix, model_state_no_prefix)
                
                # Restore prefixes and combine with original state_dict
                mapped_ckpt = state_dict.copy()  # Start with all original keys
                for key, value in mapped_ckpt_no_prefix.items():
                    # Only update mask_estimator keys that were mapped
                    if 'mask_estimators' in key:
                        prefix = prefix_map.get(key, 'model.' if key.startswith('roformer.') else '')
                        full_key = f'{prefix}{key}' if prefix else key
                        mapped_ckpt[full_key] = value
                
                state_dict = mapped_ckpt
        
        # Call parent load_state_dict
        return super().load_state_dict(state_dict, strict=strict)

    def create_mixtures(self, batch):
        # Store raw waveforms BEFORE augmentation for signal detection
        raw_waveforms = {}
        for column in self.columns:
            raw_waveforms[column] = batch[column]["waveform"].clone()
        # Also check for "other" if it exists in batch (for 9-stem models)
        if "other" in batch:
            raw_waveforms["other"] = batch["other"]["waveform"].clone()
        batch["_raw_waveforms"] = raw_waveforms
        
        processing_dtype = torch.float32
        mix = None
        clean_mix = None

        with autocast(enabled=False):
            for column in self.columns:
                waveform = batch[column]["waveform"]
                waveform_f32 = waveform.to(dtype=processing_dtype)

                if mix is None or clean_mix is None:
                    mix = torch.zeros_like(waveform_f32)
                    clean_mix = torch.zeros_like(waveform_f32)

                waveform_mask = batch_has_signal_lufs(waveform_f32, sample_rate=self.sampling_rate)

                if self.augment:
                    # Apply augmentation only to non-silent waveforms
                    if waveform_mask.any():
                        aug_waveform = self.apply_augmentation(
                            waveform_f32[waveform_mask], sample_rate=self.sampling_rate
                        )
                        # for silent waveforms, zero them out
                        if all(~waveform_mask):
                            aug_waveform = torch.zeros_like(waveform_f32)
                        elif any(~waveform_mask):
                            aug_waveform_full = torch.zeros_like(waveform_f32)
                            aug_waveform_full[waveform_mask] = aug_waveform
                            aug_waveform = aug_waveform_full
                    else:
                        # All waveforms are silent
                        aug_waveform = torch.zeros_like(waveform_f32)

                    mix += aug_waveform
                    batch[column]["aug_waveform"] = aug_waveform

                clean_mix += waveform_f32
                # lengths = batch[column]["lengths"]
                # paths = batch[column]["paths"]
                # present_flags = batch[column]["is_present"]
                batch[column]['is_present'] = waveform_mask.tolist()

            if self.augment:
                # Check if the mixture has signal before applying augmentation
                # mix_mask = batch_has_signal_lufs(mix, sample_rate=self.sampling_rate)
                # if mix_mask.any():
                #     mix = self.apply_augmentation(mix, sample_rate=self.sampling_rate)
                # If mixture is silent, keep it as zeros (no augmentation)

                # apply mastering
                mastered = self.mastering_chain(mix, sample_rate=self.sampling_rate)

                batch["mixture"] = mastered.to(dtype=processing_dtype)
            else:
                batch["mixture"] = clean_mix.to(dtype=processing_dtype)
        
        return batch

    def compute_frame_activity_mask(self, target_sources: torch.Tensor) -> torch.Tensor:
        """
        Compute per-source, per-frame activity mask from ground-truth.

        Args:
            target_sources: Tensor [B, S, C, T]

        Returns:
            frame_mask: Float tensor [B, S, 1, T] with 1.0 where active, else 0.0
        """
        B, S, C, T = target_sources.shape
        device = target_sources.device
        # mean across channels -> [B,S,1,T]
        mono = target_sources.mean(dim=2, keepdim=True)
        # moving RMS via avg_pool on squared signal
        win_samples = max(1, int(round(self.activity_window_ms * 1e-3 * self.sampling_rate)))
        # Ensure odd kernel so output length matches T with stride=1 after reflect pad
        if win_samples % 2 == 0:
            win_samples += 1
        if win_samples > 1:
            pad = (win_samples - 1) // 2
            x2 = mono.pow(2.0)
            # pool expects [N,C,L]; merge B and S into batch, keep channel=1
            x2_merged = x2.view(B * S, 1, T)
            x2_padded = F.pad(x2_merged, (pad, pad), mode='reflect')
            ma = F.avg_pool1d(x2_padded, kernel_size=win_samples, stride=1, padding=0)  # [B*S,1,T]
            rms = ma.sqrt().view(B, S, 1, T)
        else:
            rms = mono.abs()

        # Adaptive thresholds per source
        max_rms = rms.max(dim=-1, keepdim=True)[0]  # [B, S, 1, 1]
        
        # Relative threshold: x% of max energy
        relative_threshold = max_rms * self.activity_relative_threshold  # e.g., 0.01 = 1%
        
        # Absolute threshold: minimum energy level
        absolute_threshold = torch.full_like(relative_threshold, self.activity_threshold)
        
        # Use the higher of the two thresholds
        threshold = torch.max(relative_threshold, absolute_threshold)
        
        # Create mask
        mask = (rms >= threshold).to(target_sources.dtype)
        
        # Optional: require minimum number of active frames
        if self.activity_min_frames > 0:
            active_counts = mask.sum(dim=-1, keepdim=True)
            keep = (active_counts >= self.activity_min_frames).to(mask.dtype)
            mask = mask * keep
        
        return mask

    def should_show_in_progress_bar(self, metric_name: str) -> bool:
        """
        Determine if a metric should appear in the progress bar.
        
        Args:
            metric_name: Full name of the metric (e.g., 'step_train/mr_stft_loss', 'epoch_val/si_snr')
            
        Returns:
            bool: True if metric should appear in progress bar, False otherwise
        """
        if self.progress_bar_metrics is None:
            # Default behavior: show all metrics in progress bar
            return True
        else:
            # Only show metrics that are in the specified list
            return metric_name in self.progress_bar_metrics

    def encode_target_sources(self, batch):
        sources = []
        # Raw waveforms are already stored in batch["_raw_waveforms"] by create_mixtures
        
        for column in self.columns:
            waveform = batch[column][self.target_key]  # self.target_key is either "waveform" or "aug_waveform"
            sources.append(waveform)
        
        target_sources = torch.stack(sources, dim=1)  # [B, n_columns, C, T]

        if target_sources.shape[1] < self.n_sources:
            # Pad with zeros for missing stems
            n_padding = self.n_sources - target_sources.shape[1]
            B, _, C, T = target_sources.shape
            padding = torch.zeros(B, n_padding, C, T, dtype=target_sources.dtype, device=target_sources.device)
            target_sources = torch.cat([target_sources, padding], dim=1)
        
        batch["target_sources"] = target_sources
        return batch

    def training_step(self, batch_data_dict, batch_idx):
        """
        Process a training step.
        
        Args:
            batch_data_dict: Dictionary containing:
                - mixture: [batch_size, 2, waveform_length] stereo audio
                - target_sources: [batch_size, n_sources, 2, waveform_length] stereo audio
        
        Returns:
            batchsize: int
            loss_dict: dict of losses
        """
        batch_data_dict = self.create_mixtures(batch_data_dict)
        batch_data_dict = self.encode_target_sources(batch_data_dict)

        batchsize = batch_data_dict['mixture'].shape[0]
        device = batch_data_dict['mixture'].device

        # Resample mixture if enabled
        mixture = batch_data_dict['mixture']  # [bs, nch, wlen] at 48kHz
        input_dict = {'mixture': mixture}
        
        # Add label_vector for FiLM conditioning if enabled
        if self.use_film_conditioning:
            # Concatenate one-hot vectors for all sources: [n_sources, n_sources] -> [n_sources * n_sources]
            # Then repeat for batch: [bs, n_sources * n_sources]
            label_vector = self.label_vectors.flatten().unsqueeze(0).repeat(batchsize, 1)
            input_dict['label_vector'] = label_vector.to(device)

        if self.debug:
            output_dict = dict(
                waveform=batch_data_dict['target_sources'] - 0.1,
                source_mask=torch.ones(batch_data_dict['target_sources'].shape[0], 4,
                           device=batch_data_dict['target_sources'].device, dtype=batch_data_dict['target_sources'].dtype)
            )
        else:
            output_dict = self.model(input_dict) # output_dict['waveform']: [bs, n_sources, 2, wlen] (at 48kHz for dual-STFT, 44.1kHz if resampling enabled)

        # Build per-source presence mask [B, S]
        present_list = []
        all_columns = list(self.columns)
        
        for column in all_columns:
            if column in batch_data_dict:
                # stored as list[bool]
                present_flags = batch_data_dict[column]["is_present"]
                present_list.append(torch.tensor(present_flags, dtype=torch.float32, device=batch_data_dict['mixture'].device))
            else:
                # Column doesn't exist, treat as not present
                B = batch_data_dict['mixture'].shape[0]
                present_list.append(torch.zeros(B, dtype=torch.float32, device=batch_data_dict['mixture'].device))
        
        source_mask = torch.stack(present_list, dim=1)  # [B, n_columns or n_columns+1]
        
        # Pad source_mask if needed (shouldn't happen if we handled "other" correctly)
        if source_mask.shape[1] < self.n_sources:
            n_padding = self.n_sources - source_mask.shape[1]
            B = source_mask.shape[0]
            padding = torch.zeros(B, n_padding, dtype=source_mask.dtype, device=source_mask.device)
            source_mask = torch.cat([source_mask, padding], dim=1)  # [B, n_sources]

        # # True frame activity from GT per source, intersect with presence
        # frame_activity = self.compute_frame_activity_mask(batch_data_dict['target_sources'])  # [B,S,1,T]
        # frame_mask = frame_activity * source_mask[:, :, None, None]

        target_dict = {
            'waveform': batch_data_dict['target_sources'],
            'source_mask': source_mask,
            # 'frame_mask': frame_mask,
        }
        
        # Track stem presence statistics (is_present=True AND has actual signal)
        if self.stem_presence_counts is not None and self.columns is not None:
            target_sources = batch_data_dict['target_sources']  # [B, S, C, T]
            B, S, C, T = target_sources.shape
            
            # target_sources is stacked in the order of self.columns
            # So target_sources[:, stem_idx, :, :] corresponds to self.columns[stem_idx]
            for stem_idx, column in enumerate(self.columns):
                if stem_idx < S:
                    # Get is_present flags for this stem
                    if column in batch_data_dict:
                        present_flags = batch_data_dict[column]["is_present"]  # List[bool]
                    else:
                        present_flags = [False] * B
                    
                    # Get waveforms for this stem [B, C, T]
                    stem_waveforms = target_sources[:, stem_idx, :, :]
                    
                    # Check each sample in the batch
                    for batch_idx in range(B):
                        if present_flags[batch_idx]:
                            # Check if waveform has actual signal (not all zeros)
                            waveform = stem_waveforms[batch_idx]  # [C, T]
                            # Compute RMS to check if there's actual signal
                            # Use a small threshold to avoid numerical noise
                            rms = torch.sqrt(torch.mean(waveform ** 2))
                            signal_threshold = 1e-6  # Very small threshold to detect any meaningful signal
                            
                            if rms > signal_threshold:
                                self.stem_presence_counts[column] += 1
        
        loss_dict = self.loss_func(output_dict, target_dict)

        # Update FAD-CLAP metric for training
        if self.use_fad_clap and self.train_fad_clap_metric is not None:
            self.train_fad_clap_metric.update(output_dict, target_dict)
        
        # Update SI-SNR metric for training
        if self.use_si_snr and self.train_si_snr_metric is not None:
            self.train_si_snr_metric.update(output_dict, target_dict)
        
        # Update Multi-Mel-SNR metric for training
        if self.use_multi_mel_snr and self.train_multi_mel_snr_metric is not None:
            self.train_multi_mel_snr_metric.update(output_dict, target_dict)
            
            # Also calculate original multi_mel_snr function for comparison
            # Only compute for active stems
            output_wav = output_dict['waveform']  # [B, S, C, T]
            target_wav = target_dict['waveform']  # [B, S, C, T]
            B, S, C, T = output_wav.shape
            
            # Calculate original multi_mel_snr only for active stems
            for b in range(B):
                for s in range(S):
                    # Only compute if this source is active
                    if source_mask[b, s] > 0:
                        for c in range(C):
                            # Convert to mono numpy arrays for original multi_mel_snr function
                            target_mono = target_wav[b, s, c].cpu().numpy()  # [T]
                            output_mono = output_wav[b, s, c].detach().cpu().numpy()  # [T]
                            multi_mel_snr_val = multi_mel_snr(target_mono, output_mono, sr=self.sampling_rate)
                            self.train_multi_mel_snr_original_values.append(multi_mel_snr_val)

        loss = loss_dict['loss']

        # check if loss is nan
        if torch.isnan(loss_dict['loss']):
            print("loss is NaN!!!")
        # print("loss: ", loss_dict['loss'].item())

        # log all items in loss_dict
        epoc_dict = {f'epoch_train/{name}': val.item() for name, val in loss_dict.items()}
        step_dict = {f'step_train/{name}': val.item() for name, val in loss_dict.items()}
        
        # Log epoch metrics with appropriate progress bar settings
        for key, value in epoc_dict.items():
            show_in_progress_bar = self.should_show_in_progress_bar(key)
            self.log(key, value, prog_bar=show_in_progress_bar, logger=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batchsize)
        
        # Log step metrics with appropriate progress bar settings
        for key, value in step_dict.items():
            show_in_progress_bar = self.should_show_in_progress_bar(key)
            self.log(key, value, prog_bar=show_in_progress_bar, logger=True, on_epoch=False, on_step=True, sync_dist=True, batch_size=batchsize)

        self.log_dict({"epoch/lr": self.optimizer.param_groups[0]['lr']}, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch_data_dict, batch_idx):
        """
        Process a validation step.
        
        Args:
            batch_data_dict: Dictionary containing:
                - mixture: [batch_size, 2, waveform_length] stereo audio
                - target_sources: [batch_size, n_sources, 2, waveform_length] stereo audio
        
        Returns:
            batchsize: int
            loss_dict: dict of losses and metrics
        """
        # Use EMA parameters for validation if enabled
        if self.use_ema and self.ema is not None:
            self.ema.apply()

        if self.msr_bench_validation:
            # MSRBench validation: process samples with non-overlapping windows
            # batch_data_dict structure for MSRBench:
            # - 'mixture': {'waveform': [B, C, T], 'lengths': [B], 'paths': [B]}
            # - 'source': {'waveform': [B, C, T], 'lengths': [B], 'labels': [B], 'paths': [B]}
            # - 'index': [B] - indices in the dataset (these are now indices into the subset, not the original dataset)
            
            # Window size in samples
            window_samples = int(self.msr_bench_window_duration * self.sampling_rate)
            
            # Process each sample in the batch
            batch_indices = batch_data_dict.get('index', [])
            batch_size = len(batch_indices)
            device = batch_data_dict['mixture']['waveform'].device if 'mixture' in batch_data_dict and 'waveform' in batch_data_dict['mixture'] else torch.device('cpu')
            
            for batch_idx in range(batch_size):
                # Get mixture and source
                # Note: batch_indices contains indices into the subset (0 to N-1),
                # which automatically map to the original dataset indices via the Subset wrapper
                mixture = batch_data_dict['mixture']['waveform'][batch_idx]  # [C, T]
                source_gt = batch_data_dict['source']['waveform'][batch_idx]  # [C, T]
                source_label = batch_data_dict['source']['labels'][batch_idx]
                mixture_length = batch_data_dict['mixture']['lengths'][batch_idx].item()
                source_length = batch_data_dict['source']['lengths'][batch_idx].item()
                
                # Skip if source_label is not in model columns
                if self.columns is None or source_label not in self.columns:
                    continue
                
                # Get the index of this source in the model output
                source_idx = self.columns.index(source_label)
                
                # Trim to actual lengths
                mixture = mixture[:, :mixture_length]  # [C, mixture_length]
                source_gt = source_gt[:, :source_length]  # [C, source_length]
                min_length = min(mixture_length, source_length)
                
                # Segment into non-overlapping windows
                window_start = 0
                while window_start + window_samples <= min_length:
                    # Extract window
                    mixture_window = mixture[:, window_start:window_start + window_samples]  # [C, window_samples]
                    source_gt_window = source_gt[:, window_start:window_start + window_samples]  # [C, window_samples]
                    
                    # Move to device and add batch dimension
                    mixture_window = mixture_window.unsqueeze(0).to(device)  # [1, C, window_samples]
                    
                    # Run inference
                    input_dict = {'mixture': mixture_window}
                    if self.use_film_conditioning:
                        label_vector = self.label_vectors.flatten().unsqueeze(0).to(device)
                        input_dict['label_vector'] = label_vector
                    
                    with torch.no_grad():
                        output_dict = self.model(input_dict)
                    
                    separated_sources = output_dict['waveform']  # [1, n_sources, C, window_samples]
                    
                    # Extract prediction for this source
                    prediction = separated_sources[0, source_idx]  # [C, window_samples]
                    
                    # Move source_gt_window to device for loss computation
                    source_gt_window_device = source_gt_window.to(device)
                    
                    # Prepare output and target dicts for loss computation
                    # Need to reshape to [1, n_sources, C, T] format for loss function
                    # Create full output tensor with zeros for other sources
                    prediction_full = torch.zeros(1, self.n_sources, prediction.shape[0], prediction.shape[1], 
                                                  device=device, dtype=prediction.dtype)  # [1, n_sources, C, window_samples]
                    prediction_full[0, source_idx] = prediction
                    
                    # Create full target tensor with zeros for other sources
                    source_gt_full = torch.zeros(1, self.n_sources, source_gt_window_device.shape[0], source_gt_window_device.shape[1],
                                                 device=device, dtype=source_gt_window_device.dtype)  # [1, n_sources, C, window_samples]
                    source_gt_full[0, source_idx] = source_gt_window_device
                    
                    # Create source mask (only this source is active)
                    source_mask_window = torch.zeros(1, self.n_sources, device=device, dtype=prediction.dtype)
                    source_mask_window[0, source_idx] = 1.0
                    
                    # Compute loss function
                    output_dict_loss = {'waveform': prediction_full}
                    target_dict_loss = {
                        'waveform': source_gt_full,
                        'source_mask': source_mask_window
                    }
                    
                    with torch.no_grad():
                        loss_dict = self.loss_func(output_dict_loss, target_dict_loss)
                    
                    # Accumulate loss values
                    self.msr_bench_l1_loss_values.append(loss_dict['l1_loss'].item())
                    self.msr_bench_mr_stft_loss_values.append(loss_dict['mr_stft_loss'].item())
                    self.msr_bench_mean_amp_values.append(loss_dict['mean_amp'].item())
                    self.msr_bench_loss_values.append(loss_dict['loss'].item())
                    
                    # Move to CPU for metric calculation
                    prediction = prediction.cpu()
                    source_gt_window = source_gt_window.cpu()
                    
                    # Calculate SI-SNR
                    si_snr_val = si_snr(prediction, source_gt_window).mean()  # scalar
                    si_snr_val_item = si_snr_val.item()
                    
                    # Store SI-SNR
                    self.msr_bench_si_snr_values[source_label].append(si_snr_val_item)
                    self.msr_bench_all_si_snr_values.append(si_snr_val_item)
                    
                    # Calculate SNR
                    snr_val = snr(prediction, source_gt_window).mean()  # scalar
                    self.msr_bench_snr_values.append(snr_val.item())
                    
                    # Calculate Multi-Mel-SNR
                    # Convert to mono numpy arrays for multi_mel_snr
                    prediction_mono = prediction.mean(dim=0).cpu().numpy()  # [window_samples]
                    source_gt_mono = source_gt_window.mean(dim=0).cpu().numpy()  # [window_samples]
                    multi_mel_snr_val = multi_mel_snr(source_gt_mono, prediction_mono, sr=self.sampling_rate)
                    
                    # Store Multi-Mel-SNR
                    self.msr_bench_multi_mel_snr_values[source_label].append(multi_mel_snr_val)
                    self.msr_bench_all_multi_mel_snr_values.append(multi_mel_snr_val)
                    
                    # Store audio for FAD-CLAP (keep on CPU to save memory)
                    if self.use_fad_clap:
                        # Store as numpy arrays to save memory
                        self.msr_bench_fad_clap_targets[source_label].append(source_gt_window.cpu().numpy())
                        self.msr_bench_fad_clap_predictions[source_label].append(prediction.cpu().numpy())
                    
                    # Move to next window
                    window_start += window_samples
            
            # Return dummy loss (metrics computed in epoch_end)
            return torch.tensor(0.0, device=device, requires_grad=False)
            
        
        # Standard validation path
        batch_data_dict = self.create_mixtures(batch_data_dict)
        batch_data_dict = self.encode_target_sources(batch_data_dict)

        batchsize = batch_data_dict['mixture'].shape[0]
        device = batch_data_dict['mixture'].device

        # Resample mixture if enabled
        mixture = batch_data_dict['mixture']  # [bs, nch, wlen] at 48kHz
        input_dict = {
            'mixture': mixture, # [bs, nch, wlen] (at 48kHz for dual-STFT, 44.1kHz if resampling enabled, else original)
        }
        
        # Add label_vector for FiLM conditioning if enabled
        if self.use_film_conditioning:
            label_vector = self.label_vectors.flatten().unsqueeze(0).repeat(batchsize, 1)
            input_dict['label_vector'] = label_vector.to(device)

        output_dict = self.model(input_dict) # {'waveform': [bs, n_sources, 2, wlen] (at 48kHz for dual-STFT, 44.1kHz if resampling enabled)}
        
        # For validation, only use source presence mask, no frame-level masking
        present_list = []
        
        for column in list(self.columns):
            if column in batch_data_dict:
                present_flags = batch_data_dict[column]["is_present"]
                present_list.append(torch.tensor(present_flags, dtype=torch.float32, device=batch_data_dict['mixture'].device))
            else:
                # Column doesn't exist, treat as not present
                B = batch_data_dict['mixture'].shape[0]
                present_list.append(torch.zeros(B, dtype=torch.float32, device=batch_data_dict['mixture'].device))
        
        source_mask = torch.stack(present_list, dim=1)  # [B, n_columns or n_columns+1]
        
        # Pad source_mask if needed (shouldn't happen if we handled "other" correctly)
        if source_mask.shape[1] < self.n_sources:
            n_padding = self.n_sources - source_mask.shape[1]
            B = source_mask.shape[0]
            padding = torch.zeros(B, n_padding, dtype=source_mask.dtype, device=source_mask.device)
            source_mask = torch.cat([source_mask, padding], dim=1)  # [B, n_sources]

        
        # No frame masking in validation - evaluate on full audio
        target_dict = {
            'waveform': batch_data_dict['target_sources'],
            'source_mask': source_mask,
        }
        loss_dict = self.loss_func(output_dict, target_dict)

        # Update FAD-CLAP metric for validation (only computed once for whole batch)
        if self.use_fad_clap and self.val_fad_clap_metric is not None:
            self.val_fad_clap_metric.update(output_dict, target_dict)

        loss_dict = {k: v.item() for k,v in loss_dict.items()}

        # Compute metrics once per sample (overall metrics only)
        if self.use_si_snr and self.val_si_snr_metric is not None:
            output_wav = output_dict['waveform']  # [B, S, C, T]
            target_wav = target_dict['waveform']  # [B, S, C, T]
            B, S, C, T = output_wav.shape

            # Compute SI-SNR for all samples at once (vectorized)
            output_flat = output_wav.view(B * S, C, T)  # [B*S, C, T]
            target_flat = target_wav.view(B * S, C, T)  # [B*S, C, T]
            all_si_snr_values = si_snr(output_flat, target_flat)  # [B*S] or [B*S, C] if multi-channel
            # Handle multi-channel: average over channels if needed
            if all_si_snr_values.dim() > 1:
                all_si_snr_values = all_si_snr_values.mean(dim=-1)  # [B*S]

            # Route to global accumulator (for whole validation set)
            self.val_si_snr_metric.si_snr_values.extend(all_si_snr_values.cpu().tolist())

        # Compute Multi-Mel-SNR once per sample (overall metrics only)
        if self.use_multi_mel_snr and self.val_multi_mel_snr_metric is not None:
            output_wav = output_dict['waveform']  # [B, S, C, T]
            target_wav = target_dict['waveform']  # [B, S, C, T]
            B, S, C, T = output_wav.shape

            # Filter to only active stems using source_mask
            # source_mask is [B, S] where 1 means present, 0 means not present
            active_mask = source_mask.bool()  # [B, S]
            
            # Collect active stems across all batches
            active_outputs = []
            active_targets = []
            
            for b in range(B):
                # Get active sources for this batch
                batch_active_sources = active_mask[b]  # [S]
                if batch_active_sources.any():
                    # Extract only active sources: [num_active, C, T]
                    batch_output = output_wav[b][batch_active_sources]  # [num_active, C, T]
                    batch_target = target_wav[b][batch_active_sources]  # [num_active, C, T]
                    active_outputs.append(batch_output)
                    active_targets.append(batch_target)
            
            # If we have any active stems, compute the metric
            if len(active_outputs) > 0:
                # Concatenate all active stems: [total_active, C, T]
                output_active = torch.cat(active_outputs, dim=0)  # [total_active, C, T]
                target_active = torch.cat(active_targets, dim=0)  # [total_active, C, T]
                
                # Reshape to [total_active*C, T] for multi-mel-snr computation (processes each channel)
                output_flat = output_active.view(-1, T)  # [total_active*C, T]
                target_flat = target_active.view(-1, T)  # [total_active*C, T]

                # Compute Multi-Mel-SNR for all active channels at once (vectorized)
                all_multi_mel_snr_values = self.val_multi_mel_snr_metric._compute_batch_multi_mel_snr(
                    target_flat, output_flat
                )  # [total_active*C]

                # Route to global accumulator (for whole validation set)
                self.val_multi_mel_snr_metric.multi_mel_snr_values.extend(all_multi_mel_snr_values.cpu().tolist())
            
            # Also calculate original multi_mel_snr function for comparison
            # Calculate only for active stems
            for b in range(B):
                for s in range(S):
                    # Only compute if this source is active
                    if source_mask[b, s] > 0:
                        for c in range(C):
                            # Convert to mono numpy arrays for original multi_mel_snr function
                            target_mono = target_wav[b, s, c].cpu().numpy()  # [T]
                            output_mono = output_wav[b, s, c].cpu().numpy()  # [T]
                            multi_mel_snr_val = multi_mel_snr(target_mono, output_mono, sr=self.sampling_rate)
                            self.val_multi_mel_snr_original_values.append(multi_mel_snr_val)

        # Compute SNR if metric function is available
        if self.metric_func:
            output_wav = output_dict['waveform']  # [B, S, C, T]
            target_wav = target_dict['waveform']  # [B, S, C, T]
            B, S, C, T = output_wav.shape

            # Compute SNR for all samples at once (vectorized)
            output_flat = output_wav.view(B * S, C, T)  # [B*S, C, T]
            target_flat = target_wav.view(B * S, C, T)  # [B*S, C, T]
            all_snr_values = snr(output_flat, target_flat)  # [B*S] or [B*S, C] if multi-channel
            # Handle multi-channel: average over channels if needed
            if all_snr_values.dim() > 1:
                all_snr_values = all_snr_values.mean(dim=-1)  # [B*S]

            # Add SNR to loss_dict for logging (we computed it directly, so no need to call metric_func)
            loss_dict['snr'] = all_snr_values.mean().item()

        loss = loss_dict['loss']

        # log all items in loss_dict
        epoc_dict = {f'epoch_val/{name}': metric for name, metric in loss_dict.items()}
        step_dict = {f'step_val/{name}': metric for name, metric in loss_dict.items()}

        # Log epoch metrics with appropriate progress bar settings
        for key, value in epoc_dict.items():
            show_in_progress_bar = self.should_show_in_progress_bar(key)
            self.log(key, value, prog_bar=show_in_progress_bar, logger=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batchsize)

        # Log step metrics with appropriate progress bar settings
        for key, value in step_dict.items():
            show_in_progress_bar = self.should_show_in_progress_bar(key)
            self.log(key, value, prog_bar=show_in_progress_bar, logger=True, on_epoch=False, on_step=True, sync_dist=True, batch_size=batchsize)

        return loss

    def val_dataloader(self):
        """
        Override val_dataloader to use subset dataset for MSRBench validation.
        
        This ensures that when MSRBench validation is enabled, only the selected
        N samples are used, rather than iterating through all 26,000 samples.
        """
        if self.msr_bench_validation and hasattr(self.trainer.datamodule, 'val_dataset') and self.trainer.datamodule.val_dataset is not None:
            # Get original dataset (unwrap if it's already a subset)
            current_dataset = self.trainer.datamodule.val_dataset
            if isinstance(current_dataset, Subset):
                original_dataset = current_dataset.dataset
            else:
                original_dataset = current_dataset
            
            # Store reference to original dataset if not already stored
            if self.msr_bench_original_val_dataset is None:
                self.msr_bench_original_val_dataset = original_dataset
            
            # Select indices if not already selected
            if self.msr_bench_selected_indices is None:
                total_files = len(self.msr_bench_original_val_dataset)
                import random
                rng = random.Random(self.msr_bench_seed)
                all_indices = list(range(total_files))
                rng.shuffle(all_indices)
                
                if self.msr_bench_max_files is not None:
                    n_files = min(self.msr_bench_max_files, total_files)
                else:
                    n_files = total_files
                
                self.msr_bench_selected_indices = sorted(all_indices[:n_files])
                print(f"[MSRBench Validation] Selected {len(self.msr_bench_selected_indices)} files (seed={self.msr_bench_seed})")
            
            # Always create a fresh subset dataset
            subset_dataset = SubsetWithCollateFn(self.msr_bench_original_val_dataset, self.msr_bench_selected_indices)
            self.trainer.datamodule.val_dataset = subset_dataset
            
            # Create dataloader with subset dataset
            from torch.utils.data import DataLoader
            val_config = self.trainer.datamodule.val_config
            dataloader = DataLoader(
                dataset=subset_dataset,
                batch_size=val_config['batch_size'],
                collate_fn=subset_dataset.collate_fn,
                num_workers=val_config['num_workers'],
                pin_memory=True,
                persistent_workers=val_config['persistent_workers'],
                shuffle=False
            )
            print(f"[MSRBench Validation] Created dataloader with {len(subset_dataset)} samples (subset of {len(self.msr_bench_original_val_dataset)} total)")
            return dataloader
        else:
            # Use default datamodule dataloader
            return self.trainer.datamodule.val_dataloader()

    def on_fit_start(self):
        """Called at the very beginning of fit (before training starts)."""
        # Set dataset epoch to current_epoch when resuming (important for correct random sampling)
        if self.trainer.datamodule.train_dataset is not None:
            if hasattr(self.trainer.datamodule.train_dataset, 'set_epoch'):
                # When resuming, current_epoch will be > 0, so set it immediately
                # This ensures correct random sampling from the start
                initial_epoch = self.current_epoch
                print(f"\n[DataModule] on_fit_start called - setting initial dataset epoch to {initial_epoch}")
                self.trainer.datamodule.train_dataset.set_epoch(initial_epoch)
                print(f"[DataModule] Dataset epoch is now: {self.trainer.datamodule.train_dataset._epoch}")
        
        if self.trainer.datamodule.val_dataset is not None:
            if hasattr(self.trainer.datamodule.val_dataset, 'set_epoch'):
                initial_epoch = self.current_epoch
                self.trainer.datamodule.val_dataset.set_epoch(initial_epoch)
        
        # Monkey-patch datamodule's val_dataloader if MSRBench validation is enabled
        if self.msr_bench_validation and hasattr(self.trainer.datamodule, 'val_dataloader'):
            # Store original method
            if self.msr_bench_original_val_dataloader is None:
                self.msr_bench_original_val_dataloader = self.trainer.datamodule.val_dataloader
            
            # Create wrapper that uses our subset logic
            def wrapped_val_dataloader():
                # Get original dataset
                current_dataset = self.trainer.datamodule.val_dataset
                if isinstance(current_dataset, Subset):
                    original_dataset = current_dataset.dataset
                else:
                    original_dataset = current_dataset
                
                # Store reference to original dataset if not already stored
                if self.msr_bench_original_val_dataset is None:
                    self.msr_bench_original_val_dataset = original_dataset
                
                # Select indices if not already selected
                if self.msr_bench_selected_indices is None:
                    total_files = len(self.msr_bench_original_val_dataset)
                    import random
                    rng = random.Random(self.msr_bench_seed)
                    all_indices = list(range(total_files))
                    rng.shuffle(all_indices)
                    
                    if self.msr_bench_max_files is not None:
                        n_files = min(self.msr_bench_max_files, total_files)
                    else:
                        n_files = total_files
                    
                    self.msr_bench_selected_indices = sorted(all_indices[:n_files])
                    print(f"[MSRBench Validation] Selected {len(self.msr_bench_selected_indices)} files (seed={self.msr_bench_seed})")
                
                # Create subset dataset
                subset_dataset = SubsetWithCollateFn(self.msr_bench_original_val_dataset, self.msr_bench_selected_indices)
                self.trainer.datamodule.val_dataset = subset_dataset
                
                # Create dataloader with subset
                from torch.utils.data import DataLoader
                val_config = self.trainer.datamodule.val_config
                dataloader = DataLoader(
                    dataset=subset_dataset,
                    batch_size=val_config['batch_size'],
                    collate_fn=subset_dataset.collate_fn,
                    num_workers=val_config['num_workers'],
                    pin_memory=True,
                    persistent_workers=val_config['persistent_workers'],
                    shuffle=False
                )
                print(f"[MSRBench Validation] Created dataloader with {len(subset_dataset)} samples (subset of {len(self.msr_bench_original_val_dataset)} total)")
                return dataloader
            
            # Replace the method
            self.trainer.datamodule.val_dataloader = wrapped_val_dataloader

    def on_train_epoch_start(self):
        """Called at the beginning of each training epoch."""
        # Reset stem presence tracking counters for this epoch
        if self.columns is not None:
            self.stem_presence_counts = {column: 0 for column in self.columns}
        else:
            self.stem_presence_counts = {}
        
        # Update epoch in dataset for random offset resampling
        print(f"\n[DataModule] on_train_epoch_start called - current_epoch={self.current_epoch}")
        if hasattr(self.trainer.datamodule.train_dataset, 'set_epoch'):
            print(f"[DataModule] Setting dataset epoch to {self.current_epoch}")
            self.trainer.datamodule.train_dataset.set_epoch(self.current_epoch)
            print(f"[DataModule] Dataset epoch is now: {self.trainer.datamodule.train_dataset._epoch}")
        else:
            print(f"[DataModule] Dataset does not have set_epoch method")
        train_dataset = getattr(self.trainer.datamodule, "train_dataset", None)
        if train_dataset is not None and hasattr(train_dataset, "set_mix_probability"):
            start_prob, end_prob = getattr(train_dataset, "get_mix_probability_bounds", lambda: (0.9, 0.0))()
            total_epochs = getattr(self.trainer, "max_epochs", None)
            if total_epochs is None or total_epochs <= 1:
                new_prob = end_prob
            else:
                denom = max(total_epochs - 1, 1)
                progress = min(max(self.current_epoch / denom, 0.0), 1.0)
                new_prob = float(start_prob + (end_prob - start_prob) * progress)
            train_dataset.set_mix_probability(new_prob)
            self.log(
                "epoch_train/mix_probability",
                new_prob,
                prog_bar=False,
                logger=True,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )
        if self.trainer.datamodule.val_dataset is not None and hasattr(self.trainer.datamodule.val_dataset, 'set_epoch'):
            self.trainer.datamodule.val_dataset.set_epoch(self.current_epoch)
        
        # Handle phased fine-tuning phase transitions
        if self.phased_fine_tuning_enabled and hasattr(self.model, 'roformer') and hasattr(self.model.roformer, 'set_phase'):
            if self.phased_fine_tuning_phase1_epochs is not None:
                if self.current_epoch < self.phased_fine_tuning_phase1_epochs:
                    # Phase 1: Lower-frequency only
                    if self.model.roformer.current_phase != 1:
                        print(f"\n[Phased Fine-Tuning] Switching to Phase 1 (epoch {self.current_epoch}/{self.phased_fine_tuning_phase1_epochs-1})")
                        self.model.roformer.set_phase(1)
                else:
                    # Phase 2: Full frequency range
                    if self.model.roformer.current_phase != 2:
                        print(f"\n[Phased Fine-Tuning] Switching to Phase 2 (epoch {self.current_epoch}+)")
                        self.model.roformer.set_phase(2)

    def on_train_epoch_end(self):
        """Compute and log FAD-CLAP and SI-SNR metrics at the end of training epoch."""
        # print(f"🟢 TRAIN END - Epoch {self.current_epoch}")

        key_fad_clap = None
        key_si_snr = None

        value_fad_clap = 0.0
        value_si_snr = 0.0

        if self.use_fad_clap and self.train_fad_clap_metric is not None:
            # Only compute if we have collected data
            if self.train_fad_clap_metric.sample_count > 0:
                fad_clap_score = self.train_fad_clap_metric.compute()
                key_fad_clap = "fad_clap"
                value_fad_clap = fad_clap_score[key_fad_clap]
                # Log to logger always, but only show in progress bar if configured
                self.log(f'epoch_train/{key_fad_clap}', value_fad_clap, 
                        prog_bar=self.should_show_in_progress_bar(key_fad_clap), 
                        logger=True, on_epoch=True, sync_dist=True)
            self.train_fad_clap_metric.reset()
        
        if self.use_si_snr and self.train_si_snr_metric is not None:
            # Only compute if we have collected data
            if len(self.train_si_snr_metric.si_snr_values) > 0:
                si_snr_score = self.train_si_snr_metric.compute()
                key_si_snr = "si_snr"
                value_si_snr = si_snr_score[key_si_snr]
                # Log to logger always, but only show in progress bar if configured
                self.log(f'epoch_train/{key_si_snr}', value_si_snr, 
                        prog_bar=self.should_show_in_progress_bar(key_si_snr), 
                        logger=True, on_epoch=True, sync_dist=True)
            self.train_si_snr_metric.reset()
        
        if self.use_multi_mel_snr and self.train_multi_mel_snr_metric is not None:
            # Only compute if we have collected data
            if len(self.train_multi_mel_snr_metric.multi_mel_snr_values) > 0:
                multi_mel_snr_score = self.train_multi_mel_snr_metric.compute()
                key_multi_mel_snr = "multi_mel_snr"
                value_multi_mel_snr = multi_mel_snr_score[key_multi_mel_snr]
                # Log to logger always, but only show in progress bar if configured
                self.log(f'epoch_train/{key_multi_mel_snr}', value_multi_mel_snr, 
                        prog_bar=self.should_show_in_progress_bar(key_multi_mel_snr), 
                        logger=True, on_epoch=True, sync_dist=True)
            self.train_multi_mel_snr_metric.reset()
            
            # Calculate and log original multi_mel_snr function
            if len(self.train_multi_mel_snr_original_values) > 0:
                import numpy as np
                avg_multi_mel_snr_original = np.mean(self.train_multi_mel_snr_original_values)
                self.log(f'epoch_train/multi_mel_snr_original', avg_multi_mel_snr_original,
                        prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
                self.train_multi_mel_snr_original_values = []  # Reset for next epoch
        
        # Log stem presence statistics
        if self.stem_presence_counts is not None:
            for column, count in self.stem_presence_counts.items():
                # Log with prefix 'stem_stats/' to group in wandb
                self.log(
                    f'stem_stats/training_samples/{column}',
                    float(count),
                    prog_bar=False,
                    logger=True,
                    on_epoch=True,
                    on_step=False,
                    sync_dist=True,
                )

        # print()
        # if key_fad_clap is not None and key_si_snr is not None:
        #     print(f"Epoch {self.current_epoch} training finished: {key_fad_clap}: {value_fad_clap:.3f}, {key_si_snr}: {value_si_snr:.3f}")
        # elif key_fad_clap is not None:
        #     print(f"Epoch {self.current_epoch} training finished: {key_fad_clap}: {value_fad_clap:.3f}")
        # elif key_si_snr is not None:
        #     print(f"Epoch {self.current_epoch} training finished: {key_si_snr}: {value_si_snr:.3f}")
        # print()

    def on_validation_epoch_start(self):
        """Reset metric accumulators at the start of validation epoch."""
        if self.msr_bench_validation:
            # Reset MSRBench accumulators
            self.msr_bench_si_snr_values = {label: [] for label in self.columns} if self.columns else {}
            self.msr_bench_multi_mel_snr_values = {label: [] for label in self.columns} if self.columns else {}
            self.msr_bench_fad_clap_targets = {label: [] for label in self.columns} if self.columns else {}
            self.msr_bench_fad_clap_predictions = {label: [] for label in self.columns} if self.columns else {}
            self.msr_bench_all_si_snr_values = []
            self.msr_bench_all_multi_mel_snr_values = []
            # Reset loss accumulators
            self.msr_bench_l1_loss_values = []
            self.msr_bench_mr_stft_loss_values = []
            self.msr_bench_mean_amp_values = []
            self.msr_bench_loss_values = []
            self.msr_bench_snr_values = []
            
            # Ensure we have the original dataset reference and selected indices
            if hasattr(self.trainer.datamodule, 'val_dataset') and self.trainer.datamodule.val_dataset is not None:
                val_dataset = self.trainer.datamodule.val_dataset
                
                # If we have a subset from a previous epoch, unwrap it to get the original dataset
                if isinstance(val_dataset, Subset):
                    val_dataset = val_dataset.dataset
                
                # Store reference to original dataset if not already stored
                if self.msr_bench_original_val_dataset is None:
                    self.msr_bench_original_val_dataset = val_dataset
                
                # Select indices if not already selected
                if self.msr_bench_selected_indices is None:
                    total_files = len(self.msr_bench_original_val_dataset)
                    import random
                    rng = random.Random(self.msr_bench_seed)
                    all_indices = list(range(total_files))
                    rng.shuffle(all_indices)
                    
                    if self.msr_bench_max_files is not None:
                        n_files = min(self.msr_bench_max_files, total_files)
                    else:
                        n_files = total_files
                    
                    self.msr_bench_selected_indices = sorted(all_indices[:n_files])
                    print(f"[MSRBench Validation] Selected {len(self.msr_bench_selected_indices)} files (seed={self.msr_bench_seed})")
                
                # Clear trainer's cached dataloader to force recreation
                # PyTorch Lightning caches dataloaders, so we need to clear it
                if hasattr(self.trainer, '_data_connector'):
                    if hasattr(self.trainer._data_connector, '_val_dataloader'):
                        self.trainer._data_connector._val_dataloader = None
                # Also try clearing from trainer directly
                if hasattr(self.trainer, '_val_dataloaders'):
                    self.trainer._val_dataloaders = None
            else:
                self.msr_bench_selected_indices = None

    def on_validation_epoch_end(self):
        """Compute and log FAD-CLAP and SI-SNR metrics at the end of validation epoch."""
        if self.msr_bench_validation:
            # MSRBench validation: compute and log metrics
            
            # Log loss metrics
            if len(self.msr_bench_l1_loss_values) > 0:
                avg_l1_loss = torch.tensor(self.msr_bench_l1_loss_values).mean().item()
                self.log(f'epoch_val/msr_bench/l1_loss', avg_l1_loss,
                        prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
            
            if len(self.msr_bench_mr_stft_loss_values) > 0:
                avg_mr_stft_loss = torch.tensor(self.msr_bench_mr_stft_loss_values).mean().item()
                self.log(f'epoch_val/msr_bench/mr_stft_loss', avg_mr_stft_loss,
                        prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
            
            if len(self.msr_bench_mean_amp_values) > 0:
                avg_mean_amp = torch.tensor(self.msr_bench_mean_amp_values).mean().item()
                self.log(f'epoch_val/msr_bench/mean_amp', avg_mean_amp,
                        prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
            
            if len(self.msr_bench_loss_values) > 0:
                avg_loss = torch.tensor(self.msr_bench_loss_values).mean().item()
                self.log(f'epoch_val/msr_bench/loss', avg_loss,
                        prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
            
            if len(self.msr_bench_snr_values) > 0:
                avg_snr = torch.tensor(self.msr_bench_snr_values).mean().item()
                self.log(f'epoch_val/msr_bench/snr', avg_snr,
                        prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
            
            # Compute per-source and overall SI-SNR
            if self.columns is not None:
                for label in self.columns:
                    if label in self.msr_bench_si_snr_values and len(self.msr_bench_si_snr_values[label]) > 0:

                        if self.current_epoch == 0:
                            print(f"Validation samples for label {label}:", len(self.msr_bench_si_snr_values[label]))

                        avg_si_snr = torch.tensor(self.msr_bench_si_snr_values[label]).mean().item()
                        self.log(f'epoch_val/msr_bench/{label}/si_snr', avg_si_snr,
                                prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
                    
                    if label in self.msr_bench_multi_mel_snr_values and len(self.msr_bench_multi_mel_snr_values[label]) > 0:
                        avg_multi_mel_snr = torch.tensor(self.msr_bench_multi_mel_snr_values[label]).mean().item()
                        self.log(f'epoch_val/msr_bench/{label}/multi_mel_snr', avg_multi_mel_snr,
                                prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
            
            # Compute overall SI-SNR
            if len(self.msr_bench_all_si_snr_values) > 0:
                overall_si_snr = torch.tensor(self.msr_bench_all_si_snr_values).mean().item()
                self.log(f'epoch_val/msr_bench/si_snr', overall_si_snr,
                        prog_bar=self.should_show_in_progress_bar('msr_bench/si_snr'),
                        logger=True, on_epoch=True, sync_dist=True)
            
            # Compute overall Multi-Mel-SNR
            if len(self.msr_bench_all_multi_mel_snr_values) > 0:
                overall_multi_mel_snr = torch.tensor(self.msr_bench_all_multi_mel_snr_values).mean().item()
                self.log(f'epoch_val/msr_bench/multi_mel_snr', overall_multi_mel_snr,
                        prog_bar=self.should_show_in_progress_bar('msr_bench/multi_mel_snr'),
                        logger=True, on_epoch=True, sync_dist=True)
            
            # Compute FAD-CLAP if enabled
            if self.use_fad_clap and self.val_fad_clap_metric is not None:
                # Get CLAP embeddings from stored audio arrays
                try:
                    from src.evaluate import get_clap_embeddings_from_arrays, calculate_frechet_distance
                    from transformers import ClapModel, ClapProcessor
                    
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
                    clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
                    clap_model.eval()
                    clap_model.to(device)
                    
                    # Compute FAD-CLAP per source
                    for label in self.columns if self.columns else []:
                        if label in self.msr_bench_fad_clap_targets and len(self.msr_bench_fad_clap_targets[label]) > 0:
                            try:
                                target_embeddings = get_clap_embeddings_from_arrays(
                                    self.msr_bench_fad_clap_targets[label], clap_model, clap_processor,
                                    device, self.fad_clap_batch_size
                                )
                                pred_embeddings = get_clap_embeddings_from_arrays(
                                    self.msr_bench_fad_clap_predictions[label], clap_model, clap_processor,
                                    device, self.fad_clap_batch_size
                                )
                                
                                if target_embeddings.size > 0 and pred_embeddings.size > 0:
                                    fad_score = calculate_frechet_distance(target_embeddings, pred_embeddings)
                                    if fad_score is not None:
                                        self.log(f'epoch_val/msr_bench/{label}/fad_clap', fad_score,
                                                prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
                            except Exception as e:
                                print(f"Warning: Failed to calculate FAD-CLAP for {label}: {e}")
                    
                    # Compute overall FAD-CLAP
                    try:
                        all_targets = []
                        all_predictions = []
                        for label in self.columns if self.columns else []:
                            if label in self.msr_bench_fad_clap_targets:
                                all_targets.extend(self.msr_bench_fad_clap_targets[label])
                                all_predictions.extend(self.msr_bench_fad_clap_predictions[label])
                        
                        if len(all_targets) > 0:
                            target_embeddings = get_clap_embeddings_from_arrays(
                                all_targets, clap_model, clap_processor, device, self.fad_clap_batch_size
                            )
                            pred_embeddings = get_clap_embeddings_from_arrays(
                                all_predictions, clap_model, clap_processor, device, self.fad_clap_batch_size
                            )
                            
                            if target_embeddings.size > 0 and pred_embeddings.size > 0:
                                overall_fad = calculate_frechet_distance(target_embeddings, pred_embeddings)
                                if overall_fad is not None:
                                    self.log(f'epoch_val/msr_bench/fad_clap', overall_fad,
                                            prog_bar=self.should_show_in_progress_bar('msr_bench/fad_clap'),
                                            logger=True, on_epoch=True, sync_dist=True)
                    except Exception as e:
                        print(f"Warning: Failed to calculate overall FAD-CLAP: {e}")
                except Exception as e:
                    print(f"Warning: Failed to load CLAP model for FAD-CLAP: {e}")
            
            return
        
        # Standard validation: compute and log FAD-CLAP and SI-SNR metrics
        # print(f"🔵 VAL END - Epoch {self.current_epoch}")

        key_fad_clap = None
        key_si_snr = None

        value_fad_clap = 0.0
        value_si_snr = 0.0

        if self.use_fad_clap and self.val_fad_clap_metric is not None:
            # Only compute if we have collected data
            if self.val_fad_clap_metric.sample_count > 0:
                fad_clap_score = self.val_fad_clap_metric.compute()
                key_fad_clap = "fad_clap"
                value_fad_clap = fad_clap_score[key_fad_clap]
                # Log to logger always, but only show in progress bar if configured
                self.log(f'epoch_val/{key_fad_clap}', value_fad_clap, 
                        prog_bar=self.should_show_in_progress_bar(key_fad_clap), 
                        logger=True, on_epoch=True, sync_dist=True)
            self.val_fad_clap_metric.reset()
        
        if self.use_si_snr and self.val_si_snr_metric is not None:
            # Only compute if we have collected data
            if len(self.val_si_snr_metric.si_snr_values) > 0:
                si_snr_score = self.val_si_snr_metric.compute()
                key_si_snr = "si_snr"
                value_si_snr = si_snr_score[key_si_snr]
                # Log to logger always, but only show in progress bar if configured
                self.log(f'epoch_val/{key_si_snr}', value_si_snr, 
                        prog_bar=self.should_show_in_progress_bar(key_si_snr), 
                        logger=True, on_epoch=True, sync_dist=True)
            self.val_si_snr_metric.reset()
        
        # Log Multi-Mel-SNR metric for validation
        key_multi_mel_snr = None
        value_multi_mel_snr = 0.0
        
        if self.use_multi_mel_snr and self.val_multi_mel_snr_metric is not None:
            # Only compute if we have collected data
            if len(self.val_multi_mel_snr_metric.multi_mel_snr_values) > 0:
                multi_mel_snr_score = self.val_multi_mel_snr_metric.compute()
                key_multi_mel_snr = "multi_mel_snr"
                value_multi_mel_snr = multi_mel_snr_score[key_multi_mel_snr]
                # Log to logger always, but only show in progress bar if configured
                self.log(f'epoch_val/{key_multi_mel_snr}', value_multi_mel_snr, 
                        prog_bar=self.should_show_in_progress_bar(key_multi_mel_snr), 
                        logger=True, on_epoch=True, sync_dist=True)
            self.val_multi_mel_snr_metric.reset()
            
            # Calculate and log original multi_mel_snr function
            if len(self.val_multi_mel_snr_original_values) > 0:
                import numpy as np
                avg_multi_mel_snr_original = np.mean(self.val_multi_mel_snr_original_values)
                self.log(f'epoch_val/multi_mel_snr_original', avg_multi_mel_snr_original,
                        prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
                self.val_multi_mel_snr_original_values = []  # Reset for next epoch

        # print the metrics
        print("")
        print("")
        metrics_parts = []
        if key_fad_clap is not None:
            metrics_parts.append(f"{key_fad_clap}: {value_fad_clap:.3f}")
        if key_si_snr is not None:
            metrics_parts.append(f"{key_si_snr}: {value_si_snr:.3f}")
        if key_multi_mel_snr is not None:
            metrics_parts.append(f"{key_multi_mel_snr}: {value_multi_mel_snr:.3f}")
        if metrics_parts:
            print(f"Epoch {self.current_epoch} validation finished: {', '.join(metrics_parts)}")
        print("")
        print("")

    def _process_full_song_in_chunks(self, mixture: torch.Tensor, chunk_length_samples: int = 10 * 48000) -> torch.Tensor:
        """
        Process a full song by splitting into chunks, running inference, and reconstructing.
        
        Args:
            mixture: Full song mixture [2, T] (stereo)
            chunk_length_samples: Length of each chunk in samples (default: 6 seconds at 48kHz)
        
        Returns:
            Separated sources [n_sources, 2, T]
        """
        device = mixture.device
        n_channels, total_length = mixture.shape
        
        chunk_length = chunk_length_samples
        overlap_samples = int(chunk_length * self.test_chunk_overlap)
        hop_length = chunk_length - overlap_samples
        
        # Calculate number of chunks needed
        if overlap_samples == 0:
            # Non-overlapping case
            n_chunks = (total_length + chunk_length - 1) // chunk_length
        else:
            # Overlapping case
            n_chunks = max(1, (total_length - overlap_samples + hop_length - 1) // hop_length)
        
        # Initialize output tensor
        separated_full = torch.zeros((self.n_sources, n_channels, total_length), 
                                     dtype=mixture.dtype, device=device)
        
        # Create window for blending overlapping chunks (Hanning window)
        if overlap_samples > 0:
            # Window for overlap region
            overlap_window = torch.hann_window(2 * overlap_samples, device=device)
            fade_in = overlap_window[:overlap_samples]
            fade_out = overlap_window[overlap_samples:]
            # Create full window for the chunk
            chunk_window = torch.ones(chunk_length, device=device)
            chunk_window[:overlap_samples] = fade_in
            chunk_window[-overlap_samples:] = fade_out
            chunk_window = chunk_window.unsqueeze(0).unsqueeze(0)  # [1, 1, chunk_length]
        else:
            chunk_window = None
        
        # Track accumulated weights for blending
        if overlap_samples > 0:
            weight_accumulator = torch.zeros(total_length, device=device)
        
        for i in range(n_chunks):
            if overlap_samples == 0:
                # Non-overlapping case
                start_idx = i * chunk_length
                end_idx = min(start_idx + chunk_length, total_length)
            else:
                # Overlapping case
                start_idx = i * hop_length
                end_idx = min(start_idx + chunk_length, total_length)
            
            # Extract chunk [2, chunk_length]
            chunk = mixture[:, start_idx:end_idx]
            
            # Pad if necessary (for last chunk)
            if chunk.shape[-1] < chunk_length:
                pad_length = chunk_length - chunk.shape[-1]
                chunk = F.pad(chunk, (0, pad_length), mode='constant', value=0.0)
            
            # Add batch dimension: [1, 2, chunk_length]
            chunk = chunk.unsqueeze(0)
            
            # Prepare input dict
            input_dict = {'mixture': chunk}
            
            # Run model
            if self.use_film_conditioning:
                label_vector = self.label_vectors.flatten().unsqueeze(0)
                input_dict['label_vector'] = label_vector.to(device)
            
            with torch.no_grad():
                output_dict = self.model(input_dict)
            
            # Remove batch dimension: [n_sources, 2, chunk_length]
            separated_chunk = output_dict['waveform'][0]
            
            # Trim to actual length (if it was padded)
            actual_length = end_idx - start_idx
            separated_chunk = separated_chunk[..., :actual_length]
            
            # Apply window if using overlap
            if chunk_window is not None:
                window = chunk_window[:, :, :actual_length]
                separated_chunk = separated_chunk * window
            
            # Add to output with proper indexing
            if overlap_samples == 0:
                # Simple concatenation for non-overlapping
                separated_full[:, :, start_idx:end_idx] = separated_chunk
            else:
                # Weighted addition for overlapping
                chunk_end = min(start_idx + actual_length, total_length)
                separated_full[:, :, start_idx:chunk_end] += separated_chunk[:, :, :(chunk_end - start_idx)]
                
                # Update weight accumulator
                weight_end = min(start_idx + actual_length, total_length)
                if chunk_window is not None:
                    window_1d = window[0, 0, :(weight_end - start_idx)]
                    weight_accumulator[start_idx:weight_end] += window_1d
        
        # Normalize by weights if using overlap
        if overlap_samples > 0 and weight_accumulator.max() > 0:
            # Avoid division by zero
            weight_accumulator = torch.clamp(weight_accumulator, min=1e-8)
            separated_full = separated_full / weight_accumulator.unsqueeze(0).unsqueeze(0)
        
        return separated_full
    
    def _compute_frame_metrics(self, estimated: torch.Tensor, target: torch.Tensor, frame_length_samples: int = 48000, include_multi_mel_snr: bool = False) -> List[Dict[str, Any]]:
        """
        Split audio into 1-second non-overlapping frames and compute metrics per frame.
        
        Args:
            estimated: Estimated audio [n_sources, n_channels, T]
            target: Target audio [n_sources, n_channels, T]
            frame_length_samples: Length of each frame in samples (default: 1 second at 48kHz)
            include_multi_mel_snr: If True, also compute Multi-Mel-SNR (default: False)
        
        Returns:
            List of metric values (one per frame per source)
        """
        n_sources, n_channels, total_length = estimated.shape
        frame_length = frame_length_samples
        
        # Split into non-overlapping 1-second frames
        n_frames = total_length // frame_length
        
        metric_values = []
        
        # Initialize Multi-Mel-SNR metric if needed
        multi_mel_snr_metric = None
        if include_multi_mel_snr:
            multi_mel_snr_metric = MultiMelSNRMetric(sample_rate=self.sampling_rate)
        
        for frame_idx in range(n_frames):
            start_idx = frame_idx * frame_length
            end_idx = start_idx + frame_length
            
            # Extract frame: [n_sources, n_channels, frame_length]
            estimated_frame = estimated[:, :, start_idx:end_idx]
            target_frame = target[:, :, start_idx:end_idx]
            
            # Compute metrics for each source in this frame
            for source_idx in range(n_sources):
                # Flatten to [n_channels, frame_length]
                est_source = estimated_frame[source_idx]
                tgt_source = target_frame[source_idx]
                
                # Compute SNR and SI-SNR for this frame
                # Reshape to [1, n_channels, frame_length] for torchmetrics
                est_reshaped = est_source.unsqueeze(0)
                tgt_reshaped = tgt_source.unsqueeze(0)
                
                frame_snr_result = snr(est_reshaped, tgt_reshaped)
                # Handle multi-channel: average over channels if needed
                if frame_snr_result.dim() > 1:
                    frame_snr_result = frame_snr_result.mean(dim=-1)
                frame_snr = frame_snr_result.item()
                
                frame_si_snr_result = si_snr(est_reshaped, tgt_reshaped)
                # Handle multi-channel: average over channels if needed
                if frame_si_snr_result.dim() > 1:
                    frame_si_snr_result = frame_si_snr_result.mean(dim=-1)
                frame_si_snr = frame_si_snr_result.item()
                
                metric_dict = {
                    'snr': frame_snr,
                    'si_snr': frame_si_snr,
                    'source_idx': source_idx,
                }
                
                # Compute Multi-Mel-SNR if requested
                if include_multi_mel_snr and multi_mel_snr_metric is not None:
                    # Reshape to [n_channels, frame_length] for multi-mel-snr
                    # Multi-Mel-SNR processes each channel, so we need [C, T]
                    est_for_mmsnr = est_source  # [C, frame_length]
                    tgt_for_mmsnr = tgt_source  # [C, frame_length]
                    
                    # Compute Multi-Mel-SNR for each channel and average
                    # Reshape to [C*1, frame_length] to process all channels
                    est_flat = est_for_mmsnr.view(-1, frame_length)  # [C, frame_length]
                    tgt_flat = tgt_for_mmsnr.view(-1, frame_length)  # [C, frame_length]
                    
                    # Compute Multi-Mel-SNR for all channels
                    frame_multi_mel_snr_values = multi_mel_snr_metric._compute_batch_multi_mel_snr(
                        tgt_flat, est_flat
                    )  # [C]
                    
                    # Average over channels
                    frame_multi_mel_snr = frame_multi_mel_snr_values.mean().item()
                    metric_dict['multi_mel_snr'] = frame_multi_mel_snr
                
                metric_values.append(metric_dict)
        
        return metric_values

    def test_step(self, batch_data_dict, batch_idx):
        """
        Process a test step with full songs.
        
        This implements the standard MUSDB18-HQ evaluation procedure:
        1. Process full songs (chunked if needed)
        2. Split into 1-second non-overlapping frames
        3. Compute metrics per frame
        4. Accumulate for median computation at epoch end
        """
        # Use EMA parameters for testing if enabled
        if self.use_ema and self.ema is not None:
            self.ema.apply()
        
        batch_data_dict = self.create_mixtures(batch_data_dict)
        batch_data_dict = self.encode_target_sources(batch_data_dict)
        
        batch_size = batch_data_dict['mixture'].shape[0]
        device = batch_data_dict['mixture'].device
        
        # Initialize accumulators if not already done
        if not hasattr(self, 'test_metrics'):
            # Structure: source_idx -> metric_type -> list of values
            self.test_metrics = {
                source_idx: {'snr': [], 'si_snr': [], 'multi_mel_snr': []} for source_idx in range(self.n_sources)
            }
        
        # Process each sample in the batch
        # Get actual lengths for each sample (from mixture or first source)
        # The lengths should be the same for all sources in a sample
        if 'vocals' in batch_data_dict and 'lengths' in batch_data_dict['vocals']:
            sample_lengths = batch_data_dict['vocals']['lengths']
        else:
            # Fallback: use full mixture length
            sample_lengths = [batch_data_dict['mixture'].shape[-1]] * batch_size
        
        for batch_idx_sample in range(batch_size):
            # Extract full song mixture: [2, T] and trim to actual length
            actual_length = sample_lengths[batch_idx_sample]
            mixture = batch_data_dict['mixture'][batch_idx_sample, :, :actual_length]
            
            # Extract target sources: [n_sources, 2, T] and trim to actual length
            target_sources = batch_data_dict['target_sources'][batch_idx_sample, :, :, :actual_length]
            
            # Process full song (chunked if needed)
            separated_sources = self._process_full_song_in_chunks(mixture)
            
            # Trim separated sources to actual length (in case of padding)
            if separated_sources.shape[-1] > actual_length:
                separated_sources = separated_sources[:, :, :actual_length]
            elif separated_sources.shape[-1] < actual_length:
                # Pad if shorter (shouldn't happen, but handle it)
                pad_length = actual_length - separated_sources.shape[-1]
                separated_sources = F.pad(separated_sources, (0, pad_length), mode='constant', value=0.0)
            
            # Ensure target_sources matches separated_sources length
            min_length = min(separated_sources.shape[-1], target_sources.shape[-1])
            separated_sources = separated_sources[:, :, :min_length]
            target_sources = target_sources[:, :, :min_length]
            
            # Compute metrics per 1-second frame
            # Always compute Multi-Mel-SNR for test set
            frame_metrics = self._compute_frame_metrics(separated_sources, target_sources, include_multi_mel_snr=True)
            
            # Route metrics to appropriate source accumulator
            for metric_dict in frame_metrics:
                source_idx = metric_dict['source_idx']
                if source_idx < self.n_sources:
                    self.test_metrics[source_idx]['snr'].append(metric_dict['snr'])
                    self.test_metrics[source_idx]['si_snr'].append(metric_dict['si_snr'])
                    if 'multi_mel_snr' in metric_dict:
                        self.test_metrics[source_idx]['multi_mel_snr'].append(metric_dict['multi_mel_snr'])

    def on_test_epoch_start(self):
        """Initialize test metric accumulators."""
        # Structure: source_idx -> metric_type -> list of values
        self.test_metrics = {
            source_idx: {'snr': [], 'si_snr': [], 'multi_mel_snr': []} for source_idx in range(self.n_sources)
        }

    def on_test_epoch_end(self):
        """Compute and log median metrics per source and overall."""
        print("\n" + "="*80)
        print("TEST RESULTS (Standard Evaluation Procedure)")
        print("="*80)
        
        # Get source names: use LABELS for first len(LABELS) sources, then generic names
        source_names = []
        for i in range(self.n_sources):
            if i < len(LABELS):
                source_names.append(LABELS[i])
            else:
                source_names.append(f"source_{i}")
        
        # Compute per-source metrics
        all_snr_values = []
        all_si_snr_values = []
        all_multi_mel_snr_values = []
        
        for source_idx in range(self.n_sources):
            source_name = source_names[source_idx]
            snr_values = self.test_metrics[source_idx]['snr']
            si_snr_values = self.test_metrics[source_idx]['si_snr']
            multi_mel_snr_values = self.test_metrics[source_idx].get('multi_mel_snr', [])
            
            if len(snr_values) > 0:
                all_snr_values.extend(snr_values)
                all_si_snr_values.extend(si_snr_values)
                
                # Compute median for this source
                snr_median = torch.tensor(snr_values).median().item()
                si_snr_median = torch.tensor(si_snr_values).median().item()
                
                # Log per-source metrics
                self.log(f'test/{source_name}/snr_median', snr_median, 
                        logger=True, sync_dist=True)
                self.log(f'test/{source_name}/si_snr_median', si_snr_median, 
                        logger=True, sync_dist=True)
                
                print(f"  {source_name.upper()}:")
                print(f"    Median SNR: {snr_median:.3f} dB")
                print(f"    Median SI-SNR: {si_snr_median:.3f} dB")
                
                # Log and print Multi-Mel-SNR if available
                if len(multi_mel_snr_values) > 0:
                    all_multi_mel_snr_values.extend(multi_mel_snr_values)
                    multi_mel_snr_median = torch.tensor(multi_mel_snr_values).median().item()
                    self.log(f'test/{source_name}/multi_mel_snr_median', multi_mel_snr_median, 
                            logger=True, sync_dist=True)
                    print(f"    Median Multi-Mel-SNR: {multi_mel_snr_median:.3f} dB")
                
                print(f"    Total frames evaluated: {len(snr_values)}")
        
        # Compute overall metrics (across all sources)
        if len(all_snr_values) > 0:
            snr_median_overall = torch.tensor(all_snr_values).median().item()
            si_snr_median_overall = torch.tensor(all_si_snr_values).median().item()
            
            # Log overall metrics
            self.log(f'test/snr_median', snr_median_overall, 
                    logger=True, sync_dist=True)
            self.log(f'test/si_snr_median', si_snr_median_overall, 
                    logger=True, sync_dist=True)
            
            print(f"\n  OVERALL (all sources):")
            print(f"    Median SNR: {snr_median_overall:.3f} dB")
            print(f"    Median SI-SNR: {si_snr_median_overall:.3f} dB")
            
            # Log and print overall Multi-Mel-SNR if available
            if len(all_multi_mel_snr_values) > 0:
                multi_mel_snr_median_overall = torch.tensor(all_multi_mel_snr_values).median().item()
                self.log(f'test/multi_mel_snr_median', multi_mel_snr_median_overall, 
                        logger=True, sync_dist=True)
                print(f"    Median Multi-Mel-SNR: {multi_mel_snr_median_overall:.3f} dB")
            
            print(f"    Total frames evaluated: {len(all_snr_values)}")
        else:
            print(f"  No samples found")
        
        print("="*80 + "\n")

