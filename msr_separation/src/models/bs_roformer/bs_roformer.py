from __future__ import annotations

import os
from functools import partial

import torch
from torch import nn, einsum, tensor, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torchaudio

from .attend import Attend
from .lora import LoRALinearQKV

from beartype.typing import Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack

from hyper_connections import get_init_and_expand_reduce_stream_functions

from src.config_updates import RESOURCES_FOLDER

CKPT_NAME = "model_bs_roformer.ckpt"

# helper functions

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# attention

class FeedForward(Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        rotary_embed = None,
        flash = True,
        learned_value_residual_mix = False,
        lora_config = None
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head **-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(flash = flash, dropout = dropout)

        self.norm = RMSNorm(dim)
        
        # Create base linear layer
        base_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        
        # Apply LoRA if configured
        if lora_config is not None and lora_config.get('enabled', False):
            self.to_qkv = LoRALinearQKV(
                linear_layer=base_qkv,
                r=lora_config.get('r', 8),
                lora_alpha=lora_config.get('lora_alpha', 16),
                lora_dropout=lora_config.get('lora_dropout', 0.05),
                enable_lora=lora_config.get('enable_lora', [True, False, True])
            )
        else:
            self.to_qkv = base_qkv

        self.to_value_residual_mix = nn.Linear(dim, heads) if learned_value_residual_mix else None

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(self, x, value_residual = None):
        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = self.heads)

        orig_v = v

        if exists(self.to_value_residual_mix):
            mix = self.to_value_residual_mix(x)
            mix = rearrange(mix, 'b n h -> b h n 1').sigmoid()

            assert exists(value_residual)
            v = v.lerp(value_residual, mix)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), orig_v

class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        norm_output = True,
        rotary_embed = None,
        flash_attn = True,
        add_value_residual = False,
        num_residual_streams = 1,
        num_residual_fracs = 1,
        lora_config = None
    ):
        super().__init__()
        self.layers = ModuleList([])

        init_hyper_conn, *_ = get_init_and_expand_reduce_stream_functions(num_residual_streams, num_fracs = num_residual_fracs)

        for _ in range(depth):
            self.layers.append(ModuleList([
                init_hyper_conn(dim = dim, branch = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_embed = rotary_embed, flash = flash_attn, learned_value_residual_mix = add_value_residual, lora_config = lora_config)),
                init_hyper_conn(dim = dim, branch = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout))
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x, value_residual = None):

        first_values = None

        for attn, ff in self.layers:
            x, next_values = attn(x, value_residual = value_residual)

            first_values = default(first_values, next_values)

            x = ff(x)

        return self.norm(x), first_values

# bandsplit module

class BandSplit(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_inputs: tuple[int, ...]
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim = -1) # the first 8 frequencies go to the first band, next 8 to the second, etc.

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim = -2)

def MLP(
    dim_in,
    dim_out,
    dim_hidden = None,
    depth = 1,
    activation = nn.Tanh
):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)

class MaskEstimator(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_inputs: tuple[int, ...],
        depth,
        mlp_expansion_factor = 2
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            net = []

            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden = dim_hidden, depth = depth),
                nn.GLU(dim = -1)
            )

            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim = -2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim = -1)

# main class

DEFAULT_FREQS_PER_BANDS = (
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  12, 12, 12, 12, 12, 12, 12, 12,
  24, 24, 24, 24, 24, 24, 24, 24,
  48, 48, 48, 48, 48, 48, 48, 48,
  128, 129,
)

class BSRoformer(Module):

    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo = False,
        num_stems = 1,
        time_transformer_depth = 2,
        freq_transformer_depth = 2,
        freqs_per_bands: tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,  # in the paper, they divide into ~60 bands, test with 1 for starters
        freqs_per_bands_with_extra_bands = None,
        freq_range: tuple[int, int] | None = None, # specifying a frequency range, with (<min freq>, <max freq). `-1` implies 0 and inf
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        flash_attn = True,
        num_residual_streams = 4, # set to 1. to disable hyper connections
        num_residual_fracs = 1,   # can be used as an alternative to residual streams for memory efficiency while retaining benefits of hyper connections
        dim_freqs_in = 1025,
        stft_n_fft = 2048,
        stft_hop_length = 441, # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
        stft_win_length = 2048,
        stft_normalized = False,
        zero_dc = False, # @firebirdblue23 in https://github.com/lucidrains/BS-RoFormer/issues/47
        stft_window_fn: Callable | None = None,
        mask_estimator_depth = 2,
        mlp_expansion_factor = 4,
        load_pretrained: bool = False,
        lora_config = None,
        phased_fine_tuning = False,
        phased_fine_tuning_phase1_epochs: int = None,
        phased_fine_tuning_crossover_bin: int = 1025,  # Bin index where pretrained frequencies end (1025 = bins 0-1024)
        use_dual_stft = False,  # If True: Compute STFT twice (44.1kHz for first 62 bands, 48kHz for last 3 bands). If False: Process at 44.1kHz only and upsample output to 48kHz.
        sr_48k: int = 48000,  # Sample rate for 48kHz STFT (full range)
        n_fft_48k: int = 2229,  # n_fft for 48kHz STFT (full range)
        sr_pretrained=44100,  # Sample rate for pretrained model STFT (first 62 bands)
        hop_length_48k: int = 480,
        win_length_48k: int = 2229,
        time_domain_extra_merge = False,  # If True, merge extra 48kHz bands in time domain to avoid additional STFT
        resample_to_44100_for_model = False
    ):
        super().__init__()

        if use_dual_stft:
            freqs_per_bands = list(freqs_per_bands_with_extra_bands)

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems

        self.resample_to_44100_for_model = resample_to_44100_for_model

        _, self.expand_stream, self.reduce_stream = get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        self.layers = ModuleList([])

        transformer_kwargs = dict(
            dim = dim,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            flash_attn = flash_attn,
            num_residual_streams = num_residual_streams,
            num_residual_fracs = num_residual_fracs,
            norm_output = False,
            lora_config = lora_config,
        )

        time_rotary_embed = RotaryEmbedding(dim = dim_head)
        freq_rotary_embed = RotaryEmbedding(dim = dim_head)

        for layer_index in range(depth):
            is_first = layer_index == 0

            self.layers.append(nn.ModuleList([
                Transformer(depth = time_transformer_depth, rotary_embed = time_rotary_embed, add_value_residual = not is_first, **transformer_kwargs),
                Transformer(depth = freq_transformer_depth, rotary_embed = freq_rotary_embed, add_value_residual = not is_first, **transformer_kwargs)
            ]))

        self.final_norm = RMSNorm(dim)

        self.stft_kwargs_pretrained = dict(
            n_fft = stft_n_fft,
            hop_length = stft_hop_length,
            win_length = stft_win_length,
            normalized = stft_normalized
        )

        self.stft_window_fn_pretrained = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        # Calculate frequency bins based on the actual STFT we'll be using
        freq_bins_pretrained = \
        torch.stft(torch.randn(1, 4096), **self.stft_kwargs_pretrained, return_complex=True).shape[1]

        # Use pretrained STFT settings (44.1kHz)
        freqs = freq_bins_pretrained  # will eventually be overwritten

        # if use_dual_stft:
        #     self.stft_kwargs_48khz = dict(  # STFT settings for 48kHz (full range)
        #         n_fft=n_fft_48k,
        #         hop_length=hop_length_48k,
        #         win_length=win_length_48k,
        #         normalized=stft_normalized  # is always False right now
        #     )
        #
        #     self.stft_window_fn_48khz = partial(default(stft_window_fn, torch.hann_window), win_length_48k)
        #
        #     freq_bins_48k = torch.stft(torch.randn(1, 4096), **self.stft_kwargs_48khz, return_complex=True).shape[1]
        #
        #     # Calculate extra bins from 48kHz STFT (bins above 22.05kHz)
        #     freq_res_48k = sr_48k / n_fft_48k
        #     pretrained_nyquist = sr_pretrained / 2  # 22050 Hz
        #     extra_bins_start_48k = int(pretrained_nyquist / freq_res_48k) + 1
        #     extra_bins_count_48k = freq_bins_48k - extra_bins_start_48k
        #     # Total bins = pretrained bins + extra 48kHz bins
        #     freqs = freq_bins_pretrained + extra_bins_count_48k

        # enforcing a frequency range

        freq_range = default(freq_range, (-1, -1))
        min_freq, max_freq = freq_range

        min_freq = 0 if min_freq == -1 else min_freq
        max_freq = freqs if max_freq == -1 else max_freq

        assert min_freq >= 0 and max_freq <= freqs and min_freq < max_freq

        self.min_freq = min_freq
        self.max_freq = max_freq

        self.freq_slice = slice(min_freq, max_freq)  # for slicing out the frequency for training
        self.freq_pad = (min_freq, freqs - max_freq) # for reconstruction during istft


        assert len(freqs_per_bands) > 1
        # assert sum(freqs_per_bands) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}'

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(
            dim = dim,
            dim_inputs = freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim = dim,
                dim_inputs = freqs_per_bands_with_complex,
                depth = mask_estimator_depth,
                mlp_expansion_factor = mlp_expansion_factor
            )

            self.mask_estimators.append(mask_estimator)

        # whether to zero out dc

        self.zero_dc = min_freq == 0 and zero_dc
        
        # Phased fine-tuning configuration
        self.phased_fine_tuning = phased_fine_tuning
        self.phased_fine_tuning_phase1_epochs = phased_fine_tuning_phase1_epochs
        self.phased_fine_tuning_crossover_bin = phased_fine_tuning_crossover_bin
        self.current_phase = 1 if phased_fine_tuning else None  # Phase 1 or 2 (None = disabled)
        
        # Dual STFT configuration (for perfect bin alignment)
        self.use_dual_stft = use_dual_stft
        self.sr_48k = sr_48k
        self.sr_pretrained = sr_pretrained
        self.n_fft_pretrained = stft_n_fft
        self.n_fft_48k = n_fft_48k
        self.time_domain_extra_merge = time_domain_extra_merge

        print("self.resample_to_44100_for_model: ", self.resample_to_44100_for_model)

        # Initialize resamplers if resampling is enabled
        if self.resample_to_44100_for_model:
            # Down-sample: 48kHz -> 44.1kHz
            self.downsample_resampler = torchaudio.transforms.Resample(
                orig_freq=48000,
                new_freq=44100,
                resampling_method='kaiser_window',
                lowpass_filter_width=64,
                rolloff=0.99,
                beta=14.0
            )
            # Up-sample: 44.1kHz -> 48kHz
            self.upsample_resampler = torchaudio.transforms.Resample(
                orig_freq=44100,
                new_freq=48000,
                resampling_method='kaiser_window',
                lowpass_filter_width=64,
                rolloff=0.99,
                beta=14.0
            )
            print("Resampling enabled: mixture will be down-sampled to 44.1kHz for model, then up-sampled back to 48kHz")
        
        # # Use provided resamplers from Lightning module if available, otherwise create new ones
        # if downsample_resampler is not None and upsample_resampler is not None:
        #     # Use resamplers from Lightning module
        #     self._resampler_48k_to_pretrained = downsample_resampler
        #     self._resampler_pretrained_to_48k = upsample_resampler
        #     self._has_torchaudio = True  # Assume torchaudio resamplers if provided
        #     self._resamplers_from_lightning = True
        #     print("Using resamplers from Lightning module for dual-STFT approach")
        # else:
        #     # Initialize high-quality resamplers (cached for efficiency)
        #     # Use kaiser_window to match Lightning module resampling (for consistency with resample_to_44100_for_model)
        #     self._resamplers_from_lightning = False
        #     try:
        #         import torchaudio
        #         self._resampler_48k_to_pretrained = torchaudio.transforms.Resample(  # downsampler
        #             orig_freq=sr_48k,
        #             new_freq=sr_pretrained,
        #             resampling_method='kaiser_window',
        #             lowpass_filter_width=64,
        #             rolloff=0.99,
        #             beta=14.0
        #         )
        #         self._resampler_pretrained_to_48k = torchaudio.transforms.Resample(
        #             orig_freq=sr_pretrained,
        #             new_freq=sr_48k,
        #             resampling_method='kaiser_window',
        #             lowpass_filter_width=64,
        #             rolloff=0.99,
        #             beta=14.0
        #         )
        #         self._has_torchaudio = True
        #     except ImportError:
        #         self._has_torchaudio = False
        #         print("Warning: torchaudio not available, will use fallback resampling (lower quality)")
        
        if use_dual_stft:
            # Calculate number of bins for each STFT
            self.bins_pretrained = stft_n_fft // 2 + 1  # 1025 bins (44.1kHz)
            self.bins_48k = n_fft_48k // 2 + 1  # 1115 bins (48kHz)
            
            # Calculate which bins correspond to first 62 bands (pretrained) and last 3 bands (new)
            # First 62 bands = 1025 bins (from pretrained STFT, bins 0-1024, covering 0-22.05kHz)
            # Last 3 bands = 91 bins (from 48kHz STFT, bins 1024-1114, covering 22.05-24kHz)
            # Note: 48kHz bin 1024 starts at 22051.14 Hz (just above 22.05kHz), so there's a tiny gap (~1.14 Hz)
            # This is negligible and avoids overlap between the two STFTs
            # Calculate first bin in 48kHz STFT that's above 22.05kHz (Nyquist of pretrained)
            freq_res_48k = sr_48k / n_fft_48k
            pretrained_nyquist = sr_pretrained / 2  # 22050 Hz
            self.extra_bins_start_48k = int(pretrained_nyquist / freq_res_48k) + 1  # First bin above 22.05kHz (bin 1024)
            self.extra_bins_end_48k = self.bins_48k  # 1115 bins
            self.extra_bins_count_48k = self.extra_bins_end_48k - self.extra_bins_start_48k  # Should be 91 bins
            
            print(f"Dual STFT enabled:")
            nyquist_48k = sr_48k / 2  # 24000 Hz
            print(f"  - Pretrained STFT (n_fft={stft_n_fft}, {sr_pretrained}Hz): {self.bins_pretrained} bins (0-{self.bins_pretrained-1}) → first 62 bands")
            print(f"  - 48kHz STFT (n_fft={n_fft_48k}, {sr_48k}Hz): {self.extra_bins_count_48k} bins ({self.extra_bins_start_48k}-{self.extra_bins_end_48k-1}) → last 3 bands")
            print(f"    (covers {self.extra_bins_start_48k * freq_res_48k:.2f} - {nyquist_48k:.2f} Hz)")
        else:
            # When use_dual_stft=False, we use pretrained processing (44.1kHz) and upsample to 48kHz
            self.bins_pretrained = stft_n_fft // 2 + 1  # 1025 bins (44.1kHz)
            print(f"Single STFT mode (use_dual_stft=False):")
            print(f"  - Processing at {sr_pretrained}Hz (pretrained model): {self.bins_pretrained} bins (0-{self.bins_pretrained-1})")
            print(f"  - Will upsample prediction to {sr_48k}Hz")
        
        # Store number of pretrained bands for freezing during phase 1
        self.num_pretrained_bands = len(DEFAULT_FREQS_PER_BANDS) if phased_fine_tuning else None
        
        # Freeze new bands' parameters during phase 1
        if phased_fine_tuning and len(freqs_per_bands) > self.num_pretrained_bands:
            self._freeze_new_bands_phase1()

        if load_pretrained:
            ckpt_file = os.path.join(RESOURCES_FOLDER, CKPT_NAME)
            ckpt = torch.load(ckpt_file, map_location='cpu')
            
            # Handle both checkpoint formats: direct state_dict or wrapped in 'state_dict' key
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                ckpt_state_dict = ckpt['state_dict']
            else:
                ckpt_state_dict = ckpt
            
            # Map checkpoint keys to current model structure
            # Checkpoint may not have 'branch' prefix (from hyper-connections) and may have different LoRA structure
            mapped_state_dict = {}
            model_state_dict = self.state_dict()
            
            def try_map_key(ckpt_key):
                """Try to map a checkpoint key to a model key."""
                # Try direct match first
                if ckpt_key in model_state_dict:
                    return ckpt_key
                
                # Pattern: layers.X.Y.layers.Z.W.module.param -> layers.X.Y.layers.Z.W.branch.module.param
                # Need to insert 'branch' before module names (to_qkv, to_gates, net, norm, rotary_embed, etc.)
                
                # List of module names that come after 'branch' in the hyper-connection wrapper
                module_names = ['to_qkv', 'to_gates', 'to_out', 'to_value_residual_mix', 'net', 'norm', 'rotary_embed']
                
                # Special case: to_qkv.weight -> to_qkv.linear.weight (LoRA wrapping) with branch
                if '.to_qkv.weight' in ckpt_key:
                    # Replace .to_qkv.weight with .branch.to_qkv.linear.weight
                    potential_key = ckpt_key.replace('.to_qkv.weight', '.branch.to_qkv.linear.weight')
                    if potential_key in model_state_dict:
                        return potential_key
                
                # Try adding 'branch' prefix before module names
                for module_name in module_names:
                    # Pattern: ...layers.X.Y.module_name... -> ...layers.X.Y.branch.module_name...
                    # Use rsplit to only replace the last occurrence (in case module name appears multiple times)
                    if ckpt_key.endswith(f'.{module_name}'):
                        # Handle case where module_name is at the end
                        potential_key = ckpt_key.rsplit(f'.{module_name}', 1)[0] + f'.branch.{module_name}'
                        if potential_key in model_state_dict:
                            return potential_key
                    elif f'.{module_name}.' in ckpt_key:
                        # Handle case where module_name is in the middle (e.g., to_qkv.weight)
                        # Replace only the first occurrence after layers pattern
                        # Find the position after layers.X.Y pattern
                        potential_key = ckpt_key.replace(f'.{module_name}.', f'.branch.{module_name}.', 1)
                        if potential_key in model_state_dict:
                            return potential_key
                
                return None
            
            for ckpt_key, ckpt_value in ckpt_state_dict.items():
                model_key = try_map_key(ckpt_key)
                if model_key is not None:
                    # Check for shape mismatches that could indicate configuration differences
                    if model_key in model_state_dict:
                        ckpt_shape = ckpt_value.shape
                        model_shape = model_state_dict[model_key].shape
                        if ckpt_shape != model_shape:
                            print(f"Warning: Shape mismatch for {model_key}: checkpoint {ckpt_shape} vs model {model_shape} - skipping")
                            continue
                    mapped_state_dict[model_key] = ckpt_value
                # If no mapping found, we'll skip it (may be hyper-connection params not in checkpoint)
            
            # Load with strict=False to allow missing keys (LoRA params, hyper-connection params, etc.)
            missing_keys, unexpected_keys = self.load_state_dict(mapped_state_dict, strict=False)
            
            # Initialize value_residual_mix layers for NO mixing (they're not in reference checkpoints)
            # sigmoid(negative_bias) → 0 means v.lerp(value_residual, 0) = v (no mixing, uses original v)
            # This matches reference behavior (which doesn't have value residual mixing)
            for name, param in self.named_parameters():
                if 'value_residual_mix' in name:
                    if 'weight' in name:
                        nn.init.zeros_(param)  # Zero weights
                    elif 'bias' in name:
                        # Initialize bias to large negative value so sigmoid(bias) ≈ 0
                        # This ensures no mixing: v.lerp(value_residual, 0) = v
                        nn.init.constant_(param, -10.0)
            
            # Print informative messages
            if missing_keys:
                lora_keys = [k for k in missing_keys if 'lora' in k.lower()]
                hyper_keys = [k for k in missing_keys if any(x in k for x in ['static_alpha', 'dynamic_alpha', 'static_beta', 'dynamic_beta'])]
                value_residual_keys = [k for k in missing_keys if 'value_residual_mix' in k]
                other_keys = [k for k in missing_keys if k not in lora_keys and k not in hyper_keys and k not in value_residual_keys]
                
                if lora_keys:
                    print(f"Info: {len(lora_keys)} LoRA parameters not in checkpoint (will use zero initialization)")
                if hyper_keys:
                    print(f"Info: {len(hyper_keys)} hyper-connection parameters not in checkpoint (will use default initialization)")

                    print("-"*20)
                    print(hyper_keys)
                    print("-"*20)

                if value_residual_keys:
                    print(f"Info: {len(value_residual_keys)} value_residual_mix parameters not in checkpoint (initialized to zero for neutral behavior)")

                    print("-" * 20)
                    print(value_residual_keys)
                    print("-" * 20)

                if other_keys:
                    print(f"Warning: {len(other_keys)} other parameters not in checkpoint: {other_keys[:5]}...")

                    print("-" * 20)
                    print(other_keys)
                    print("-" * 20)

            if unexpected_keys:
                print(f"Info: {len(unexpected_keys)} unexpected keys in checkpoint (ignored)")
            
            # Check for potential configuration mismatches
            if num_residual_streams > 1:
                print(f"Warning: Using hyper-connections (num_residual_streams={num_residual_streams}) but checkpoint likely trained without them.")
                print(f"         Hyper-connection parameters use default initialization and may affect performance.")
                print(f"         Consider setting num_residual_streams=1 for best checkpoint compatibility.")
            
            # Initialize new frequency bands by copying/interpolating from highest pretrained bands
            # This is useful when extending frequency range (e.g., 44.1kHz -> 48kHz)
            num_pretrained_bands = len(DEFAULT_FREQS_PER_BANDS)  # 62 bands in pretrained model
            num_current_bands = len(freqs_per_bands)
            
            if num_current_bands > num_pretrained_bands:
                num_new_bands = num_current_bands - num_pretrained_bands
                print(f"Info: Detected {num_new_bands} new frequency bands (bands {num_pretrained_bands}-{num_current_bands-1})")
                print(f"      Initializing new bands from highest pretrained band (band {num_pretrained_bands-1})")
                
                # Initialize new bands in BandSplit
                last_pretrained_band_idx = num_pretrained_bands - 1
                source_band = self.band_split.to_features[last_pretrained_band_idx]
                
                for i in range(num_pretrained_bands, num_current_bands):
                    new_band_idx = i
                    new_dim_in = freqs_per_bands_with_complex[new_band_idx]
                    source_dim_in = freqs_per_bands_with_complex[last_pretrained_band_idx]
                    
                    # Create new band network
                    new_net = nn.Sequential(
                        RMSNorm(new_dim_in),
                        nn.Linear(new_dim_in, dim)
                    )
                    
                    # Copy weights from source band
                    if new_dim_in == source_dim_in:
                        # Same input dimension - direct copy
                        new_net.load_state_dict(source_band.state_dict())
                    else:
                        # Different input dimension - interpolate weights
                        source_rms_norm = source_band[0]  # RMSNorm
                        source_linear = source_band[1]    # Linear layer
                        
                        # Interpolate RMSNorm gamma (shape: [dim_in])
                        # gamma has shape [dim_in], so we need to interpolate from [source_dim_in] to [new_dim_in]
                        # Use 1D linear interpolation for 1D tensors
                        if hasattr(source_rms_norm, 'gamma'):
                            source_gamma = source_rms_norm.gamma.data  # [source_dim_in]
                            source_gamma_3d = source_gamma.unsqueeze(0).unsqueeze(0)  # [1, 1, source_dim_in] for 1D interpolation
                            interpolated_gamma = F.interpolate(
                                source_gamma_3d,
                                size=new_dim_in,
                                mode='linear',
                                align_corners=False
                            ).squeeze(0).squeeze(0)  # [new_dim_in]
                            new_net[0].gamma.data.copy_(interpolated_gamma)
                        
                        # Interpolate Linear layer weights
                        # Weight shape: [dim_out, dim_in] = [dim, dim_in]
                        source_weight = source_linear.weight.data  # [dim, source_dim_in]
                        
                        # Interpolate weight matrix: [dim, source_dim_in] -> [dim, new_dim_in]
                        # Use bilinear interpolation on the weight matrix
                        # F.interpolate expects 4D input: (N, C, H, W)
                        source_weight_4d = source_weight.unsqueeze(0).unsqueeze(0)  # [1, 1, dim, source_dim_in]
                        interpolated_weight = F.interpolate(
                            source_weight_4d, 
                            size=(dim, new_dim_in), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0).squeeze(0)  # [dim, new_dim_in]
                        
                        new_net[1].weight.data.copy_(interpolated_weight)
                        
                        # Copy bias if exists (bias shape: [dim], same for both)
                        if source_linear.bias is not None and new_net[1].bias is not None:
                            new_net[1].bias.data.copy_(source_linear.bias.data)
                    
                    # Replace the randomly initialized band with the interpolated one
                    self.band_split.to_features[new_band_idx] = new_net
                
                # Initialize new bands in MaskEstimator (for each stem)
                for stem_idx, mask_estimator in enumerate(self.mask_estimators):
                    source_mlp = mask_estimator.to_freqs[last_pretrained_band_idx]
                    
                    for i in range(num_pretrained_bands, num_current_bands):
                        new_band_idx = i
                        new_dim_in = freqs_per_bands_with_complex[new_band_idx]
                        source_dim_in = freqs_per_bands_with_complex[last_pretrained_band_idx]
                        
                        # Create new MLP (use same structure as MaskEstimator)
                        dim_hidden = dim * mlp_expansion_factor
                        new_mlp = nn.Sequential(
                            MLP(dim, new_dim_in * 2, dim_hidden=dim_hidden, depth=mask_estimator_depth),
                            nn.GLU(dim=-1)
                        )
                        
                        if new_dim_in == source_dim_in:
                            # Same output dimension - direct copy
                            new_mlp.load_state_dict(source_mlp.state_dict())
                        else:
                            # Different output dimension - interpolate weights
                            # The MLP structure is: MLP -> GLU
                            # MLP outputs new_dim_in * 2, GLU splits it in half
                            source_mlp_net = source_mlp[0]  # The MLP part (nn.Sequential)
                            new_mlp_net = new_mlp[0]        # The MLP part (nn.Sequential)
                            
                            # Get all Linear layers (skip activations)
                            source_linear_layers = [layer for layer in source_mlp_net if isinstance(layer, nn.Linear)]
                            new_linear_layers = [layer for layer in new_mlp_net if isinstance(layer, nn.Linear)]
                            
                            # Copy all layers except the last one (which has different output size)
                            for layer_idx in range(len(new_linear_layers) - 1):
                                if layer_idx < len(source_linear_layers) - 1:
                                    if new_linear_layers[layer_idx].weight.shape == source_linear_layers[layer_idx].weight.shape:
                                        new_linear_layers[layer_idx].weight.data.copy_(source_linear_layers[layer_idx].weight.data)
                                        if source_linear_layers[layer_idx].bias is not None and new_linear_layers[layer_idx].bias is not None:
                                            new_linear_layers[layer_idx].bias.data.copy_(source_linear_layers[layer_idx].bias.data)
                            
                            # Interpolate the last layer (output layer)
                            if len(source_linear_layers) > 0 and len(new_linear_layers) > 0:
                                last_source_layer = source_linear_layers[-1]  # Last Linear layer
                                last_new_layer = new_linear_layers[-1]        # Last Linear layer
                                
                                # Last layer shape: [source_dim_in * 2, hidden_dim] -> [new_dim_in * 2, hidden_dim]
                                source_weight = last_source_layer.weight.data  # [source_dim_in * 2, hidden_dim]
                                # F.interpolate expects 4D input: (N, C, H, W)
                                source_weight_4d = source_weight.unsqueeze(0).unsqueeze(0)  # [1, 1, source_dim_in * 2, hidden_dim]
                                interpolated_weight = F.interpolate(
                                    source_weight_4d,
                                    size=(new_dim_in * 2, source_weight.shape[1]),
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze(0).squeeze(0)  # [new_dim_in * 2, hidden_dim]
                                
                                last_new_layer.weight.data.copy_(interpolated_weight)
                                
                                # Interpolate bias (1D tensor, use linear interpolation)
                                if last_source_layer.bias is not None and last_new_layer.bias is not None:
                                    source_bias = last_source_layer.bias.data  # [source_dim_in * 2]
                                    source_bias_3d = source_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, source_dim_in * 2] for 1D interpolation
                                    interpolated_bias = F.interpolate(
                                        source_bias_3d,
                                        size=new_dim_in * 2,
                                        mode='linear',
                                        align_corners=False
                                    ).squeeze(0).squeeze(0)  # [new_dim_in * 2]
                                    last_new_layer.bias.data.copy_(interpolated_bias)
                        
                        # Replace the randomly initialized MLP with the interpolated one
                        mask_estimator.to_freqs[new_band_idx] = new_mlp
                
                print(f"      Successfully initialized {num_new_bands} new bands from pretrained band {last_pretrained_band_idx}")
    
    def _freeze_new_bands_phase1(self):
        """Freeze parameters of new frequency bands during phase 1 of phased fine-tuning."""
        if not self.phased_fine_tuning or self.num_pretrained_bands is None:
            return
        
        num_current_bands = len(self.band_split.to_features)
        if num_current_bands <= self.num_pretrained_bands:
            return
        
        # Freeze new bands in BandSplit
        for i in range(self.num_pretrained_bands, num_current_bands):
            for param in self.band_split.to_features[i].parameters():
                param.requires_grad = False
        
        # Freeze new bands in MaskEstimator (for each stem)
        for mask_estimator in self.mask_estimators:
            for i in range(self.num_pretrained_bands, num_current_bands):
                for param in mask_estimator.to_freqs[i].parameters():
                    param.requires_grad = False
        
        print(f"Phase 1: Frozen {num_current_bands - self.num_pretrained_bands} new frequency bands (bands {self.num_pretrained_bands}-{num_current_bands-1})")
    
    def _unfreeze_new_bands_phase2(self):
        """Unfreeze parameters of new frequency bands for phase 2 of phased fine-tuning."""
        if not self.phased_fine_tuning or self.num_pretrained_bands is None:
            return
        
        num_current_bands = len(self.band_split.to_features)
        if num_current_bands <= self.num_pretrained_bands:
            return
        
        # Unfreeze new bands in BandSplit
        for i in range(self.num_pretrained_bands, num_current_bands):
            for param in self.band_split.to_features[i].parameters():
                param.requires_grad = True
        
        # Unfreeze new bands in MaskEstimator (for each stem)
        for mask_estimator in self.mask_estimators:
            for i in range(self.num_pretrained_bands, num_current_bands):
                for param in mask_estimator.to_freqs[i].parameters():
                    param.requires_grad = True
        
        print(f"Phase 2: Unfrozen {num_current_bands - self.num_pretrained_bands} new frequency bands (bands {self.num_pretrained_bands}-{num_current_bands-1})")
    
    def set_phase(self, phase: int):
        """Set the current phase for phased fine-tuning (1 or 2)."""
        if not self.phased_fine_tuning:
            return
        
        if phase == 1:
            self.current_phase = 1
            self._freeze_new_bands_phase1()
        elif phase == 2:
            self.current_phase = 2
            self._unfreeze_new_bands_phase2()
        else:
            raise ValueError(f"Phase must be 1 or 2, got {phase}")

    def forward(
        self,
        raw_audio,
        target = None,
        return_loss_breakdown = False
    ):
        """
        einops

        b - batch
        f - freq
        t - time
        s - audio channel (1 for mono, 2 for stereo)
        n - number of 'stems'
        c - complex (2)
        d - feature dimension
        """

        device = raw_audio.device

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

        # Store original length for proper reconstruction
        original_length = raw_audio.shape[-1]
        
        # Pad input to ensure compatibility with STFT parameters
        # The input length should be compatible with hop_length for perfect reconstruction
        # hop_length_48k = self.stft_kwargs_pretrained['hop_length']
        # pad_length = (hop_length_48k - (original_length % hop_length_48k)) % hop_length_48k
        # if pad_length > 0:
        #     raw_audio = F.pad(raw_audio, (0, pad_length))

        # to stft

        stft_window_pretrained = self.stft_window_fn_pretrained(device = device)
        
        # Ensure STFT runs in fp32 precision for numerical stability
        with autocast(enabled=False):

            if self.resample_to_44100_for_model:
                # Resample raw_audio to 44.1kHz for model processing
                # Use torchaudio resampler (kaiser_window) for high-quality resampling
                # Cast to float32 for resampling (resampler weights are float32)
                original_dtype = raw_audio.dtype
                raw_audio = self.downsample_resampler(raw_audio.float())
                raw_audio = raw_audio.to(dtype=original_dtype)

            raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')

            # Compute STFT at pretrained sample rate for pretrained bands (first 62 bands, 1025 bins covering 0-22.05kHz)
            # stft_window_pretrained = self.stft_window_fn_pretrained(device=raw_audio.device)
            stft_pretrained = torch.stft(
                raw_audio,
                **self.stft_kwargs_pretrained,
                window=stft_window_pretrained,
                return_complex=True
            )  # [..., 1025, time_pretrained] - bins 0-1024, covering 0-22050 Hz


            if self.use_dual_stft:
                # Compute STFT at 48kHz for extra bands (last 3 bands, bins above 22.05kHz)
                stft_window_48khz = self.stft_window_fn_48khz(device=raw_audio.device)
                stft_48k = torch.stft(
                    raw_audio,
                    **self.stft_kwargs_48khz,
                    window=stft_window_48khz,
                    return_complex=True
                )  # [..., 1115, time_48k] - bins 0-1114, covering 0-24000 Hz

                # Extract extra frequencies from 48kHz STFT (bins covering 22.05-24kHz)
                # Start from bin self.extra_bins_start_48k (1024) which represents ~22051 Hz
                # This avoids overlap with pretrained STFT (which covers 0-22050 Hz)
                stft_extra_48k = stft_48k[..., self.extra_bins_start_48k:, :]  # [..., 91, time_48k] - bins 1024-1114

                # Verify we have the expected number of bins
                assert stft_extra_48k.shape[-2] == self.extra_bins_count_48k, \
                    f"Expected {self.extra_bins_count_48k} extra bins, got {stft_extra_48k.shape[-2]}"

                # Match time dimensions: stft_pretrained and stft_extra_48k may have different time frames
                # due to different sample rates and STFT parameters
                time_pretrained = stft_pretrained.shape[-1]
                time_48k = stft_extra_48k.shape[-1]

                if time_pretrained != time_48k:
                    # Pad or trim to match the smaller time dimension
                    min_time = min(time_pretrained, time_48k)
                    stft_pretrained = stft_pretrained[..., :min_time]
                    stft_extra_48k = stft_extra_48k[..., :min_time]

                # Combine: first 1025 bins from pretrained STFT (0-22.05kHz) + 91 bins from 48kHz STFT (22.05-24kHz)
                # Total: 1025 + 91 = 1116 bins (note: this is 1 more than 48kHz STFT's 1115 bins)
                # This is because we're combining bins 0-1024 from pretrained with bins 1024-1114 from 48kHz
                # Bin 1024 appears in both, but represents different frequencies:
                #   - Pretrained bin 1024: 22050 Hz (Nyquist)
                #   - 48kHz bin 1024: 22051.14 Hz (just above Nyquist)
                # So there's a tiny gap (~1.14 Hz) but no overlap
                stft_repr_complex = torch.cat([stft_pretrained, stft_extra_48k], dim=-2)  # [..., 1025+91=1116, time]

                # Verify total frequency bins match expected combined size
                expected_total_bins = self.bins_pretrained + self.extra_bins_count_48k
                assert stft_repr_complex.shape[-2] == expected_total_bins, \
                    f"Expected {expected_total_bins} total bins, got {stft_repr_complex.shape[-2]}"
            else:
                # When use_dual_stft=False: process at pretrained sample rate
                stft_repr_complex = stft_pretrained
            
            stft_repr = torch.view_as_real(stft_repr_complex)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')
        
        # Phase 1: Mask out high frequencies (above crossover bin) during phased fine-tuning
        # This happens BEFORE freq_slice, so we mask in the original STFT space
        if self.phased_fine_tuning and self.current_phase == 1:
            # Zero out frequencies above the crossover bin (bins >= crossover_bin)
            # stft_repr shape: [batch, channels, freq_bins, time, complex]
            # crossover_bin is in original STFT space (before freq_slice)
            if self.phased_fine_tuning_crossover_bin < stft_repr.shape[2]:
                stft_repr[:, :, self.phased_fine_tuning_crossover_bin:, :, :] = 0
        
        stft_repr = stft_repr[:, :, self.freq_slice] # slice out frequency range

        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c') # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting  # 1025 --> 2050 (stereo)

        x = rearrange(stft_repr, 'b f t c -> b t (f c)')

        x = self.band_split(x)  # input: bs x 1001 x 4100, output: bs x 1001 x 62 x 384

        # value residuals

        time_v_residual = None
        freq_v_residual = None

        # maybe expand residual streams

        x = self.expand_stream(x)

        # axial / hierarchical attention

        for time_transformer, freq_transformer in self.layers:  # per layer we have a time and freq transformer

            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            x, next_time_v_residual = time_transformer(x, value_residual = time_v_residual)

            time_v_residual = default(time_v_residual, next_time_v_residual)

            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')

            x, next_freq_v_residual = freq_transformer(x, value_residual = freq_v_residual)

            freq_v_residual = default(freq_v_residual, next_freq_v_residual)

            x, = unpack(x, ps, '* f d')

        # maybe reduce residual streams

        x = self.reduce_stream(x)

        x = self.final_norm(x)

        num_stems = len(self.mask_estimators)

        mask = torch.stack([fn(x) for fn in self.mask_estimators], dim = 1)
        mask = rearrange(mask, 'b n t (f c) -> b n f t c', c = 2)

        # modulate frequency representation

        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        # complex number multiplication

        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask)

        stft_repr = stft_repr * mask

        # istft

        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s = self.audio_channels)

        if self.zero_dc:
            # whether to dc filter
            stft_repr = stft_repr.index_fill(1, tensor(0, device = device), 0.)

        # Ensure iSTFT runs in fp32 precision for numerical stability
        # Use length parameter for perfect reconstruction (especially important for non-power-of-2 n_fft)
        with autocast(enabled=False):
            # stft_repr is already complex at this point (converted at line 921)
            # Check if it's already complex or needs conversion
            if torch.is_complex(stft_repr):
                stft_repr_complex = stft_repr
            else:
                # Convert from real representation to complex
                stft_repr_complex = torch.view_as_complex(stft_repr)

            stft_pretrained = stft_repr_complex[..., :self.bins_pretrained, :]  # [..., 1025, time]

            if self.use_dual_stft:
                stft_extra_48k = stft_repr_complex[..., self.bins_pretrained:, :]  # [..., extra_bins, time]

            recon_audio = torch.istft(
                stft_pretrained,  # [..., 1025, time_pretrained]
                **self.stft_kwargs_pretrained,
                window=stft_window_pretrained,
                length=None,
                return_complex=False
            )

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=self.audio_channels, n=num_stems)

        # Trim back to original length
        recon_audio = recon_audio[..., :original_length]

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        # up-sample prediction
        bs, n_sources, nch, wlen = recon_audio.shape
        target_length = original_length
        waveform_reshaped = recon_audio.reshape(bs * n_sources, nch, wlen)
        waveform_upsampled = self.upsample_resampler(waveform_reshaped)
        waveform_upsampled = waveform_upsampled.view(bs, n_sources, nch, -1)
        current_length = waveform_upsampled.shape[-1]
        if current_length > target_length:
            waveform_upsampled = waveform_upsampled[..., :target_length]
        elif current_length < target_length:
            pad_length = target_length - current_length
            waveform_upsampled = F.pad(waveform_upsampled, (0, pad_length), mode='constant', value=0.0)

        return waveform_upsampled

        #     if not self.use_dual_stft: # NOT
        #         recon_audio = audio_pretrained_upsampled
        #
        #     else: # Dual STFT
        #         if self.time_domain_extra_merge:
        #             # Ensure the upsampled audio length matches the original 48kHz length
        #             if audio_pretrained_upsampled.shape[-1] > original_length:
        #                 audio_pretrained_upsampled = audio_pretrained_upsampled[..., :original_length]
        #             elif audio_pretrained_upsampled.shape[-1] < original_length:
        #                 pad_amount = original_length - audio_pretrained_upsampled.shape[-1]
        #                 audio_pretrained_upsampled = F.pad(audio_pretrained_upsampled, (0, pad_amount))
        #
        #             # Align time dimensions between pretrained and extra-band STFTs
        #             min_time = min(stft_pretrained.shape[-1], stft_extra_48k.shape[-1])
        #             stft_pretrained = stft_pretrained[..., :min_time]
        #             stft_extra_48k = stft_extra_48k[..., :min_time]
        #
        #             # Build full 48kHz STFT containing only the extra high-frequency bins
        #             stft_window_48khz = self.stft_window_fn_48khz(device=stft_repr_complex.device)
        #             stft_48k_extra_full = stft_extra_48k.new_zeros(
        #                 stft_extra_48k.shape[:-2] + (self.bins_48k, stft_extra_48k.shape[-1])
        #             )
        #             extra_bins_to_add = min(
        #                 stft_48k_extra_full.shape[-2] - self.extra_bins_start_48k,
        #                 stft_extra_48k.shape[-2]
        #             )
        #             if extra_bins_to_add > 0:
        #                 stft_48k_extra_full[
        #                 ...,
        #                 self.extra_bins_start_48k:self.extra_bins_start_48k + extra_bins_to_add,
        #                 :
        #                 ] = stft_extra_48k[..., :extra_bins_to_add, :]
        #
        #             # Reconstruct the extra-band audio directly in the time domain
        #             extra_audio_48k = torch.istft(
        #                 stft_48k_extra_full,
        #                 **self.stft_kwargs_48khz,
        #                 window=stft_window_48khz,
        #                 length=original_length,
        #                 return_complex=False
        #             )
        #
        #             # Combine low-band (upsampled) audio with extra-band contribution
        #             recon_audio = audio_pretrained_upsampled + extra_audio_48k
        #         else:
        #             # Compute STFT of upsampled audio at 48kHz (Dual-STFT)
        #             stft_window_48khz = self.stft_window_fn_48khz(device=stft_repr_complex.device)
        #             stft_pretrained_upsampled_48k = torch.stft(
        #                 audio_pretrained_upsampled,
        #                 **self.stft_kwargs_48khz,
        #                 window=stft_window_48khz,
        #                 return_complex=True
        #             )  # [..., 1115, time_48k]
        #
        #             # Step 2: Pad extra frequencies to full 48kHz STFT structure
        #             # Match time dimensions
        #             min_time = min(stft_pretrained_upsampled_48k.shape[-1], stft_extra_48k.shape[-1])
        #             stft_pretrained_upsampled_48k = stft_pretrained_upsampled_48k[..., :min_time]
        #             stft_extra_48k = stft_extra_48k[..., :min_time]
        #
        #             # Create full 48kHz STFT with extra frequencies (zeros for 0-22.05kHz, extra for 22.05-24kHz)
        #             stft_48k_extra_full = torch.zeros_like(stft_pretrained_upsampled_48k)
        #             extra_bins_to_add = min(
        #                 stft_48k_extra_full.shape[-2] - self.extra_bins_start_48k,
        #                 stft_extra_48k.shape[-2]
        #             ) # should be 91 (or 90)
        #             if extra_bins_to_add > 0:
        #                 stft_48k_extra_full[...,
        #                 self.extra_bins_start_48k:self.extra_bins_start_48k + extra_bins_to_add, :] = stft_extra_48k[
        #                                                                                               ...,
        #                                                                                               :extra_bins_to_add,
        #                                                                                               :]
        #
        #             # Step 3: Combine: pretrained frequencies (0-22.05kHz) + extra frequencies (22.05-24kHz)
        #             stft_combined_48k = stft_pretrained_upsampled_48k + stft_48k_extra_full
        #
        #             # Final reconstruction at 48kHz
        #             recon_audio = torch.istft(
        #                 stft_combined_48k,
        #                 **self.stft_kwargs_48khz,
        #                 window=stft_window_48khz,
        #                 length=original_length,
        #                 return_complex=False
        #             )
        #
        #     # Reshape to match expected output format
        #     recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=self.audio_channels, n=num_stems)
        #
        # if num_stems == 1:
        #     recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')
        #
        # return recon_audio