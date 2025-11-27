"""DSP utilities for loudness, mixing, segmentation, and resampling."""

from .lufs import measure_lufs, normalize_to_lufs, true_peak_db  # noqa: F401
from .utils import batch_has_signal_lufs, ensure_sample_rate, fade_edges  # noqa: F401
from .music_effect_pipeline import MusicEffectPipeline  # noqa: F401

__all__ = [
    "measure_lufs",
    "normalize_to_lufs",
    "true_peak_db",
    "batch_has_signal_lufs",
    "ensure_sample_rate",
    "fade_edges",
    "MusicEffectPipeline",
]


from .torch_mastering import (  # noqa: F401
    LinearPhaseHPF,
    LoudnessNormalize,
    MidSideWidth,
    PeakNormalizer,
    TruePeakLimiter,
    build_mastering_compose,
)

__all__.extend(
    [
        "LinearPhaseHPF",
        "LoudnessNormalize",
        "MidSideWidth",
        "PeakNormalizer",
        "TruePeakLimiter",
        "build_mastering_compose",
    ]
)