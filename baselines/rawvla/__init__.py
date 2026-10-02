"""RAW-VLA streaming ISP baseline."""

try:
    from .rawvla import (
        DEFAULT_EXPOSURE_TARGET,
        FFTConservativeMerge,
        FIXED_DENOISE_ETA,
        MAX_EXPOSURE_EV,
        MAX_GAIN_EV,
        MAX_WB_EV,
        RAWChromaStatistics,
        RAWHistogram,
        RAWLuminanceStatistics,
        RAWVLA,
        RAWVLAOutput,
        RAWVLATheta,
        StatisticsAttentionPool,
        exposure_prior_loss,
        monotonic_bernstein_tone,
        neutral_theta,
        smooth_theta,
        summarize_theta,
    )
except ModuleNotFoundError as exc:  # Keep repository metadata importable without torch.
    if exc.name != "torch":
        raise
    FFTConservativeMerge = None
    FIXED_DENOISE_ETA = None
    DEFAULT_EXPOSURE_TARGET = None
    MAX_EXPOSURE_EV = None
    MAX_GAIN_EV = None
    MAX_WB_EV = None
    RAWHistogram = None
    RAWLuminanceStatistics = None
    RAWChromaStatistics = None
    RAWVLA = None
    RAWVLAOutput = None
    RAWVLATheta = None
    StatisticsAttentionPool = None
    exposure_prior_loss = None
    monotonic_bernstein_tone = None
    neutral_theta = None
    smooth_theta = None
    summarize_theta = None

__all__ = [
    "DEFAULT_EXPOSURE_TARGET",
    "FFTConservativeMerge",
    "FIXED_DENOISE_ETA",
    "MAX_EXPOSURE_EV",
    "MAX_GAIN_EV",
    "MAX_WB_EV",
    "RAWHistogram",
    "RAWLuminanceStatistics",
    "RAWChromaStatistics",
    "RAWVLA",
    "RAWVLAOutput",
    "RAWVLATheta",
    "StatisticsAttentionPool",
    "exposure_prior_loss",
    "monotonic_bernstein_tone",
    "neutral_theta",
    "smooth_theta",
    "summarize_theta",
]
