"""Streaming illumination-adaptive RAW ISP for VLA frontends.

The module implements the architecture specified in ``design_v1.md`` and
``network_structure.md``.  It consumes a causal burst of three-channel linear
RGB RAW frames and returns a VLA-facing RGB frame together with two compact
streaming states:

* ``state`` is a GRU state for exposure/noise trends and adaptation hysteresis.
* ``theta`` is the explicit color/tone ISP operating point; its compatibility
  field for burst strength carries a configured static value (default ``0.5``).

The downstream task loss is intentionally external to this module.  No
Self-Boost or clean-image reconstruction objective is used here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_EXPOSURE_TARGET = 0.48
DEFAULT_DENOISE_ETA = 0.5
# Backward-compatible alias for downstream imports. The value is static within
# a run, but can now be selected with ``burst_denoise_eta`` in the config.
FIXED_DENOISE_ETA = DEFAULT_DENOISE_ETA
MAX_EXPOSURE_EV = 6.0
MAX_WB_EV = 1.0
# Backward-compatible public alias. The revised model uses this range only for
# the achromatic scalar exposure, never independently per RGB channel.
MAX_GAIN_EV = MAX_EXPOSURE_EV


def _zero_init(module: nn.Module, *, bias: float = 0.0) -> None:
    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        raise TypeError(f"Expected Conv2d or Linear, got {type(module).__name__}")
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


def _tiny_init(module: nn.Module, *, bias: float = 0.0, std: float = 1.0e-3) -> None:
    """Keep a head close to neutral without blocking its upstream gradient."""

    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        raise TypeError(f"Expected Conv2d or Linear, got {type(module).__name__}")
    nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


class ConvBlock(nn.Module):
    """3x3 convolution followed by GroupNorm and SiLU."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


@dataclass
class RAWVLATheta:
    """Compact explicit ISP operating point retained between frames.

    Spatially varying residuals are deliberately excluded.  They are predicted
    from the current frame and are not recurrent, so state memory remains
    independent of image resolution.
    """

    eta_denoise: Tensor  # [B, 1], configured static burst merge strength
    exposure_ev: Tensor  # [B, 1], achromatic exposure in EV
    wb_ev: Tensor  # [B, 3], zero-sum relative white balance in EV
    ccm_matrix: Tensor  # [B, 3, 3], unit diagonal with bounded off-diagonals
    tone_logits: Tensor  # [B, 3, 7], relative logits; the eighth logit is fixed to zero

    @property
    def exposure_gain(self) -> Tensor:
        return torch.exp2(self.exposure_ev)

    @property
    def color_matrix(self) -> Tensor:
        """Return the fused, scale-free WB/CCM matrix ``C @ W``."""

        wb_matrix = torch.diag_embed(torch.exp2(self.wb_ev))
        return self.ccm_matrix @ wb_matrix

    def detach(self) -> "RAWVLATheta":
        return RAWVLATheta(**{field.name: getattr(self, field.name).detach() for field in fields(self)})


@dataclass
class RAWVLAOutput:
    """Detailed output of one streaming RAW-VLA step."""

    rgb: Tensor
    state: Tensor
    theta: RAWVLATheta
    theta_candidate: RAWVLATheta
    update_gate: Tensor
    denoised_raw: Tensor
    spatial_attention: Tensor


def neutral_theta(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    denoise_eta: float = DEFAULT_DENOISE_ETA,
) -> RAWVLATheta:
    """Return an identity-like ISP state for a new sequence."""

    if not 0.0 <= denoise_eta <= 1.0:
        raise ValueError("denoise_eta must be in [0, 1]")

    return RAWVLATheta(
        eta_denoise=torch.full(
            (batch_size, 1), denoise_eta, device=device, dtype=dtype
        ),
        exposure_ev=torch.zeros(batch_size, 1, device=device, dtype=dtype),
        wb_ev=torch.zeros(batch_size, 3, device=device, dtype=dtype),
        ccm_matrix=torch.eye(3, device=device, dtype=dtype).expand(batch_size, -1, -1).clone(),
        tone_logits=torch.zeros(batch_size, 3, 7, device=device, dtype=dtype),
    )


def summarize_theta(
    theta: RAWVLATheta,
    *,
    max_exposure_ev: float = MAX_EXPOSURE_EV,
    max_wb_ev: float = MAX_WB_EV,
) -> Tensor:
    """Encode the 30 predicted coordinates of ``theta``.

    The burst merge strength is fixed and therefore is not part of the
    recurrent operating-point descriptor.
    """

    if max_exposure_ev <= 0.0 or max_wb_ev <= 0.0:
        raise ValueError("max_exposure_ev and max_wb_ev must be positive")

    matrix = theta.ccm_matrix
    off_diagonal = torch.stack(
        (matrix[:, 0, 1], matrix[:, 0, 2], matrix[:, 1, 0], matrix[:, 1, 2], matrix[:, 2, 0], matrix[:, 2, 1]),
        dim=1,
    )
    return torch.cat(
        (
            theta.exposure_ev / max_exposure_ev,
            theta.wb_ev[:, :2] / max_wb_ev,
            off_diagonal / 0.25,
            (theta.tone_logits / 2.0).flatten(1),
        ),
        dim=1,
    )


def _straight_through_clamp(value: Tensor, minimum: float = 0.0, maximum: float = 1.0) -> Tensor:
    """Use hard-clamped values in forward while retaining identity gradients."""

    clipped = value.clamp(minimum, maximum)
    return value + (clipped - value).detach()


def smooth_theta(candidate: RAWVLATheta, previous: RAWVLATheta, gate: Tensor) -> RAWVLATheta:
    """Apply exposure/chroma/tone group-wise interpolation gates."""

    if gate.ndim != 2 or gate.shape[1] not in (2, 3):
        raise ValueError(f"gate must have shape [B, 2] or [B, 3], got {tuple(gate.shape)}")
    if gate.shape[1] == 2:
        # Legacy grouping: all color controls, tone.
        gate_e = gate_c = gate[:, 0:1]
        gate_t = gate[:, 1:2]
    else:
        # Split grouping: achromatic exposure, chroma, tone.
        gate_e = gate[:, 0:1]
        gate_c = gate[:, 1:2]
        gate_t = gate[:, 2:3]

    def blend(old: Tensor, new: Tensor, alpha: Tensor) -> Tensor:
        while alpha.ndim < old.ndim:
            alpha = alpha.unsqueeze(-1)
        return old + alpha * (new - old)

    return RAWVLATheta(
        # Static hyperparameter: take the current model's candidate value and
        # never blend it with a potentially stale previous checkpoint state.
        eta_denoise=candidate.eta_denoise,
        exposure_ev=blend(previous.exposure_ev, candidate.exposure_ev, gate_e),
        wb_ev=blend(previous.wb_ev, candidate.wb_ev, gate_c),
        ccm_matrix=blend(previous.ccm_matrix, candidate.ccm_matrix, gate_c),
        tone_logits=blend(previous.tone_logits, candidate.tone_logits, gate_t),
    )


class FFTConservativeMerge(nn.Module):
    """Patchwise current-frame-anchored FFT burst merge.

    Reliability is estimated independently per patch, channel, and frequency.
    The current frame is always the final frame in ``raw_burst``.
    """

    _HISTORY_WEIGHTS = {
        1: (),
        2: (0.3750,),
        3: (0.1875, 0.3750),
        4: (0.1250, 0.1875, 0.3750),
        5: (0.0625, 0.1250, 0.1875, 0.3750),
        6: (0.0500, 0.0750, 0.1250, 0.1875, 0.3750),
        7: (0.0375, 0.0500, 0.0750, 0.1250, 0.1875, 0.3750),
        8: (0.0250, 0.0375, 0.0500, 0.0750, 0.1250, 0.1875, 0.3750),
    }

    def __init__(self, patch_size: int = 32, stride: int = 16, sigma_scale: float = 1.8) -> None:
        super().__init__()
        if patch_size < 2 or stride < 1 or stride > patch_size:
            raise ValueError("Expected patch_size >= 2 and 1 <= stride <= patch_size")
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.sigma_scale = float(sigma_scale)
        window_1d = torch.hann_window(self.patch_size, periodic=False)
        self.register_buffer("window", window_1d[:, None] * window_1d[None, :], persistent=False)

    def _padding(self, height: int, width: int) -> tuple[int, int, int, int]:
        base = self.stride

        def axis_padding(size: int) -> tuple[int, int]:
            padded = size + 2 * base
            extra = max(0, self.patch_size - padded)
            if padded + extra > self.patch_size:
                extra += (self.stride - (padded + extra - self.patch_size) % self.stride) % self.stride
            before = base + extra // 2
            after = base + extra - extra // 2
            return before, after

        top, bottom = axis_padding(height)
        left, right = axis_padding(width)
        return left, right, top, bottom

    def forward(self, raw_burst: Tensor, eta_denoise: Tensor) -> Tensor:
        if raw_burst.ndim != 5 or raw_burst.shape[2] != 3:
            raise ValueError(f"raw_burst must have shape [B, K, 3, H, W], got {tuple(raw_burst.shape)}")
        batch, frames, channels, height, width = raw_burst.shape
        if frames not in self._HISTORY_WEIGHTS:
            raise ValueError(f"FFT merge supports 1--8 frames, got {frames}")
        if eta_denoise.shape != (batch, 1):
            raise ValueError(f"eta_denoise must have shape {(batch, 1)}, got {tuple(eta_denoise.shape)}")
        if frames == 1:
            return raw_burst[:, -1]

        original_dtype = raw_burst.dtype
        work = raw_burst.float() if raw_burst.dtype in (torch.float16, torch.bfloat16) else raw_burst
        left, right, top, bottom = self._padding(height, width)
        pad_mode = "reflect"
        if height <= max(top, bottom) or width <= max(left, right):
            pad_mode = "replicate"
        work = F.pad(work.flatten(0, 1), (left, right, top, bottom), mode=pad_mode)
        padded_h, padded_w = work.shape[-2:]

        patches = F.unfold(work, kernel_size=self.patch_size, stride=self.stride)
        num_patches = patches.shape[-1]
        patches = patches.transpose(1, 2).reshape(
            batch, frames, num_patches, channels, self.patch_size, self.patch_size
        )
        window = self.window.to(device=patches.device, dtype=patches.dtype)
        spectra = torch.fft.rfft2(patches * window, dim=(-2, -1))
        reference = spectra[:, -1]
        merged = reference
        correction = torch.zeros_like(reference)
        for index, base_weight in enumerate(self._HISTORY_WEIGHTS[frames]):
            difference = spectra[:, index] - reference
            power = difference.abs().square()
            sigma2 = self.sigma_scale * power.mean(dim=(-2, -1), keepdim=True)
            reliability = sigma2 / (power + sigma2 + 1.0e-8)
            correction = correction + base_weight * reliability * difference

        eta = eta_denoise.to(dtype=reference.real.dtype).view(batch, 1, 1, 1, 1)
        merged = merged + eta * correction
        merged_patches = torch.fft.irfft2(merged, s=(self.patch_size, self.patch_size), dim=(-2, -1))
        merged_patches = merged_patches.reshape(batch * num_patches, channels, self.patch_size, self.patch_size)
        merged_patches = merged_patches.reshape(batch, num_patches, channels * self.patch_size**2).transpose(1, 2)
        output = F.fold(
            merged_patches,
            output_size=(padded_h, padded_w),
            kernel_size=self.patch_size,
            stride=self.stride,
        )

        norm_patch = window.reshape(1, -1, 1).expand(1, -1, num_patches)
        norm = F.fold(
            norm_patch,
            output_size=(padded_h, padded_w),
            kernel_size=self.patch_size,
            stride=self.stride,
        )
        output = output / norm.clamp_min(1.0e-6)
        output = output[..., top : top + height, left : left + width]
        return output.to(dtype=original_dtype).clamp(0.0, 1.0)


def _normalized_luminance_weights(weights: Sequence[float]) -> Tensor:
    if len(weights) != 3:
        raise ValueError("luminance_weights must contain exactly three values")
    tensor = torch.tensor(tuple(float(value) for value in weights))
    if not torch.isfinite(tensor).all() or (tensor < 0).any() or tensor.sum() <= 0:
        raise ValueError("luminance_weights must be finite, non-negative, and have a positive sum")
    return (tensor / tensor.sum()).view(1, 3, 1, 1)


class RAWHistogram(nn.Module):
    """Linear/log RAW histograms plus compact exposure statistics."""

    def __init__(
        self,
        bins: int = 64,
        eps: float = 1.0e-6,
        luminance_weights: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        if bins < 4:
            raise ValueError("bins must be at least 4")
        self.bins = int(bins)
        self.eps = float(eps)
        self.output_dim = 7 * self.bins + 7
        self.register_buffer("luma_weights", _normalized_luminance_weights(luminance_weights), persistent=False)
        self.register_buffer(
            "quantile_levels", torch.tensor((0.01, 0.05, 0.50, 0.95, 0.99)), persistent=False
        )

    def _histogram(self, values: Tensor) -> Tensor:
        batch, channels, samples = values.shape
        indices = torch.floor(values.clamp(0.0, 1.0) * self.bins).long().clamp_max(self.bins - 1)
        hist = values.new_zeros(batch, channels, self.bins)
        hist.scatter_add_(2, indices, torch.ones_like(values))
        return hist / float(samples)

    def forward(self, raw: Tensor) -> Tensor:
        if raw.ndim != 4 or raw.shape[1] != 3:
            raise ValueError(f"raw must have shape [B, 3, H, W], got {tuple(raw.shape)}")
        raw = raw.clamp(0.0, 1.0)
        luma = (raw * self.luma_weights.to(device=raw.device, dtype=raw.dtype)).sum(dim=1, keepdim=True)
        linear = torch.cat((raw, luma), dim=1).flatten(2)
        linear_hist = self._histogram(linear).flatten(1)

        log_min = math.log(self.eps)
        log_max = math.log(1.0 + self.eps)
        log_raw = (torch.log(raw + self.eps) - log_min) / (log_max - log_min)
        log_hist = self._histogram(log_raw.flatten(2)).flatten(1)

        luma_flat = luma.flatten(1)
        quantiles = torch.quantile(
            luma_flat.float(), self.quantile_levels.to(device=raw.device), dim=1
        ).transpose(0, 1).to(dtype=raw.dtype)
        dark_ratio = (luma_flat < 0.05).to(raw.dtype).mean(dim=1, keepdim=True)
        clip_ratio = (luma_flat > 0.98).to(raw.dtype).mean(dim=1, keepdim=True)
        return torch.cat((linear_hist, log_hist, quantiles, dark_ratio, clip_ratio), dim=1)


class RAWLuminanceStatistics(nn.Module):
    """Absolute-brightness descriptor for exposure and achromatic tone.

    Unlike :class:`RAWHistogram`, this descriptor deliberately contains no
    per-channel histogram.  It carries one linear luma histogram, one absolute
    log-luma histogram, five luma quantiles, and dark/clip ratios.
    """

    def __init__(
        self,
        bins: int = 64,
        eps: float = 1.0e-6,
        luminance_weights: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        if bins < 4:
            raise ValueError("bins must be at least 4")
        self.bins = int(bins)
        self.eps = float(eps)
        self.output_dim = 2 * self.bins + 7
        self.register_buffer("luma_weights", _normalized_luminance_weights(luminance_weights), persistent=False)
        self.register_buffer(
            "quantile_levels", torch.tensor((0.01, 0.05, 0.50, 0.95, 0.99)), persistent=False
        )

    def _histogram(self, values: Tensor) -> Tensor:
        batch, channels, samples = values.shape
        indices = torch.floor(values.clamp(0.0, 1.0) * self.bins).long().clamp_max(self.bins - 1)
        hist = values.new_zeros(batch, channels, self.bins)
        hist.scatter_add_(2, indices, torch.ones_like(values))
        return hist / float(samples)

    def forward(self, raw: Tensor) -> Tensor:
        if raw.ndim != 4 or raw.shape[1] != 3:
            raise ValueError(f"raw must have shape [B, 3, H, W], got {tuple(raw.shape)}")
        raw = raw.clamp(0.0, 1.0)
        luma = (raw * self.luma_weights.to(device=raw.device, dtype=raw.dtype)).sum(dim=1, keepdim=True)
        linear_hist = self._histogram(luma.flatten(2)).flatten(1)
        log_min = math.log(self.eps)
        log_max = math.log(1.0 + self.eps)
        log_luma = (torch.log(luma + self.eps) - log_min) / (log_max - log_min)
        log_hist = self._histogram(log_luma.flatten(2)).flatten(1)
        luma_flat = luma.flatten(1)
        quantiles = torch.quantile(
            luma_flat.float(), self.quantile_levels.to(device=raw.device), dim=1
        ).transpose(0, 1).to(dtype=raw.dtype)
        dark_ratio = (luma_flat < 0.05).to(raw.dtype).mean(dim=1, keepdim=True)
        clip_ratio = (luma_flat > 0.98).to(raw.dtype).mean(dim=1, keepdim=True)
        return torch.cat((linear_hist, log_hist, quantiles, dark_ratio, clip_ratio), dim=1)


class RAWChromaStatistics(nn.Module):
    """Scale-invariant chroma descriptor for relative WB and CCM.

    RAW is first divided by its per-image mean. Consequently a common scalar
    exposure multiplier cancels before log-channel ratios and chromaticities
    are computed. Noise floors and channel-dependent clipping can still alter
    the descriptor, as they should: those effects are not achromatic exposure.
    """

    def __init__(self, bins: int = 64, eps: float = 1.0e-6, log_ratio_limit: float = 4.0) -> None:
        super().__init__()
        if bins < 4:
            raise ValueError("bins must be at least 4")
        if log_ratio_limit <= 0.0:
            raise ValueError("log_ratio_limit must be positive")
        self.bins = int(bins)
        self.eps = float(eps)
        self.log_ratio_limit = float(log_ratio_limit)
        # Five histograms (r, g, b chromaticity and log R/G, log B/G)
        # plus their means and standard deviations.
        self.output_dim = 5 * self.bins + 10

    def _histogram(self, values: Tensor) -> Tensor:
        batch, channels, samples = values.shape
        indices = torch.floor(values.clamp(0.0, 1.0) * self.bins).long().clamp_max(self.bins - 1)
        hist = values.new_zeros(batch, channels, self.bins)
        hist.scatter_add_(2, indices, torch.ones_like(values))
        return hist / float(samples)

    def forward(self, raw: Tensor) -> Tensor:
        if raw.ndim != 4 or raw.shape[1] != 3:
            raise ValueError(f"raw must have shape [B, 3, H, W], got {tuple(raw.shape)}")
        raw = raw.clamp(0.0, 1.0)
        # Dividing by a per-image scalar makes eps relative to image intensity,
        # preserving invariance to a common exposure multiplier away from the
        # numerical floor and sensor clipping.
        image_scale = raw.mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        normalized = raw / image_scale
        channel_sum = normalized.sum(dim=1, keepdim=True).clamp_min(self.eps)
        chromaticity = normalized / channel_sum
        red, green, blue = normalized.unbind(dim=1)
        log_rg = torch.log((red + self.eps) / (green + self.eps))
        log_bg = torch.log((blue + self.eps) / (green + self.eps))
        limit = self.log_ratio_limit
        log_rg = (log_rg.clamp(-limit, limit) + limit) / (2.0 * limit)
        log_bg = (log_bg.clamp(-limit, limit) + limit) / (2.0 * limit)
        values = torch.cat((chromaticity, log_rg[:, None], log_bg[:, None]), dim=1).flatten(2)
        histogram = self._histogram(values).flatten(1)
        means = values.mean(dim=2)
        stds = torch.sqrt((values - means[:, :, None]).square().mean(dim=2) + self.eps)
        return torch.cat((histogram, means, stds), dim=1)


class SpatialEncoder(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.out_channels = 4 * width
        self.layers = nn.Sequential(
            ConvBlock(3, width),
            ConvBlock(width, width),
            ConvBlock(width, 2 * width, stride=2),
            ConvBlock(2 * width, 2 * width),
            ConvBlock(2 * width, 4 * width, stride=2),
            ConvBlock(4 * width, 4 * width),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class StatisticsAttentionPool(nn.Module):
    """Mean + standard deviation + learned spatial attention pooling."""

    def __init__(self, channels: int, output_dim: int = 128) -> None:
        super().__init__()
        self.attention_score = nn.Conv2d(channels, 1, kernel_size=1)
        _zero_init(self.attention_score)
        self.projection = nn.Sequential(
            nn.Linear(3 * channels, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, output_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        mean = features.mean(dim=(-2, -1))
        variance = (features - mean[:, :, None, None]).square().mean(dim=(-2, -1))
        std = torch.sqrt(variance + 1.0e-6)
        scores = self.attention_score(features).flatten(2)
        attention = torch.softmax(scores, dim=-1)
        values = features.flatten(2)
        attended = (values * attention).sum(dim=-1)
        pooled = self.projection(torch.cat((mean, std, attended), dim=1))
        return pooled, attention.view(features.shape[0], 1, *features.shape[-2:])


def monotonic_bernstein_tone(x: Tensor, relative_interval_logits: Tensor) -> Tensor:
    """Apply an endpoint-preserving monotonic Bernstein tone curve.

    ``relative_interval_logits`` contains seven free logits for an eight-
    interval curve.  A fixed zero reference logit is appended before softmax,
    removing softmax's otherwise-unidentifiable common offset.  Zero free
    logits produce ``p_k = k / 8`` and exactly recover ``T(x) = x``.
    """

    if x.ndim != 4:
        raise ValueError(f"x must have shape [B, C, H, W], got {tuple(x.shape)}")
    if relative_interval_logits.ndim not in (3, 5):
        raise ValueError(
            "relative_interval_logits must have shape [B, C, 7] or [B, C, 7, H, W], "
            f"got {tuple(relative_interval_logits.shape)}"
        )
    if relative_interval_logits.shape[:2] != x.shape[:2]:
        raise ValueError("x and relative_interval_logits must have matching batch/channel dimensions")
    if relative_interval_logits.shape[2] != 7:
        raise ValueError("Expected seven free relative logits for eight tone intervals")
    if relative_interval_logits.ndim == 5 and relative_interval_logits.shape[-2:] != x.shape[-2:]:
        raise ValueError("Local relative_interval_logits must match the spatial size of x")

    logits = relative_interval_logits
    if logits.ndim == 3:
        logits = logits[:, :, :, None, None]
    logits = torch.cat((logits, torch.zeros_like(logits[:, :, :1])), dim=2)
    order = logits.shape[2]
    increments = torch.softmax(logits, dim=2)
    zero = torch.zeros_like(increments[:, :, :1])
    control_points = torch.cat((zero, increments.cumsum(dim=2)), dim=2)

    x = x.clamp(0.0, 1.0).unsqueeze(2)
    one_minus = 1.0 - x
    basis = torch.cat(
        [math.comb(order, k) * x.pow(k) * one_minus.pow(order - k) for k in range(order + 1)],
        dim=2,
    )
    return (control_points * basis).sum(dim=2)


def exposure_prior_loss(rgb: Tensor, *, target_mean: float = DEFAULT_EXPOSURE_TARGET) -> Tensor:
    """Normalized one-sided fourth-power prior on final ISP RGB brightness.

    The three channels and all pixels are averaged independently for every
    batch item before the losses are averaged.  Samples at or above the target
    receive exactly zero loss; increasingly dark samples receive increasingly
    stronger gradients.  ``rgb`` must be the final ISP output in ``[0, 1]``.
    """

    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must have shape [B, 3, H, W], got {tuple(rgb.shape)}")
    if not 0.0 < target_mean <= 1.0:
        raise ValueError("target_mean must be in (0, 1]")
    per_image_mean = rgb.mean(dim=(1, 2, 3))
    normalized_deficit = F.relu((target_mean - per_image_mean) / target_mean)
    return normalized_deficit.pow(4).mean()


class RAWVLA(nn.Module):
    """Streaming structured RAW ISP frontend."""

    theta_dim = 30

    def __init__(
        self,
        *,
        spatial_width: int = 32,
        hist_bins: int = 64,
        state_dim: int = 128,
        use_local_color: bool = False,
        use_local_tone: bool = False,
        fft_patch_size: int = 32,
        fft_stride: int = 16,
        burst_denoise_eta: float = DEFAULT_DENOISE_ETA,
        head_init_std: float = 1.0e-3,
        max_exposure_ev: float = MAX_EXPOSURE_EV,
        fixed_update_alpha: float | None = 1.0,
        split_luma_chroma_condition: bool = True,
        luma_state_dim: int | None = None,
        luminance_weights: Sequence[float] = (1.0, 1.0, 1.0),
        luma_spatial_input: str = "luma",
        wb_anchor_ev: Sequence[float] = (0.0, 0.0, 0.0),
        wb_residual_scale_ev: float = MAX_WB_EV,
        gray_preserving_ccm: bool = False,
        disable_burst_denoise: bool = False,
        disable_chroma_descriptor: bool = False,
        disable_luma_descriptor: bool = False,
        disable_previous_hidden: bool = False,
        disable_previous_theta: bool = False,
        disable_spatial_feature: bool = False,
        recurrent_mode: str = "split",
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.use_local_color = bool(use_local_color)
        self.use_local_tone = bool(use_local_tone)
        self.split_luma_chroma_condition = bool(split_luma_chroma_condition)
        self.luma_spatial_input = str(luma_spatial_input).lower()
        if self.luma_spatial_input not in {"luma", "rgb"}:
            raise ValueError("luma_spatial_input must be 'luma' or 'rgb'")
        if head_init_std <= 0.0:
            raise ValueError("head_init_std must be positive")
        if not 0.0 <= burst_denoise_eta <= 1.0:
            raise ValueError("burst_denoise_eta must be in [0, 1]")
        if max_exposure_ev <= 0.0:
            raise ValueError("max_exposure_ev must be positive")
        if fixed_update_alpha is not None and not 0.0 < fixed_update_alpha <= 1.0:
            raise ValueError("fixed_update_alpha must be in (0, 1]")
        wb_anchor = tuple(float(value) for value in wb_anchor_ev)
        if len(wb_anchor) != 3:
            raise ValueError("wb_anchor_ev must contain three channel EV values")
        if abs(sum(wb_anchor)) > 1.0e-5:
            raise ValueError("wb_anchor_ev must be zero-sum so exposure remains a separate scalar")
        if wb_residual_scale_ev <= 0.0:
            raise ValueError("wb_residual_scale_ev must be positive")
        self.head_init_std = float(head_init_std)
        self.burst_denoise_eta = float(burst_denoise_eta)
        self.max_exposure_ev = float(max_exposure_ev)
        self.fixed_update_alpha = (
            None if fixed_update_alpha is None else float(fixed_update_alpha)
        )
        self.register_buffer("wb_anchor_ev", torch.tensor(wb_anchor, dtype=torch.float32))
        self.wb_residual_scale_ev = float(wb_residual_scale_ev)
        self.gray_preserving_ccm = bool(gray_preserving_ccm)
        self.disable_burst_denoise = bool(disable_burst_denoise)
        self.disable_chroma_descriptor = bool(disable_chroma_descriptor)
        self.disable_luma_descriptor = bool(disable_luma_descriptor)
        self.disable_previous_hidden = bool(disable_previous_hidden)
        self.disable_previous_theta = bool(disable_previous_theta)
        self.disable_spatial_feature = bool(disable_spatial_feature)
        self.recurrent_mode = str(recurrent_mode).lower()
        if self.recurrent_mode not in {"split", "shared", "luma_only", "chroma_only"}:
            raise ValueError(
                "recurrent_mode must be one of split, shared, luma_only, chroma_only"
            )
        if not self.split_luma_chroma_condition and self.recurrent_mode != "split":
            raise ValueError("recurrent_mode ablations require split luma/chroma conditioning")

        self.spatial_encoder = SpatialEncoder(spatial_width)
        feature_channels = self.spatial_encoder.out_channels
        self.spatial_pool = StatisticsAttentionPool(feature_channels, output_dim=128)
        if not self.split_luma_chroma_condition:
            self.histogram = RAWHistogram(hist_bins, luminance_weights=luminance_weights)
            self.hist_encoder = nn.Sequential(
                nn.Linear(self.histogram.output_dim, 256),
                nn.SiLU(inplace=True),
                nn.Linear(256, 128),
                nn.SiLU(inplace=True),
                nn.Linear(128, 64),
            )
            self.theta_encoder = nn.Sequential(
                nn.Linear(self.theta_dim, 64),
                nn.SiLU(inplace=True),
                nn.Linear(64, 32),
                nn.SiLU(inplace=True),
            )
            self.fusion = nn.Sequential(
                nn.Linear(128 + 64 + 32, 256),
                nn.SiLU(inplace=True),
                nn.Linear(256, 128),
                nn.SiLU(inplace=True),
            )
            self.state_gru = nn.GRUCell(input_size=128, hidden_size=self.state_dim)

            head_input_dim = 128 + self.state_dim + 64
            self.color_head = nn.Sequential(
                nn.Linear(head_input_dim, 128), nn.SiLU(inplace=True), nn.Linear(128, 9)
            )
            self.tone_head = nn.Sequential(
                nn.Linear(head_input_dim, 128), nn.SiLU(inplace=True), nn.Linear(128, 21)
            )
            gate_input_dim = 128 + 64 + 32 + self.state_dim
            self.update_gate = nn.Sequential(
                nn.Linear(gate_input_dim, 128), nn.SiLU(inplace=True), nn.Linear(128, 2)
            )
        else:
            if self.state_dim < 2:
                raise ValueError("split luma/chroma conditioning requires state_dim >= 2")
            self.luma_state_dim = self.state_dim // 2 if luma_state_dim is None else int(luma_state_dim)
            if not 1 <= self.luma_state_dim < self.state_dim:
                raise ValueError("luma_state_dim must be in [1, state_dim - 1]")
            self.chroma_state_dim = self.state_dim - self.luma_state_dim
            self.luma_statistics = RAWLuminanceStatistics(
                hist_bins, luminance_weights=luminance_weights
            )
            self.chroma_statistics = RAWChromaStatistics(hist_bins)
            self.luma_encoder = nn.Sequential(
                nn.Linear(self.luma_statistics.output_dim, 128),
                nn.SiLU(inplace=True),
                nn.Linear(128, 64),
                nn.SiLU(inplace=True),
            )
            self.chroma_encoder = nn.Sequential(
                nn.Linear(self.chroma_statistics.output_dim, 128),
                nn.SiLU(inplace=True),
                nn.Linear(128, 64),
                nn.SiLU(inplace=True),
            )
            # Split mode uses one shared seven-parameter tone curve, so the
            # The previous luma state has exposure + seven tone coordinates.
            # Previous relative WB/CCM contributes eight chroma coordinates.
            self.luma_theta_encoder = nn.Sequential(
                nn.Linear(8, 64), nn.SiLU(inplace=True), nn.Linear(64, 32), nn.SiLU(inplace=True)
            )
            self.chroma_theta_encoder = nn.Sequential(
                nn.Linear(8, 64), nn.SiLU(inplace=True), nn.Linear(64, 32), nn.SiLU(inplace=True)
            )
            self.luma_fusion = nn.Sequential(
                nn.Linear(128 + 64 + 32, 256), nn.SiLU(inplace=True), nn.Linear(256, 128), nn.SiLU(inplace=True)
            )
            self.chroma_fusion = nn.Sequential(
                nn.Linear(64 + 32, 128), nn.SiLU(inplace=True), nn.Linear(128, 128), nn.SiLU(inplace=True)
            )
            if self.recurrent_mode == "split":
                self.luma_gru = nn.GRUCell(input_size=128, hidden_size=self.luma_state_dim)
                self.chroma_gru = nn.GRUCell(input_size=128, hidden_size=self.chroma_state_dim)
                recurrent_luma_dim = self.luma_state_dim
                recurrent_chroma_dim = self.chroma_state_dim
            elif self.recurrent_mode == "shared":
                self.shared_fusion = nn.Sequential(
                    nn.Linear(256, 256),
                    nn.SiLU(inplace=True),
                    nn.Linear(256, 128),
                    nn.SiLU(inplace=True),
                )
                self.shared_gru = nn.GRUCell(input_size=128, hidden_size=self.state_dim)
                recurrent_luma_dim = recurrent_chroma_dim = self.state_dim
            elif self.recurrent_mode == "luma_only":
                self.luma_gru_128 = nn.GRUCell(input_size=128, hidden_size=self.state_dim)
                recurrent_luma_dim = recurrent_chroma_dim = self.state_dim
            else:
                self.chroma_gru_128 = nn.GRUCell(input_size=128, hidden_size=self.state_dim)
                recurrent_luma_dim = recurrent_chroma_dim = self.state_dim
            # The descriptor embeddings are already inputs to luma_fused and
            # chroma_fused.  Feeding them directly to the decoder again creates
            # a redundant shortcut around both fusion and recurrent state.
            # Decode theta from the fused current observation and the updated
            # recurrent state only.
            luma_head_dim = 128 + recurrent_luma_dim
            chroma_head_dim = 128 + recurrent_chroma_dim
            self.exposure_head = nn.Sequential(
                nn.Linear(luma_head_dim, 128), nn.SiLU(inplace=True), nn.Linear(128, 1)
            )
            # One seven-parameter curve is shared by RGB. Chroma is owned only
            # by relative WB and CCM, so the luma path cannot create a tint.
            self.tone_head = nn.Sequential(
                nn.Linear(luma_head_dim, 128), nn.SiLU(inplace=True), nn.Linear(128, 7)
            )
            self.chroma_head = nn.Sequential(
                nn.Linear(chroma_head_dim, 128), nn.SiLU(inplace=True), nn.Linear(128, 8)
            )
            self.luma_update_gate = nn.Sequential(
                nn.Linear(128 + 64 + 32 + recurrent_luma_dim, 128),
                nn.SiLU(inplace=True),
                nn.Linear(128, 2),
            )
            self.chroma_update_gate = nn.Sequential(
                nn.Linear(64 + 32 + recurrent_chroma_dim, 128),
                nn.SiLU(inplace=True),
                nn.Linear(128, 1),
            )

        self.local_color = None
        if self.use_local_color:
            self.local_color = nn.Sequential(
                ConvBlock(feature_channels, 2 * spatial_width),
                ConvBlock(2 * spatial_width, 2 * spatial_width),
                nn.Conv2d(2 * spatial_width, 9, kernel_size=1),
            )
            _tiny_init(self.local_color[-1], std=self.head_init_std)

        self.local_tone = None
        if self.use_local_tone:
            self.local_tone = nn.Sequential(
                ConvBlock(3, spatial_width),
                ConvBlock(spatial_width, spatial_width),
                nn.Conv2d(spatial_width, 21, kernel_size=3, padding=1),
            )
            _tiny_init(self.local_tone[-1], std=self.head_init_std)

        self.fft_merge = FFTConservativeMerge(patch_size=fft_patch_size, stride=fft_stride)
        self._initialize_heads()

    def _initialize_heads(self) -> None:
        if self.split_luma_chroma_condition:
            _tiny_init(self.exposure_head[-1], std=self.head_init_std)
            _tiny_init(self.chroma_head[-1], std=self.head_init_std)
            _tiny_init(self.luma_update_gate[-1], bias=-2.0, std=self.head_init_std)
            _tiny_init(self.chroma_update_gate[-1], bias=-2.0, std=self.head_init_std)
        else:
            _tiny_init(self.color_head[-1], std=self.head_init_std)
            _tiny_init(self.update_gate[-1], bias=-2.0, std=self.head_init_std)
        _tiny_init(self.tone_head[-1], std=self.head_init_std)

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch_size, self.state_dim, device=device, dtype=dtype)

    def initial_theta(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> RAWVLATheta:
        theta = neutral_theta(
            batch_size,
            device=device,
            dtype=dtype,
            denoise_eta=self.burst_denoise_eta,
        )
        theta.wb_ev = self.wb_anchor_ev.to(device=device, dtype=dtype).expand(batch_size, -1).clone()
        return theta

    def _validate_state(self, state: Tensor, batch_size: int) -> None:
        if state.shape != (batch_size, self.state_dim):
            raise ValueError(f"state must have shape {(batch_size, self.state_dim)}, got {tuple(state.shape)}")

    def _validate_theta(self, theta: RAWVLATheta, batch_size: int) -> None:
        expected = {
            "eta_denoise": (batch_size, 1),
            "exposure_ev": (batch_size, 1),
            "wb_ev": (batch_size, 3),
            "ccm_matrix": (batch_size, 3, 3),
            "tone_logits": (batch_size, 3, 7),
        }
        for name, shape in expected.items():
            value = getattr(theta, name)
            if value.shape != shape:
                raise ValueError(f"theta.{name} must have shape {shape}, got {tuple(value.shape)}")

    def _decode_chroma(self, chroma_raw: Tensor) -> tuple[Tensor, Tensor]:
        batch = chroma_raw.shape[0]
        wb_raw = torch.tanh(chroma_raw[:, 0:2])
        off_diagonal_raw = chroma_raw[:, 2:8]
        # Two predicted coordinates generate three zero-sum relative WB EVs.
        # Max-norm normalization only activates near the boundary and guarantees
        # every channel remains within +/- MAX_WB_EV.
        wb_unscaled = torch.stack((wb_raw[:, 0], wb_raw[:, 1], -wb_raw.sum(dim=1)), dim=1)
        wb_scale = wb_unscaled.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
        wb_residual_ev = self.wb_residual_scale_ev * wb_unscaled / wb_scale
        wb_ev = wb_residual_ev + self.wb_anchor_ev.to(dtype=chroma_raw.dtype)[None]

        identity = torch.eye(3, device=chroma_raw.device, dtype=chroma_raw.dtype).expand(batch, -1, -1)
        off_diagonal_basis = chroma_raw.new_tensor(
            (
                ((0, 1, 0), (0, 0, 0), (0, 0, 0)),
                ((0, 0, 1), (0, 0, 0), (0, 0, 0)),
                ((0, 0, 0), (1, 0, 0), (0, 0, 0)),
                ((0, 0, 0), (0, 0, 1), (0, 0, 0)),
                ((0, 0, 0), (0, 0, 0), (1, 0, 0)),
                ((0, 0, 0), (0, 0, 0), (0, 1, 0)),
            )
        )
        delta_matrix = torch.einsum("bk,kij->bij", 0.25 * torch.tanh(off_diagonal_raw), off_diagonal_basis)
        if self.gray_preserving_ccm:
            # Constrain A @ [1, 1, 1]^T = [1, 1, 1]^T. The six learned
            # off-diagonals remain free, while each diagonal compensates its
            # row sum; neutral gray therefore cannot be tinted by the CCM.
            delta_matrix = delta_matrix - torch.diag_embed(delta_matrix.sum(dim=2))
        ccm_matrix = identity + delta_matrix
        return wb_ev, ccm_matrix

    def _predict_theta(self, condition: Tensor, chroma_condition: Tensor | None = None) -> RAWVLATheta:
        batch = condition.shape[0]
        eta = condition.new_full((batch, 1), self.burst_denoise_eta)
        if not self.split_luma_chroma_condition:
            color_raw = self.color_head(condition)
            exposure_raw = color_raw[:, 0:1]
            wb_ev, ccm_matrix = self._decode_chroma(color_raw[:, 1:9])
            tone_logits = 2.0 * torch.tanh(self.tone_head(condition).view(batch, 3, 7))
        else:
            if chroma_condition is None:
                raise ValueError("split luma/chroma conditioning requires chroma_condition")
            exposure_raw = self.exposure_head(condition)
            wb_ev, ccm_matrix = self._decode_chroma(self.chroma_head(chroma_condition))
            shared_tone = 2.0 * torch.tanh(self.tone_head(condition).view(batch, 1, 7))
            tone_logits = shared_tone.expand(-1, 3, -1)
        exposure_ev = self.max_exposure_ev * torch.tanh(exposure_raw)
        return RAWVLATheta(eta, exposure_ev, wb_ev, ccm_matrix, tone_logits)

    def _split_theta_summaries(self, theta: RAWVLATheta) -> tuple[Tensor, Tensor]:
        matrix = theta.ccm_matrix
        off_diagonal = torch.stack(
            (matrix[:, 0, 1], matrix[:, 0, 2], matrix[:, 1, 0], matrix[:, 1, 2], matrix[:, 2, 0], matrix[:, 2, 1]),
            dim=1,
        )
        luma = torch.cat(
            (
                theta.exposure_ev / self.max_exposure_ev,
                (theta.tone_logits[:, :1] / 2.0).flatten(1),
            ),
            dim=1,
        )
        # Keep the two independent zero-sum WB coordinates and six CCM
        # off-diagonals; no exposure/tone information enters this path.
        chroma = torch.cat((theta.wb_ev[:, :2] / MAX_WB_EV, off_diagonal / 0.25), dim=1)
        return luma, chroma

    def _apply_color(self, raw: Tensor, theta: RAWVLATheta, features: Tensor) -> Tensor:
        exposure_gain = theta.exposure_gain[:, :, None, None]
        color = torch.einsum("boi,bihw->bohw", theta.color_matrix, raw)
        if self.local_color is not None:
            local = 0.05 * torch.tanh(self.local_color(features))
            local = F.interpolate(local, size=raw.shape[-2:], mode="bilinear", align_corners=False)
            local = local.view(raw.shape[0], 3, 3, *raw.shape[-2:])
            # Remove the local isotropic diagonal component so exposure scale
            # remains owned by the scalar exposure parameter.
            common_scale = local.diagonal(dim1=1, dim2=2).mean(dim=-1)
            eye = torch.eye(3, device=raw.device, dtype=raw.dtype)[None, :, :, None, None]
            local = local - common_scale[:, None, None, :, :] * eye
            color = color + torch.einsum("boihw,bihw->bohw", local, raw)
        return exposure_gain * color

    def _apply_tone(self, color: Tensor, theta: RAWVLATheta, *, clamp_mode: str = "hard") -> Tensor:
        if clamp_mode == "hard":
            clamp = lambda value: value.clamp(0.0, 1.0)
        elif clamp_mode == "ste":
            clamp = _straight_through_clamp
        else:
            raise ValueError(f"clamp_mode must be 'hard' or 'ste', got {clamp_mode!r}")
        x = clamp(color)
        logits = theta.tone_logits
        if self.local_tone is not None:
            local_logits = 0.5 * torch.tanh(
                self.local_tone(x).view(x.shape[0], 3, 7, *x.shape[-2:])
            )
            logits = logits[:, :, :, None, None] + local_logits
        return clamp(monotonic_bernstein_tone(x, logits))

    def exposure_objective_rgb(self, output: RAWVLAOutput, *, clamp_mode: str = "hard") -> Tensor:
        """Render the image used by the scalar exposure objective.

        In split mode, relative WB and CCM are treated as constants for this
        auxiliary path. The loss value still comes from a final RGB rendering,
        but its gradient cannot brighten an image by tinting it. The downstream
        action loss continues to update every ISP parameter through
        ``output.rgb``.
        """

        if not self.split_luma_chroma_condition and clamp_mode == "hard":
            return output.rgb
        theta = RAWVLATheta(
            eta_denoise=output.theta.eta_denoise,
            exposure_ev=output.theta.exposure_ev,
            wb_ev=output.theta.wb_ev.detach(),
            ccm_matrix=output.theta.ccm_matrix.detach(),
            tone_logits=output.theta.tone_logits,
        )
        color = theta.exposure_gain[:, :, None, None] * torch.einsum(
            "boi,bihw->bohw", theta.color_matrix, output.denoised_raw
        )
        return self._apply_tone(color, theta, clamp_mode=clamp_mode)

    def chroma_objective_rgb(self, output: RAWVLAOutput, *, clamp_mode: str = "ste") -> Tensor:
        """Render a chroma-only auxiliary path.

        Denoise, scalar exposure, and tone parameters are constants on this
        path. Gradients therefore reach relative WB/CCM and their chroma
        recurrent state, without turning a weak color prior into another
        brightness or denoising objective.
        """

        theta = RAWVLATheta(
            eta_denoise=output.theta.eta_denoise.detach(),
            exposure_ev=output.theta.exposure_ev.detach(),
            wb_ev=output.theta.wb_ev,
            ccm_matrix=output.theta.ccm_matrix,
            tone_logits=output.theta.tone_logits.detach(),
        )
        color = theta.exposure_gain[:, :, None, None] * torch.einsum(
            "boi,bihw->bohw", theta.color_matrix, output.denoised_raw.detach()
        )
        return self._apply_tone(color, theta, clamp_mode=clamp_mode)

    def _advance_operating_point(
        self,
        x_ref: Tensor,
        state: Tensor | None,
        theta_prev: RAWVLATheta | None,
    ) -> tuple[Tensor, RAWVLATheta, RAWVLATheta, Tensor, Tensor, Tensor]:
        """Advance recurrent ISP controls from one current RAW frame."""

        if x_ref.ndim != 4 or x_ref.shape[1] != 3:
            raise ValueError(f"x_ref must have shape [B, 3, H, W], got {tuple(x_ref.shape)}")
        batch = x_ref.shape[0]
        x_ref = x_ref.clamp(0.0, 1.0)
        if state is None:
            state = self.initial_state(batch, device=x_ref.device, dtype=x_ref.dtype)
        if theta_prev is None:
            theta_prev = self.initial_theta(batch, device=x_ref.device, dtype=x_ref.dtype)
        if self.disable_previous_hidden:
            state = self.initial_state(batch, device=x_ref.device, dtype=x_ref.dtype)
        if self.disable_previous_theta:
            theta_prev = self.initial_theta(batch, device=x_ref.device, dtype=x_ref.dtype)
        self._validate_state(state, batch)
        self._validate_theta(theta_prev, batch)

        spatial_input = x_ref
        if self.split_luma_chroma_condition and self.luma_spatial_input == "luma":
            # The luma recurrent path must not recover chroma through the RGB
            # spatial CNN.  Use the same normalized equal-channel statistic as
            # RAWLuminanceStatistics and replicate it only for CNN shape
            # compatibility.
            luma = (
                x_ref
                * self.luma_statistics.luma_weights.to(device=x_ref.device, dtype=x_ref.dtype)
            ).sum(dim=1, keepdim=True)
            spatial_input = luma.expand(-1, 3, -1, -1)
        features = self.spatial_encoder(spatial_input)
        global_feature, spatial_attention = self.spatial_pool(features)
        if self.disable_spatial_feature:
            global_feature = torch.zeros_like(global_feature)
        if not self.split_luma_chroma_condition:
            histogram_embedding = self.hist_encoder(self.histogram(x_ref))
            theta_embedding = self.theta_encoder(
                summarize_theta(theta_prev, max_exposure_ev=self.max_exposure_ev)
            )
            fused = self.fusion(torch.cat((global_feature, histogram_embedding, theta_embedding), dim=1))
            new_state = self.state_gru(fused, state)

            head_condition = torch.cat((fused, new_state, histogram_embedding), dim=1)
            theta_candidate = self._predict_theta(head_condition)
            gate_condition = torch.cat((global_feature, histogram_embedding, theta_embedding, new_state), dim=1)
            update_gate = torch.sigmoid(self.update_gate(gate_condition))
        else:
            luma_state, chroma_state = torch.split(
                state, (self.luma_state_dim, self.chroma_state_dim), dim=1
            )
            luma_embedding = self.luma_encoder(self.luma_statistics(x_ref))
            chroma_embedding = self.chroma_encoder(self.chroma_statistics(x_ref))
            if self.disable_luma_descriptor:
                luma_embedding = torch.zeros_like(luma_embedding)
            if self.disable_chroma_descriptor:
                chroma_embedding = torch.zeros_like(chroma_embedding)
            luma_theta, chroma_theta = self._split_theta_summaries(theta_prev)
            luma_theta_embedding = self.luma_theta_encoder(luma_theta)
            chroma_theta_embedding = self.chroma_theta_encoder(chroma_theta)
            luma_fused = self.luma_fusion(
                torch.cat((global_feature, luma_embedding, luma_theta_embedding), dim=1)
            )
            chroma_fused = self.chroma_fusion(
                torch.cat((chroma_embedding, chroma_theta_embedding), dim=1)
            )
            if self.recurrent_mode == "split":
                new_luma_state = self.luma_gru(luma_fused, luma_state)
                new_chroma_state = self.chroma_gru(chroma_fused, chroma_state)
                new_state = torch.cat((new_luma_state, new_chroma_state), dim=1)
            elif self.recurrent_mode == "shared":
                shared_fused = self.shared_fusion(torch.cat((luma_fused, chroma_fused), dim=1))
                new_state = self.shared_gru(shared_fused, state)
                new_luma_state = new_chroma_state = new_state
            elif self.recurrent_mode == "luma_only":
                new_state = self.luma_gru_128(luma_fused, state)
                new_luma_state = new_chroma_state = new_state
            else:
                new_state = self.chroma_gru_128(chroma_fused, state)
                new_luma_state = new_chroma_state = new_state
            luma_condition = torch.cat((luma_fused, new_luma_state), dim=1)
            chroma_condition = torch.cat((chroma_fused, new_chroma_state), dim=1)
            theta_candidate = self._predict_theta(luma_condition, chroma_condition)
            luma_gate_condition = torch.cat(
                (global_feature, luma_embedding, luma_theta_embedding, new_luma_state), dim=1
            )
            chroma_gate_condition = torch.cat(
                (chroma_embedding, chroma_theta_embedding, new_chroma_state), dim=1
            )
            luma_gates = torch.sigmoid(self.luma_update_gate(luma_gate_condition))
            chroma_gate = torch.sigmoid(self.chroma_update_gate(chroma_gate_condition))
            update_gate = torch.cat(
                (luma_gates[:, 0:1], chroma_gate, luma_gates[:, 1:2]), dim=1
            )
        if self.fixed_update_alpha is not None:
            update_gate = torch.full_like(update_gate, self.fixed_update_alpha)
        theta = smooth_theta(theta_candidate, theta_prev, update_gate)
        return new_state, theta, theta_candidate, update_gate, features, spatial_attention

    def advance_state(
        self,
        raw_current: Tensor,
        state: Tensor | None = None,
        theta_prev: RAWVLATheta | None = None,
    ) -> tuple[Tensor, RAWVLATheta]:
        """Advance streaming state without rendering an intermediate RGB frame."""

        new_state, theta, _, _, _, _ = self._advance_operating_point(raw_current, state, theta_prev)
        return new_state, theta

    def forward(
        self,
        raw_burst: Tensor,
        state: Tensor | None = None,
        theta_prev: RAWVLATheta | None = None,
    ) -> RAWVLAOutput:
        """Process one causal burst.

        Args:
            raw_burst: Linear RGB RAW in ``[B, K, 3, H, W]``, with the current
                reference frame at index ``K - 1``.
            state: Previous GRU state.  ``None`` starts a new sequence.
            theta_prev: Previous explicit ISP state.  ``None`` uses the neutral
                identity-like ISP state.
        """

        if raw_burst.ndim != 5 or raw_burst.shape[2] != 3:
            raise ValueError(f"raw_burst must have shape [B, K, 3, H, W], got {tuple(raw_burst.shape)}")
        x_ref = raw_burst[:, -1].clamp(0.0, 1.0)
        new_state, theta, theta_candidate, update_gate, features, spatial_attention = (
            self._advance_operating_point(x_ref, state, theta_prev)
        )

        if self.disable_burst_denoise:
            denoised = x_ref
        else:
            denoised = self.fft_merge(raw_burst.clamp(0.0, 1.0), theta.eta_denoise)
        color = self._apply_color(denoised, theta, features)
        rgb = self._apply_tone(color, theta)
        return RAWVLAOutput(rgb, new_state, theta, theta_candidate, update_gate, denoised, spatial_attention)
