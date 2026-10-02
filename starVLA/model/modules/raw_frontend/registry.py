"""Small registry for RAW-to-RGB frontend baselines.

The benchmark owns RAW formation. This module only adapts those RAW tensors
inside StarVLA so action loss can train the frontend end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn import functional as F


_WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
if (_WORKSPACE_ROOT / "baselines").exists() and str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, DictConfig):
        return OmegaConf.select(cfg, key, default=default)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def is_raw_frontend_enabled(cfg: Any) -> bool:
    name = str(_cfg_get(cfg, "name", "none") or "none").lower()
    return name not in {"", "none", "identity", "rgb"}


class IdentityRAWFrontend(nn.Module):
    def forward(self, raw: torch.Tensor, *, input_format: str = "rgb_raw", **_: Any) -> torch.Tensor:
        if raw.ndim != 4 or raw.shape[1] != 3:
            raise ValueError(f"identity RAW frontend expects [B, 3, H, W], got {tuple(raw.shape)}")
        return raw.float().clamp(0.0, 1.0)


class _DarkISPFrontend(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        from baselines.darkisp import DarkISP

        if DarkISP is None:
            raise ImportError("baselines.darkisp.DarkISP is unavailable; check torch installation.")
        self.model = DarkISP(
            hidden_dim=int(_cfg_get(cfg, "hidden_dim", 48)),
            max_linear_residual=float(_cfg_get(cfg, "max_linear_residual", 0.25)),
            tone_coefficient_scale=float(_cfg_get(cfg, "tone_coefficient_scale", 0.25)),
        )

    def forward(self, raw: torch.Tensor, *, input_format: str = "rgb_raw", **_: Any) -> torch.Tensor:
        return self.model(raw, input_format=input_format).rgb


class _RAWVLAFrontend(nn.Module):
    """StarVLA adapter for the streaming RAWVLA frontend.

    Training samples carry a causal sequence of policy observations, each of
    which owns a short adjacent-frame denoising burst. The adapter replays the
    outer sequence so recurrent state follows policy cadence, while the inner
    sequence is used only by burst merging.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        from baselines.rawvla import RAWVLA, exposure_prior_loss

        if RAWVLA is None or exposure_prior_loss is None:
            raise ImportError("baselines.rawvla is unavailable; check torch installation.")
        self.model = RAWVLA(
            spatial_width=int(_cfg_get(cfg, "spatial_width", 32)),
            hist_bins=int(_cfg_get(cfg, "hist_bins", 64)),
            state_dim=int(_cfg_get(cfg, "state_dim", 128)),
            use_local_color=bool(_cfg_get(cfg, "use_local_color", False)),
            use_local_tone=bool(_cfg_get(cfg, "use_local_tone", False)),
            fft_patch_size=int(_cfg_get(cfg, "fft_patch_size", 32)),
            fft_stride=int(_cfg_get(cfg, "fft_stride", 16)),
            burst_denoise_eta=float(_cfg_get(cfg, "burst_denoise_eta", 0.5)),
            head_init_std=float(_cfg_get(cfg, "head_init_std", 1.0e-3)),
            max_exposure_ev=float(_cfg_get(cfg, "max_exposure_ev", 6.0)),
            fixed_update_alpha=_cfg_get(cfg, "fixed_update_alpha", 1.0),
            split_luma_chroma_condition=bool(_cfg_get(cfg, "split_luma_chroma_condition", True)),
            luma_state_dim=_cfg_get(cfg, "luma_state_dim", None),
            luminance_weights=_cfg_get(cfg, "luminance_weights", (1.0, 1.0, 1.0)),
            luma_spatial_input=str(_cfg_get(cfg, "luma_spatial_input", "luma")),
            wb_anchor_ev=_cfg_get(cfg, "wb_anchor_ev", (0.0, 0.0, 0.0)),
            wb_residual_scale_ev=float(_cfg_get(cfg, "wb_residual_scale_ev", 1.0)),
            gray_preserving_ccm=bool(_cfg_get(cfg, "gray_preserving_ccm", False)),
            disable_burst_denoise=bool(_cfg_get(cfg, "disable_burst_denoise", False)),
            disable_chroma_descriptor=bool(_cfg_get(cfg, "disable_chroma_descriptor", False)),
            disable_luma_descriptor=bool(_cfg_get(cfg, "disable_luma_descriptor", False)),
            disable_previous_hidden=bool(_cfg_get(cfg, "disable_previous_hidden", False)),
            disable_previous_theta=bool(_cfg_get(cfg, "disable_previous_theta", False)),
            disable_spatial_feature=bool(_cfg_get(cfg, "disable_spatial_feature", False)),
            recurrent_mode=str(_cfg_get(cfg, "recurrent_mode", "split")),
        )
        self.burst_frames = int(_cfg_get(cfg, "burst_frames", 6))
        self.rnn_frames = int(_cfg_get(cfg, "rnn_frames", 6))
        if not 1 <= self.burst_frames <= 8:
            raise ValueError("RAWVLA burst_frames must be in [1, 8]")
        if self.rnn_frames < 1:
            raise ValueError("RAWVLA rnn_frames must be positive")
        self.exposure_target = float(_cfg_get(cfg, "exposure_target", 0.48))
        domain_targets = _cfg_get(cfg, "exposure_target_by_lighting_domain", {}) or {}
        if isinstance(domain_targets, DictConfig):
            domain_targets = OmegaConf.to_container(domain_targets, resolve=True)
        elif hasattr(domain_targets, "items"):
            domain_targets = dict(domain_targets.items())
        if not isinstance(domain_targets, dict):
            raise ValueError("exposure_target_by_lighting_domain must be a mapping")
        self.exposure_target_by_lighting_domain = {
            str(domain).lower(): float(target) for domain, target in domain_targets.items()
        }
        if not 0.0 < self.exposure_target <= 1.0:
            raise ValueError("exposure_target must be in (0, 1]")
        if any(
            not 0.0 < target <= 1.0
            for target in self.exposure_target_by_lighting_domain.values()
        ):
            raise ValueError(
                "every exposure_target_by_lighting_domain value must be in (0, 1]"
            )
        self.exposure_loss_type = str(_cfg_get(cfg, "exposure_loss_type", "two_sided_l1")).lower()
        self.exposure_objective_clamp = str(_cfg_get(cfg, "exposure_objective_clamp", "ste")).lower()
        self.exposure_smooth_l1_beta = float(_cfg_get(cfg, "exposure_smooth_l1_beta", 0.1))
        self.extreme_low_exposure_multiplier = float(
            _cfg_get(cfg, "extreme_low_exposure_multiplier", 1.0)
        )
        if self.extreme_low_exposure_multiplier < 1.0:
            raise ValueError("extreme_low_exposure_multiplier must be at least 1.0")
        domain_multipliers = _cfg_get(cfg, "exposure_loss_multiplier_by_lighting_domain", {}) or {}
        if isinstance(domain_multipliers, DictConfig):
            domain_multipliers = OmegaConf.to_container(domain_multipliers, resolve=True)
        elif hasattr(domain_multipliers, "items"):
            domain_multipliers = dict(domain_multipliers.items())
        if not isinstance(domain_multipliers, dict):
            raise ValueError("exposure_loss_multiplier_by_lighting_domain must be a mapping")
        self.exposure_loss_multiplier_by_lighting_domain = {
            str(domain).lower(): float(multiplier)
            for domain, multiplier in domain_multipliers.items()
        }
        self.exposure_loss_multiplier_by_lighting_domain.setdefault(
            "extremelow", self.extreme_low_exposure_multiplier
        )
        if any(
            multiplier <= 0.0
            for multiplier in self.exposure_loss_multiplier_by_lighting_domain.values()
        ):
            raise ValueError("every domain exposure-loss multiplier must be positive")
        if self.exposure_loss_type not in {"quartic", "smooth_l1", "l1", "two_sided_l1"}:
            raise ValueError(
                "exposure_loss_type must be 'quartic', 'smooth_l1', or 'two_sided_l1'"
            )
        if self.exposure_objective_clamp not in {"hard", "ste"}:
            raise ValueError("exposure_objective_clamp must be 'hard' or 'ste'")
        self.chroma_prior_enabled = bool(_cfg_get(cfg, "chroma_prior_enabled", True))
        self.chroma_prior_mode = str(
            _cfg_get(cfg, "chroma_prior_mode", "output_envelope")
        ).lower()
        if self.chroma_prior_mode not in {"output_envelope", "wb_anchor"}:
            raise ValueError("chroma_prior_mode must be 'output_envelope' or 'wb_anchor'")
        self.wb_anchor_loss_scale_ev = float(_cfg_get(cfg, "wb_anchor_loss_scale_ev", 0.5))
        if self.wb_anchor_loss_scale_ev <= 0.0:
            raise ValueError("wb_anchor_loss_scale_ev must be positive")
        self.chroma_prior_descriptor = str(
            _cfg_get(cfg, "chroma_prior_descriptor", "mean_log")
        ).lower()
        if self.chroma_prior_descriptor not in {"mean_log", "gray_edge"}:
            raise ValueError(
                "chroma_prior_descriptor must be 'mean_log' or 'gray_edge'"
            )
        self.chroma_prior_bias = tuple(float(value) for value in _cfg_get(
            cfg, "chroma_prior_bias", (0.0, 0.0)
        ))
        self.chroma_prior_lower = tuple(float(value) for value in _cfg_get(
            cfg, "chroma_prior_lower", (0.025, -0.358)
        ))
        self.chroma_prior_upper = tuple(float(value) for value in _cfg_get(
            cfg, "chroma_prior_upper", (0.257, -0.032)
        ))
        self.chroma_prior_margin = float(_cfg_get(cfg, "chroma_prior_margin", 0.10))
        self.chroma_supervision_beta = float(
            _cfg_get(cfg, "chroma_supervision_beta", 0.05)
        )
        self.chroma_supervision_min_luma = float(
            _cfg_get(cfg, "chroma_supervision_min_luma", 0.02)
        )
        self.chroma_supervision_max_luma = float(
            _cfg_get(cfg, "chroma_supervision_max_luma", 0.98)
        )
        self.chroma_supervision_highlight_luma_threshold = float(
            _cfg_get(cfg, "chroma_supervision_highlight_luma_threshold", 0.8)
        )
        highlight_domain_multipliers = _cfg_get(
            cfg, "chroma_supervision_highlight_multiplier_by_lighting_domain", {}
        ) or {}
        if isinstance(highlight_domain_multipliers, DictConfig):
            highlight_domain_multipliers = OmegaConf.to_container(
                highlight_domain_multipliers, resolve=True
            )
        elif hasattr(highlight_domain_multipliers, "items"):
            highlight_domain_multipliers = dict(highlight_domain_multipliers.items())
        if not isinstance(highlight_domain_multipliers, dict):
            raise ValueError(
                "chroma_supervision_highlight_multiplier_by_lighting_domain must be a mapping"
            )
        self.chroma_supervision_highlight_multiplier_by_lighting_domain = {
            str(domain).lower(): float(multiplier)
            for domain, multiplier in highlight_domain_multipliers.items()
        }
        chroma_domain_multipliers = _cfg_get(
            cfg, "chroma_supervision_multiplier_by_lighting_domain", {}
        ) or {}
        if isinstance(chroma_domain_multipliers, DictConfig):
            chroma_domain_multipliers = OmegaConf.to_container(
                chroma_domain_multipliers, resolve=True
            )
        elif hasattr(chroma_domain_multipliers, "items"):
            chroma_domain_multipliers = dict(chroma_domain_multipliers.items())
        if not isinstance(chroma_domain_multipliers, dict):
            raise ValueError(
                "chroma_supervision_multiplier_by_lighting_domain must be a mapping"
            )
        self.chroma_supervision_multiplier_by_lighting_domain = {
            str(domain).lower(): float(multiplier)
            for domain, multiplier in chroma_domain_multipliers.items()
        }
        if any(
            multiplier <= 0.0
            for multiplier in self.chroma_supervision_multiplier_by_lighting_domain.values()
        ):
            raise ValueError("every domain chroma-supervision multiplier must be positive")
        if not 0.0 <= self.chroma_supervision_highlight_luma_threshold <= 1.0:
            raise ValueError("chroma highlight luma threshold must be in [0, 1]")
        if any(
            multiplier < 1.0
            for multiplier in self.chroma_supervision_highlight_multiplier_by_lighting_domain.values()
        ):
            raise ValueError("every domain highlight-chroma multiplier must be at least 1")
        if (
            len(self.chroma_prior_lower) != 2
            or len(self.chroma_prior_upper) != 2
            or len(self.chroma_prior_bias) != 2
        ):
            raise ValueError("chroma prior lower/upper bounds must each contain log(R/G), log(B/G)")
        if any(low >= high for low, high in zip(self.chroma_prior_lower, self.chroma_prior_upper)):
            raise ValueError("each chroma prior lower bound must be below its upper bound")
        if self.chroma_prior_margin < 0.0:
            raise ValueError("chroma_prior_margin must be non-negative")
        if self.chroma_supervision_beta <= 0.0:
            raise ValueError("chroma_supervision_beta must be positive")
        if not 0.0 <= self.chroma_supervision_min_luma < self.chroma_supervision_max_luma <= 1.0:
            raise ValueError(
                "chroma supervision luma thresholds must satisfy 0 <= min < max <= 1"
            )
        self._exposure_prior_loss = exposure_prior_loss
        self.last_aux_losses: dict[str, torch.Tensor] = {}

    def _chroma_descriptor(self, chroma_rgb: torch.Tensor, raw_reference: torch.Tensor) -> torch.Tensor:
        """Return a two-axis absolute-chroma descriptor for the weak prior.

        ``gray_edge`` removes a fixed generic-dataset median after measuring
        per-channel edge energy. Unlike per-image centering, this preserves the
        additive log-ratio shift caused by a global WB/CCM color cast.
        """

        valid = (
            (raw_reference.mean(dim=1) > 0.01)
            & (raw_reference.amax(dim=1) < 0.995)
        ).to(dtype=chroma_rgb.dtype)
        if self.chroma_prior_descriptor == "mean_log":
            log_rgb = chroma_rgb.clamp_min(1.0 / 255.0).log()
            log_ratios = torch.stack(
                (log_rgb[:, 0] - log_rgb[:, 1], log_rgb[:, 2] - log_rgb[:, 1]), dim=1
            )
            denominator = valid.sum(dim=(1, 2), keepdim=False).clamp_min(1.0)
            descriptor = (log_ratios * valid[:, None]).sum(dim=(2, 3)) / denominator[:, None]
        else:
            dx = (chroma_rgb[:, :, :, 1:] - chroma_rgb[:, :, :, :-1]).abs()
            dy = (chroma_rgb[:, :, 1:, :] - chroma_rgb[:, :, :-1, :]).abs()
            valid_x = valid[:, :, 1:] * valid[:, :, :-1]
            valid_y = valid[:, 1:, :] * valid[:, :-1, :]
            denominator = (
                valid_x.sum(dim=(1, 2)) + valid_y.sum(dim=(1, 2))
            ).clamp_min(1.0)
            energy = (
                (dx * valid_x[:, None]).sum(dim=(2, 3))
                + (dy * valid_y[:, None]).sum(dim=(2, 3))
            ) / denominator[:, None]
            log_energy = energy.clamp_min(1.0e-6).log()
            descriptor = torch.stack(
                (log_energy[:, 0] - log_energy[:, 1], log_energy[:, 2] - log_energy[:, 1]),
                dim=1,
            )
        bias = descriptor.new_tensor(self.chroma_prior_bias)
        return descriptor - bias[None]

    def forward(
        self,
        raw: torch.Tensor,
        *,
        input_format: str = "rgb_raw",
        lighting_domains: Any = None,
        chroma_targets: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if input_format.lower() not in {"rgb_raw", "raw_rgb", "pseudo_rgb"}:
            raise ValueError(f"RAWVLA requires three-channel RGB RAW, got input_format={input_format!r}")
        if raw.ndim == 4:
            raw = raw[:, None, None]
        elif raw.ndim == 5:
            # Backward-compatible single recurrent observation.
            raw = raw[:, None]
        if raw.ndim != 6 or raw.shape[3] != 3:
            raise ValueError(
                "RAWVLA frontend expects [B, T_rnn, K_burst, 3, H, W], "
                f"got {tuple(raw.shape)}"
            )
        model_param = next(self.model.parameters())
        raw = raw[:, -self.rnn_frames :, -self.burst_frames :].to(
            dtype=model_param.dtype
        ).clamp(0.0, 1.0)

        state = None
        theta = None
        output = None
        for observation_index in range(raw.shape[1]):
            output = self.model(raw[:, observation_index], state=state, theta_prev=theta)
            state, theta = output.state, output.theta
        assert output is not None
        exposure_rgb = self.model.exposure_objective_rgb(
            output, clamp_mode=self.exposure_objective_clamp
        )
        exposure_sample_weights = exposure_rgb.new_ones(exposure_rgb.shape[0])
        exposure_sample_targets = exposure_rgb.new_full(
            (exposure_rgb.shape[0],), self.exposure_target
        )
        if lighting_domains is not None:
            if len(lighting_domains) != exposure_rgb.shape[0]:
                raise ValueError(
                    "lighting_domains must have one entry per flattened RAW view: "
                    f"got {len(lighting_domains)} for {exposure_rgb.shape[0]} views"
                )
            exposure_sample_weights = exposure_rgb.new_tensor(
                [
                    self.exposure_loss_multiplier_by_lighting_domain.get(
                        str(domain).lower(), 1.0
                    )
                    for domain in lighting_domains
                ]
            )
            exposure_sample_targets = exposure_rgb.new_tensor(
                [
                    self.exposure_target_by_lighting_domain.get(
                        str(domain).lower(), self.exposure_target
                    )
                    for domain in lighting_domains
                ]
            )
        per_image_mean = exposure_rgb.mean(dim=(1, 2, 3))
        if self.exposure_loss_type == "quartic":
            normalized_deficit = torch.relu(
                (exposure_sample_targets - per_image_mean) / exposure_sample_targets
            )
            exposure_loss = (
                normalized_deficit.pow(4) * exposure_sample_weights
            ).mean()
        elif self.exposure_loss_type == "smooth_l1":
            deficit = torch.relu(
                (exposure_sample_targets - per_image_mean) / exposure_sample_targets
            )
            per_image_loss = torch.nn.functional.smooth_l1_loss(
                deficit,
                torch.zeros_like(deficit),
                beta=self.exposure_smooth_l1_beta,
                reduction="none",
            )
            exposure_loss = (per_image_loss * exposure_sample_weights).mean()
        else:
            per_image_loss = (per_image_mean - exposure_sample_targets).abs()
            exposure_loss = (per_image_loss * exposure_sample_weights).mean()
        self.last_aux_losses = {
            "exposure_prior_loss": exposure_loss,
            "exposure_sample_weight_mean": exposure_sample_weights.mean().detach(),
            "exposure_sample_target_mean": exposure_sample_targets.mean().detach(),
            "isp_output_mean": output.rgb.mean().detach(),
            "isp_exposure_ev_mean": output.theta.exposure_ev.mean().detach(),
            "isp_update_gate_mean": output.update_gate.mean().detach(),
            "isp_saturation_ratio": (output.rgb >= 0.995).float().mean().detach(),
        }
        if chroma_targets is not None:
            if chroma_targets.ndim != 4 or chroma_targets.shape[1] != 3:
                raise ValueError(
                    "RAWVLA chroma targets must be [B, 3, H, W], "
                    f"got {tuple(chroma_targets.shape)}"
                )
            if chroma_targets.shape[0] != output.rgb.shape[0]:
                raise ValueError(
                    "RAWVLA chroma target batch size does not match output: "
                    f"{chroma_targets.shape[0]} vs {output.rgb.shape[0]}"
                )
            target = chroma_targets.detach().to(device=output.rgb.device, dtype=output.rgb.dtype)
            if target.shape[-2:] != output.rgb.shape[-2:]:
                target = F.interpolate(
                    target,
                    size=output.rgb.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            target = target.clamp(0.0, 1.0)
            predicted = output.rgb.clamp(0.0, 1.0)
            predicted_chroma = predicted / predicted.sum(dim=1, keepdim=True).clamp_min(1.0e-4)
            target_chroma = target / target.sum(dim=1, keepdim=True).clamp_min(1.0e-4)
            target_luma = target.mean(dim=1)
            valid = (
                (target_luma >= self.chroma_supervision_min_luma)
                & (target_luma <= self.chroma_supervision_max_luma)
            ).to(dtype=predicted.dtype)
            per_pixel = F.smooth_l1_loss(
                predicted_chroma,
                target_chroma,
                beta=self.chroma_supervision_beta,
                reduction="none",
            ).mean(dim=1)
            chroma_sample_weights = predicted.new_ones(predicted.shape[0])
            highlight_sample_weights = predicted.new_ones(predicted.shape[0])
            if lighting_domains is not None:
                chroma_sample_weights = predicted.new_tensor(
                    [
                        self.chroma_supervision_multiplier_by_lighting_domain.get(
                            str(domain).lower(), 1.0
                        )
                        for domain in lighting_domains
                    ]
                )
                highlight_sample_weights = predicted.new_tensor(
                    [
                        self.chroma_supervision_highlight_multiplier_by_lighting_domain.get(
                            str(domain).lower(), 1.0
                        )
                        for domain in lighting_domains
                    ]
                )
            highlight = (
                target_luma >= self.chroma_supervision_highlight_luma_threshold
            ).to(dtype=predicted.dtype)
            pixel_weights = 1.0 + (
                highlight_sample_weights[:, None, None] - 1.0
            ) * highlight
            weighted_valid = valid * pixel_weights
            per_image_chroma_loss = (per_pixel * weighted_valid).sum(
                dim=(1, 2)
            ) / weighted_valid.sum(dim=(1, 2)).clamp_min(1.0)
            chroma_supervision_loss = (
                per_image_chroma_loss * chroma_sample_weights
            ).mean()
            highlight_valid = valid * highlight
            highlight_chroma_loss = (per_pixel * highlight_valid).sum() / (
                highlight_valid.sum().clamp_min(1.0)
            )
            self.last_aux_losses.update(
                {
                    "chroma_supervision_loss": chroma_supervision_loss,
                    "chroma_supervision_highlight_loss": highlight_chroma_loss.detach(),
                    "chroma_supervision_highlight_weight_mean": (
                        highlight_sample_weights.mean().detach()
                    ),
                    "chroma_supervision_sample_weight_mean": (
                        chroma_sample_weights.mean().detach()
                    ),
                    "chroma_supervision_valid_ratio": valid.mean().detach(),
                    "chroma_target_mean": target.mean().detach(),
                }
            )
        if self.chroma_prior_enabled:
            if self.chroma_prior_mode == "wb_anchor":
                anchor = self.model.wb_anchor_ev.to(dtype=output.theta.wb_ev.dtype)[None]
                normalized_residual = (output.theta.wb_ev - anchor) / self.wb_anchor_loss_scale_ev
                chroma_loss = torch.nn.functional.smooth_l1_loss(
                    normalized_residual,
                    torch.zeros_like(normalized_residual),
                    beta=0.25,
                )
                self.last_aux_losses.update(
                    {
                        "chroma_prior_loss": chroma_loss,
                        "isp_wb_r_ev": output.theta.wb_ev[:, 0].mean().detach(),
                        "isp_wb_g_ev": output.theta.wb_ev[:, 1].mean().detach(),
                        "isp_wb_b_ev": output.theta.wb_ev[:, 2].mean().detach(),
                    }
                )
            else:
                chroma_rgb = self.model.chroma_objective_rgb(output, clamp_mode="ste")
                raw_reference = output.denoised_raw.detach()
                descriptor = self._chroma_descriptor(chroma_rgb, raw_reference)
                lower = descriptor.new_tensor(self.chroma_prior_lower)
                upper = descriptor.new_tensor(self.chroma_prior_upper)
                width = (upper - lower).clamp_min(1.0e-4)
                lower = lower - self.chroma_prior_margin * width
                upper = upper + self.chroma_prior_margin * width
                violation = torch.relu(lower - descriptor) + torch.relu(descriptor - upper)
                chroma_loss = torch.nn.functional.smooth_l1_loss(
                    violation / width,
                    torch.zeros_like(violation),
                    beta=0.25,
                )
                self.last_aux_losses.update(
                    {
                        "chroma_prior_loss": chroma_loss,
                        "isp_log_rg_mean": descriptor[:, 0].mean().detach(),
                        "isp_log_bg_mean": descriptor[:, 1].mean().detach(),
                    }
                )
        return output.rgb


class _RAWAdapterFrontend(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        from baselines.raw_adapter import RAWAdapter

        if RAWAdapter is None:
            raise ImportError("baselines.raw_adapter.RAWAdapter is unavailable; check torch installation.")
        model_adapter_dim = _cfg_get(cfg, "model_adapter_dim", 24)
        self.model = RAWAdapter(
            mode=str(_cfg_get(cfg, "mode", "normal")),
            lut_dim=int(_cfg_get(cfg, "lut_dim", 32)),
            k_size=int(_cfg_get(cfg, "k_size", 3)),
            w_lut=bool(_cfg_get(cfg, "w_lut", True)),
            model_adapter_dim=None if model_adapter_dim is None else int(model_adapter_dim),
        )

    def forward(self, raw: torch.Tensor, *, input_format: str = "rgb_raw", **_: Any) -> torch.Tensor:
        return self.model(raw, input_format=input_format).image


class _RAMFrontend(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        from baselines.ram import RawAdaptationModule

        if RawAdaptationModule is None:
            raise ImportError("baselines.ram.RawAdaptationModule is unavailable; check torch installation.")
        functions = _cfg_get(cfg, "functions", ("wb", "ccm", "gamma", "brightness"))
        if isinstance(functions, str):
            functions = tuple(item.strip() for item in functions.split(",") if item.strip())
        self.model = RawAdaptationModule(
            in_channels=int(_cfg_get(cfg, "in_channels", 3)),
            img_size=int(_cfg_get(cfg, "img_size", 256)),
            out_channels=int(_cfg_get(cfg, "out_channels", 128)),
            functions=tuple(functions),
            clamp_values=bool(_cfg_get(cfg, "clamp_values", False)),
            output_channels=int(_cfg_get(cfg, "output_channels", 3)),
        )

    def forward(self, raw: torch.Tensor, *, input_format: str = "rgb_raw", **_: Any) -> torch.Tensor:
        return self.model(raw, input_format=input_format)


class _RAWildFrontend(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        from baselines.rawild import RAWildAdapter

        if RAWildAdapter is None:
            raise ImportError("baselines.rawild.RAWildAdapter is unavailable; check torch installation.")
        self.raw_bit_depth = float(_cfg_get(cfg, "raw_bit_depth", 10))
        self.model = RAWildAdapter(
            grid_depth=int(_cfg_get(cfg, "grid_depth", 8)),
            transformer_dim=int(_cfg_get(cfg, "transformer_dim", 128)),
            num_heads=int(_cfg_get(cfg, "num_heads", 4)),
            num_layers=int(_cfg_get(cfg, "num_layers", 2)),
            curve_n=int(_cfg_get(cfg, "curve_n", 6)),
            use_bezier=bool(_cfg_get(cfg, "use_bezier", True)),
            use_grid=bool(_cfg_get(cfg, "use_grid", True)),
            cond_injection_type=str(_cfg_get(cfg, "cond_injection_type", "adaln")),
        )

    def forward(self, raw: torch.Tensor, *, input_format: str = "rgb_raw", **_: Any) -> torch.Tensor:
        return self.model(raw, input_format=input_format, raw_bit_depth=self.raw_bit_depth)


def build_raw_frontend(cfg: Any) -> nn.Module | None:
    name = str(_cfg_get(cfg, "name", "none") or "none").lower()
    if name in {"", "none"}:
        return None
    if name in {"identity", "rgb"}:
        frontend: nn.Module = IdentityRAWFrontend()
    elif name == "darkisp":
        frontend = _DarkISPFrontend(cfg)
    elif name in {"rawvla", "raw-vla"}:
        frontend = _RAWVLAFrontend(cfg)
    elif name in {"raw_adapter", "raw-adapter", "rawadapter"}:
        frontend = _RAWAdapterFrontend(cfg)
    elif name in {"ram", "raw_adaptation_module", "raw-adaptation-module"}:
        frontend = _RAMFrontend(cfg)
    elif name in {"rawild", "rawild_adapter"}:
        frontend = _RAWildFrontend(cfg)
    else:
        raise ValueError(f"Unknown framework.raw_frontend.name={name!r}")

    checkpoint = _cfg_get(cfg, "checkpoint", None)
    if checkpoint:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # The released baseline files are resumable training checkpoints. Their
        # frontend module is nested under a family-specific top-level key,
        # whereas frontend-only RAWVLA checkpoints already contain adapter keys.
        checkpoint_keys = {
            "darkisp": "darkisp",
            "ram": "ram",
            "raw_adapter": "raw_adapter",
            "raw-adapter": "raw_adapter",
            "rawadapter": "raw_adapter",
            "rawild": "rawild",
            "rawild_adapter": "rawild",
        }
        nested_key = checkpoint_keys.get(name)
        if nested_key and isinstance(state, dict) and nested_key in state:
            state = state[nested_key]
            # Registry adapters own the baseline module under ``model``.
            state = {f"model.{key}": value for key, value in state.items()}
        if not isinstance(state, dict):
            raise TypeError(f"Frontend checkpoint must contain a state dict, got {type(state).__name__}")
        strict_load = bool(_cfg_get(cfg, "strict_load", True))
        if strict_load and name in {"rawvla", "raw-vla"}:
            incompatible = frontend.load_state_dict(state, strict=False)
            allowed_missing = {"model.wb_anchor_ev"}
            if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "RAWVLA checkpoint mismatch: "
                    f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
                )
        else:
            frontend.load_state_dict(state, strict=strict_load)

    trainable = bool(_cfg_get(cfg, "trainable", True))
    frontend.requires_grad_(trainable)
    trainable_prefixes = list(_cfg_get(cfg, "trainable_parameter_prefixes", []) or [])
    if trainable and trainable_prefixes:
        matched = []
        for parameter_name, parameter in frontend.named_parameters():
            selected = any(
                parameter_name.startswith(str(prefix)) for prefix in trainable_prefixes
            )
            parameter.requires_grad_(selected)
            if selected:
                matched.append(parameter_name)
        if not matched:
            raise ValueError(
                "trainable_parameter_prefixes did not match any RAW frontend parameters"
            )
    return frontend


def get_raw_frontend_aux_losses(frontend: nn.Module | None) -> dict[str, torch.Tensor]:
    """Return differentiable auxiliary losses produced by the latest forward."""

    if frontend is None:
        return {}
    losses = getattr(frontend, "last_aux_losses", None)
    return dict(losses) if isinstance(losses, dict) else {}
