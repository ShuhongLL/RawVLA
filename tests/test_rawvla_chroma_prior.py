from __future__ import annotations

import math

import torch
from omegaconf import OmegaConf

from starVLA.model.modules.raw_frontend.registry import _RAWVLAFrontend


def test_extreme_low_exposure_multiplier_scales_only_selected_samples() -> None:
    frontend = _RAWVLAFrontend(
        OmegaConf.create(
            {
                "spatial_width": 4,
                "hist_bins": 8,
                "state_dim": 16,
                "luma_state_dim": 8,
                "fft_patch_size": 8,
                "fft_stride": 4,
                "exposure_target": 0.789,
                "exposure_loss_type": "two_sided_l1",
                "extreme_low_exposure_multiplier": 4.0,
            }
        )
    )
    raw = torch.full((2, 1, 1, 3, 12, 16), 0.1)

    frontend(raw, lighting_domains=["Normal", "Normal"])
    normal_loss = frontend.last_aux_losses["exposure_prior_loss"]
    torch.testing.assert_close(
        frontend.last_aux_losses["exposure_sample_weight_mean"],
        torch.tensor(1.0),
    )

    frontend(raw, lighting_domains=["ExtremeLow", "ExtremeLow"])
    extreme_low_loss = frontend.last_aux_losses["exposure_prior_loss"]
    torch.testing.assert_close(extreme_low_loss, 4.0 * normal_loss)
    torch.testing.assert_close(
        frontend.last_aux_losses["exposure_sample_weight_mean"],
        torch.tensor(4.0),
    )


def test_lighting_domain_exposure_target_overrides_only_selected_samples() -> None:
    frontend = _RAWVLAFrontend(
        OmegaConf.create(
            {
                "spatial_width": 4,
                "hist_bins": 8,
                "state_dim": 16,
                "luma_state_dim": 8,
                "fft_patch_size": 8,
                "fft_stride": 4,
                "exposure_target": 0.789,
                "exposure_target_by_lighting_domain": {
                    "Low": 0.72,
                    "ExtremeOver": 0.74,
                },
                "exposure_loss_type": "two_sided_l1",
            }
        )
    )
    raw = torch.full((2, 1, 1, 3, 12, 16), 0.1)

    frontend(raw, lighting_domains=["Low", "Normal"])
    output_mean = frontend.last_aux_losses["isp_output_mean"]
    expected = 0.5 * ((output_mean - 0.72).abs() + (output_mean - 0.789).abs())
    torch.testing.assert_close(frontend.last_aux_losses["exposure_prior_loss"], expected)
    torch.testing.assert_close(
        frontend.last_aux_losses["exposure_sample_target_mean"],
        torch.tensor((0.72 + 0.789) / 2.0),
    )

    frontend(raw, lighting_domains=["ExtremeOver", "ExtremeOver"])
    output_mean = frontend.last_aux_losses["isp_output_mean"]
    torch.testing.assert_close(
        frontend.last_aux_losses["exposure_prior_loss"],
        (output_mean - 0.74).abs(),
    )


def test_gray_edge_fixed_debias_preserves_global_cast() -> None:
    frontend = _RAWVLAFrontend(
        OmegaConf.create(
            {
                "spatial_width": 4,
                "hist_bins": 8,
                "state_dim": 16,
                "luma_state_dim": 8,
                "fft_patch_size": 8,
                "fft_stride": 4,
                "chroma_prior_descriptor": "gray_edge",
                "chroma_prior_bias": [0.0148334559, -0.0044445377],
            }
        )
    )
    torch.manual_seed(17)
    rgb = (0.05 + 0.20 * torch.rand(2, 3, 12, 16)).requires_grad_()
    raw_reference = torch.full_like(rgb, 0.25)
    clean = frontend._chroma_descriptor(rgb, raw_reference)

    green_gain = rgb.new_tensor([0.65, 1.0, 0.65])[None, :, None, None]
    green = frontend._chroma_descriptor(rgb * green_gain, raw_reference)
    torch.testing.assert_close(
        green - clean,
        torch.full_like(clean, math.log(0.65)),
        atol=2.0e-6,
        rtol=2.0e-6,
    )

    blue_gain = rgb.new_tensor([0.75, 0.75, 1.50])[None, :, None, None]
    blue = frontend._chroma_descriptor(rgb * blue_gain, raw_reference)
    expected_blue = clean.new_tensor([0.0, math.log(2.0)])[None].expand_as(clean)
    torch.testing.assert_close(blue - clean, expected_blue, atol=2.0e-6, rtol=2.0e-6)

    green.square().mean().backward()
    assert rgb.grad is not None
    assert torch.isfinite(rgb.grad).all()
    assert rgb.grad.abs().sum() > 0


def test_v9_wb_anchor_and_gray_preserving_ccm() -> None:
    anchor = [0.27840286, -0.51591979, 0.23751693]
    frontend = _RAWVLAFrontend(
        OmegaConf.create(
            {
                "spatial_width": 4,
                "hist_bins": 8,
                "state_dim": 16,
                "luma_state_dim": 8,
                "fft_patch_size": 8,
                "fft_stride": 4,
                "wb_anchor_ev": anchor,
                "wb_residual_scale_ev": 0.5,
                "gray_preserving_ccm": True,
                "chroma_prior_mode": "wb_anchor",
                "wb_anchor_loss_scale_ev": 0.5,
            }
        )
    )
    model = frontend.model
    theta0 = model.initial_theta(2, device=torch.device("cpu"), dtype=torch.float32)
    torch.testing.assert_close(theta0.wb_ev, torch.tensor(anchor)[None].expand(2, -1))

    chroma_raw = torch.randn(2, 8, requires_grad=True)
    wb_ev, ccm = model._decode_chroma(chroma_raw)
    torch.testing.assert_close(
        ccm.sum(dim=2), torch.ones(2, 3), atol=2.0e-7, rtol=0.0
    )
    torch.testing.assert_close(
        ccm @ torch.ones(2, 3, 1), torch.ones(2, 3, 1), atol=2.0e-7, rtol=0.0
    )
    loss = torch.nn.functional.smooth_l1_loss(
        (wb_ev - model.wb_anchor_ev[None]) / frontend.wb_anchor_loss_scale_ev,
        torch.zeros_like(wb_ev),
        beta=0.25,
    )
    loss.backward()
    assert chroma_raw.grad is not None
    assert chroma_raw.grad[:, :2].abs().sum() > 0
    # The WB anchor loss must not directly optimize CCM coordinates.
    torch.testing.assert_close(chroma_raw.grad[:, 2:], torch.zeros_like(chroma_raw.grad[:, 2:]))
