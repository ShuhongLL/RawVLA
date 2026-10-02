"""Small RAW-VLA shape, identity, streaming, and gradient tests.

Run with:

    python -m baselines.rawvla.smoke_test
"""

from __future__ import annotations


def test_rawvla_forward() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("torch is not installed; skipped RAW-VLA forward smoke test")
        return

    from .rawvla import RAWVLA, summarize_theta

    torch.manual_seed(0)
    model = RAWVLA(
        spatial_width=8,
        hist_bins=8,
        state_dim=32,
        fft_patch_size=16,
        fft_stride=8,
        # Keep this test on the legacy joint path; split/default routing is
        # covered independently below.
        split_luma_chroma_condition=False,
        fixed_update_alpha=None,
    )
    burst = torch.rand(2, 3, 3, 24, 32)
    out = model(burst)

    assert out.rgb.shape == (2, 3, 24, 32)
    assert out.denoised_raw.shape == (2, 3, 24, 32)
    assert out.state.shape == (2, 32)
    assert out.update_gate.shape == (2, 3)
    assert out.spatial_attention.shape == (2, 1, 6, 8)
    assert summarize_theta(out.theta).shape == (2, 31)
    torch.testing.assert_close(
        out.theta_candidate.ccm_matrix.diagonal(dim1=1, dim2=2),
        torch.ones(2, 3),
    )
    torch.testing.assert_close(out.theta_candidate.wb_ev.sum(dim=1), torch.zeros(2), atol=1.0e-6, rtol=0.0)
    assert out.theta_candidate.exposure_ev.shape == (2, 1)
    assert torch.isfinite(out.rgb).all()
    assert torch.isfinite(out.state).all()
    assert ((0.0 <= out.rgb) & (out.rgb <= 1.0)).all()

    next_burst = torch.rand_like(burst)
    next_out = model(next_burst, state=out.state, theta_prev=out.theta)
    assert next_out.rgb.shape == out.rgb.shape
    assert not torch.equal(next_out.state, out.state)

    # A first-frame image-task loss must reach the shared trunk and every
    # candidate/gate MLP; exact-zero final weights used to block this path.
    loss = (out.rgb * torch.randn_like(out.rgb)).mean()
    loss.backward()
    first_conv = model.spatial_encoder.layers[0].block[0]
    assert first_conv.weight.grad is not None
    assert torch.isfinite(first_conv.weight.grad).all()
    assert first_conv.weight.grad.abs().sum() > 0
    assert model.state_gru.weight_ih.grad is not None
    assert model.state_gru.weight_ih.grad.abs().sum() > 0
    assert model.color_head[0].weight.grad is not None
    assert model.color_head[0].weight.grad.abs().sum() > 0
    assert model.tone_head[0].weight.grad is not None
    assert model.tone_head[0].weight.grad.abs().sum() > 0
    assert model.denoise_head[0].weight.grad is not None
    assert model.denoise_head[0].weight.grad.abs().sum() > 0
    assert model.update_gate[0].weight.grad is not None
    assert model.update_gate[0].weight.grad.abs().sum() > 0


def test_fft_identity_fallback() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    from .rawvla import FFTConservativeMerge

    torch.manual_seed(1)
    merge = FFTConservativeMerge(patch_size=16, stride=8)
    burst = torch.rand(1, 4, 3, 23, 29)
    eta = torch.zeros(1, 1)
    output = merge(burst, eta)
    torch.testing.assert_close(output, burst[:, -1], atol=2.0e-5, rtol=2.0e-5)

    single = merge(burst[:, -1:], torch.ones(1, 1))
    torch.testing.assert_close(single, burst[:, -1])


def test_histogram_contract() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    from .rawvla import RAWHistogram

    histogram = RAWHistogram(bins=8)
    descriptor = histogram(torch.rand(2, 3, 12, 16))
    assert descriptor.shape == (2, 7 * 8 + 7)
    assert torch.isfinite(descriptor).all()


def test_split_luma_chroma_condition() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    from .rawvla import RAWChromaStatistics, RAWLuminanceStatistics, RAWVLA

    torch.manual_seed(7)
    raw = 0.05 + 0.15 * torch.rand(2, 3, 16, 20)
    chroma = RAWChromaStatistics(bins=8)
    luma = RAWLuminanceStatistics(bins=8)
    # A common exposure multiplier leaves chroma descriptors unchanged while
    # the absolute-luminance descriptor changes.
    torch.testing.assert_close(chroma(raw), chroma(2.0 * raw), atol=2.0e-6, rtol=2.0e-6)
    assert not torch.equal(luma(raw), luma(2.0 * raw))

    model = RAWVLA(
        spatial_width=4,
        hist_bins=8,
        state_dim=16,
        fft_patch_size=8,
        fft_stride=4,
        split_luma_chroma_condition=True,
        luma_state_dim=12,
    )
    burst = torch.rand(2, 3, 3, 16, 20)
    output = model(burst)
    assert output.state.shape == (2, 16)
    assert model.luma_state_dim == 12
    assert model.chroma_state_dim == 4
    # Theta decoders consume only [fused current feature, updated hidden].
    # Descriptor embeddings already enter the fused feature and must not be
    # concatenated into the decoder a second time.
    assert model.denoise_head[0].in_features == 128 + model.luma_state_dim
    assert model.exposure_head[0].in_features == 128 + model.luma_state_dim
    assert model.tone_head[0].in_features == 128 + model.luma_state_dim
    assert model.chroma_head[0].in_features == 128 + model.chroma_state_dim
    assert output.update_gate.shape == (2, 4)
    # Split mode uses one achromatic tone curve shared by all RGB channels.
    torch.testing.assert_close(output.theta_candidate.tone_logits[:, 0], output.theta_candidate.tone_logits[:, 1])
    torch.testing.assert_close(output.theta_candidate.tone_logits[:, 1], output.theta_candidate.tone_logits[:, 2])
    output.rgb.mean().backward()
    assert model.luma_gru.weight_ih.grad is not None
    assert model.luma_gru.weight_ih.grad.abs().sum() > 0
    assert model.chroma_gru.weight_ih.grad is not None
    assert model.chroma_gru.weight_ih.grad.abs().sum() > 0

    model.zero_grad(set_to_none=True)
    exposure_output = model(burst)
    model.exposure_objective_rgb(exposure_output).mean().backward()
    assert model.luma_gru.weight_ih.grad is not None
    assert model.luma_gru.weight_ih.grad.abs().sum() > 0
    chroma_grad = model.chroma_gru.weight_ih.grad
    assert chroma_grad is None or chroma_grad.abs().sum() == 0

    # The symmetric chroma-only auxiliary path must update WB/CCM state while
    # leaving luma/exposure/tone state untouched.
    model.zero_grad(set_to_none=True)
    chroma_output = model(burst)
    chroma_rgb = model.chroma_objective_rgb(chroma_output, clamp_mode="ste")
    log_rgb = chroma_rgb.clamp_min(1.0 / 255.0).log()
    chroma_loss = (log_rgb[:, 0] - log_rgb[:, 1]).mean().square()
    chroma_loss.backward()
    assert model.chroma_gru.weight_ih.grad is not None
    assert model.chroma_gru.weight_ih.grad.abs().sum() > 0
    luma_grad = model.luma_gru.weight_ih.grad
    assert luma_grad is None or luma_grad.abs().sum() == 0

    equal_luma = RAWLuminanceStatistics(bins=8)
    rec709_luma = RAWLuminanceStatistics(bins=8, luminance_weights=(0.2126, 0.7152, 0.0722))
    torch.testing.assert_close(equal_luma.luma_weights.flatten(), torch.full((3,), 1.0 / 3.0))
    assert not torch.equal(equal_luma(raw), rec709_luma(raw))

    # A zero-sum channel redistribution preserves equal-channel luminance.
    # With luma-only spatial input it must therefore preserve the complete
    # luma state/candidate path, while the chroma state remains free to react.
    chroma_shifted = raw.clone()
    delta = torch.minimum(chroma_shifted[:, 1:2], 0.01 * torch.ones_like(chroma_shifted[:, 1:2]))
    chroma_shifted[:, 0:1] += delta
    chroma_shifted[:, 1:2] -= delta
    burst_a = raw[:, None].expand(-1, 3, -1, -1, -1).contiguous()
    burst_b = chroma_shifted[:, None].expand(-1, 3, -1, -1, -1).contiguous()
    out_a = model(burst_a)
    out_b = model(burst_b)
    torch.testing.assert_close(
        out_a.state[:, : model.luma_state_dim],
        out_b.state[:, : model.luma_state_dim],
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    assert not torch.equal(
        out_a.state[:, model.luma_state_dim :], out_b.state[:, model.luma_state_dim :]
    )


def test_monotonic_bernstein_tone() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    from .rawvla import RAWVLA, monotonic_bernstein_tone

    x = torch.linspace(0.0, 1.0, 257).view(1, 1, 1, -1).expand(2, 3, -1, -1)
    identity_logits = torch.zeros(2, 3, 7)
    identity = monotonic_bernstein_tone(x, identity_logits)
    torch.testing.assert_close(identity, x, atol=2.0e-6, rtol=2.0e-6)

    global_logits = torch.randn(2, 3, 7)
    global_curve = monotonic_bernstein_tone(x, global_logits)
    assert (global_curve[..., 1:] - global_curve[..., :-1] >= -1.0e-6).all()
    torch.testing.assert_close(global_curve[..., 0], torch.zeros_like(global_curve[..., 0]))
    torch.testing.assert_close(global_curve[..., -1], torch.ones_like(global_curve[..., -1]))

    local_logits = torch.randn(2, 3, 7, 1, 257)
    local_curve = monotonic_bernstein_tone(x, local_logits)
    assert torch.isfinite(local_curve).all()

    # Deployment keeps a hard clamp, while the auxiliary STE rendering keeps
    # a recovery gradient after an over-exposed value has saturated.
    model = RAWVLA(
        spatial_width=4,
        hist_bins=4,
        state_dim=8,
        fft_patch_size=8,
        fft_stride=4,
        split_luma_chroma_condition=False,
        fixed_update_alpha=None,
    )
    theta = model.initial_theta(1, device=x.device, dtype=x.dtype)
    saturated = torch.full((1, 3, 4, 5), 2.0, requires_grad=True)
    hard = model._apply_tone(saturated, theta, clamp_mode="hard").mean()
    hard_gradient = torch.autograd.grad(hard, saturated, retain_graph=True)[0]
    ste = model._apply_tone(saturated, theta, clamp_mode="ste").mean()
    ste_gradient = torch.autograd.grad(ste, saturated)[0]
    assert hard_gradient.abs().sum() == 0
    assert ste_gradient.abs().sum() > 0


def test_exposure_prior_loss() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    from .rawvla import exposure_prior_loss

    target = 0.48
    black = torch.zeros(1, 3, 4, 5)
    half_target = torch.full_like(black, target / 2.0)
    at_target = torch.full_like(black, target)
    above_target = torch.full_like(black, 0.75)
    torch.testing.assert_close(exposure_prior_loss(black), torch.tensor(1.0))
    torch.testing.assert_close(exposure_prior_loss(half_target), torch.tensor(0.5**4))
    torch.testing.assert_close(exposure_prior_loss(at_target), torch.tensor(0.0))
    torch.testing.assert_close(exposure_prior_loss(above_target), torch.tensor(0.0))

    batch = torch.cat((black, at_target), dim=0).requires_grad_()
    loss = exposure_prior_loss(batch)
    torch.testing.assert_close(loss, torch.tensor(0.5))
    loss.backward()
    assert batch.grad is not None
    assert (batch.grad[0] < 0).all()
    assert (batch.grad[1] == 0).all()


def test_exposure_wb_ccm_parameterization() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    from .rawvla import MAX_EXPOSURE_EV, MAX_WB_EV, RAWVLA, smooth_theta, summarize_theta

    model = RAWVLA(
        spatial_width=4,
        hist_bins=4,
        state_dim=8,
        fft_patch_size=8,
        fft_stride=4,
        split_luma_chroma_condition=False,
        fixed_update_alpha=None,
    )
    condition = torch.zeros(1, 128 + 8 + 64)
    final = model.color_head[-1]
    with torch.no_grad():
        final.weight.zero_()
        final.bias.zero_()
        final.bias[0] = 100.0
        final.bias[1] = 100.0
        final.bias[2] = -100.0
    high = model._predict_theta(condition)
    torch.testing.assert_close(high.exposure_ev, torch.full_like(high.exposure_ev, MAX_EXPOSURE_EV))
    torch.testing.assert_close(high.exposure_gain, torch.full_like(high.exposure_gain, 2.0**MAX_EXPOSURE_EV))
    torch.testing.assert_close(high.wb_ev.sum(dim=1), torch.zeros(1), atol=1.0e-6, rtol=0.0)
    assert high.wb_ev.abs().amax() <= MAX_WB_EV
    torch.testing.assert_close(summarize_theta(high)[:, 1:2], torch.ones(1, 1))

    with torch.no_grad():
        final.bias.zero_()
        final.bias[0] = -100.0
    low = model._predict_theta(condition)
    torch.testing.assert_close(low.exposure_ev, torch.full_like(low.exposure_ev, -MAX_EXPOSURE_EV))
    torch.testing.assert_close(low.exposure_gain, torch.full_like(low.exposure_gain, 2.0**-MAX_EXPOSURE_EV))
    torch.testing.assert_close(summarize_theta(low)[:, 1:2], -torch.ones(1, 1))

    # The existing color gate smooths exposure in EV space, not linear-gain space.
    neutral = model.initial_theta(1, device=condition.device, dtype=condition.dtype)
    gate = torch.tensor([[0.0, 0.5, 0.0]])
    halfway = smooth_theta(high, neutral, gate)
    torch.testing.assert_close(
        halfway.exposure_ev, torch.full_like(halfway.exposure_ev, MAX_EXPOSURE_EV / 2.0)
    )
    torch.testing.assert_close(
        halfway.exposure_gain, torch.full_like(halfway.exposure_gain, 2.0 ** (MAX_EXPOSURE_EV / 2.0))
    )


if __name__ == "__main__":
    test_rawvla_forward()
    test_fft_identity_fallback()
    test_histogram_contract()
    test_split_luma_chroma_condition()
    test_monotonic_bernstein_tone()
    test_exposure_prior_loss()
    test_exposure_wb_ccm_parameterization()
    print("RAW-VLA smoke tests passed")
