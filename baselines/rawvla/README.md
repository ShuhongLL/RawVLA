# RAW-VLA Baseline

> **Visualization status (2026-08-22):** the existing LIBERO 1k/2k checkpoints
> were kept unchanged and re-rendered on the paired
> `action_xmlfix_20260816T0839Z` RAW/RGB data. The checked-in PNGs and JSON
> metrics now use scenes with the per-demo assets restored; no retraining was
> performed for this refresh.

This directory contains the PyTorch implementation and design specification for
the streaming, illumination-adaptive RAW ISP frontend.

The implementation includes:

- a current-frame-anchored patchwise FFT burst merge;
- a six-block shared spatial CNN;
- mean, standard-deviation, and learned-attention feature pooling;
- separate absolute-luminance and scale-invariant log-chroma descriptors;
- separate luminance and chroma GRUs packed into one compact recurrent state;
- group-wise exposure/chroma/tone update gates;
- a scalar exposure gain, a fused WB/CCM color matrix, and monotonic Bernstein tone operators;
- optional current-frame local color and tone residuals (both disabled by default).

The recommended split-conditioning mode sends absolute luminance
histograms/quantiles/clipping statistics only to the denoise, exposure, and
achromatic-tone path. Scale-invariant chromaticity and log `R/G`, `B/G`
histograms alone condition relative WB and CCM. Its total recurrent state size
is unchanged but is divided between independent luminance and chroma GRUs, so
absolute brightness cannot leak into the WB/CCM predictor through recurrent
state. The legacy shared-condition architecture remains loadable for existing
checkpoints.

The FFT burst merge uses the static config value `burst_denoise_eta` (default
`0.5`); RAW-VLA does not predict a denoise coefficient. The predicted ISP operating point contains
one scalar exposure gain, eight scale-free WB/CCM color coordinates, and tone
coordinates.
The split mode shares one seven-parameter monotonic tone curve across RGB, so
the luminance path cannot create a color cast. Candidate heads are initialized
with tiny random final weights so task gradients reach both recurrent paths
from the first training step. Each candidate head decodes from the concatenation
of its 128-dimensional fused current feature and updated recurrent state. The
64-dimensional descriptor embedding is already part of the fused feature and
is not concatenated into the decoder a second time.

Self-Boost and test-time parameter training are intentionally not part of this
baseline. The frontend is optimized by the frozen downstream VLA action loss,
a two-sided luminance auxiliary, and a weak unpaired dataset-level chroma
envelope. No paired clean RGB is used as training supervision.

The StarVLA adapter supports the six LIBERO benchmark policies: Qwen3-PI,
Qwen3-OFT, WM4A Cosmos-Predict2-GR00T, WM4A Wan2.2-OFT, PI0, and PI0.5.
All routes consume the same causal six-frame RAW burst and return the same
exposure auxiliary-loss dictionary to the shared trainer.

## Usage

```python
import torch

from baselines.rawvla import RAWVLA

model = RAWVLA(
    split_luma_chroma_condition=True,
    state_dim=128,
    luma_state_dim=64,
    luminance_weights=(1.0, 1.0, 1.0),
    luma_spatial_input="luma",
    max_exposure_ev=6.0,
    burst_denoise_eta=0.5,
    fixed_update_alpha=1.0,
)
raw_burst = torch.rand(2, 6, 3, 224, 224)  # [B, K, RGB, H, W]

out = model(raw_burst)
rgb = out.rgb

# Carry both states to the next streaming step.
next_out = model(next_raw_burst, state=out.state, theta_prev=out.theta)
```

For truncated backpropagation through time or inference, detach both states at
the chosen sequence boundary:

```python
state = out.state.detach()
theta = out.theta.detach()
```

The current frame must be the final burst element. Values are expected to be
linear RGB RAW normalized to `[0, 1]`.

The current split design uses two recurrent states: 64 dimensions for
luminance/exposure/tone and 64 for scale-invariant chroma/WB/CCM. Luminance is
the equal-channel mean `(R+G+B)/3`, not Rec.709 Y. The spatial encoder feeding
the luma state also receives only that scalar luminance replicated across
three channels, so RGB chroma cannot bypass the separated descriptors.

## Training objective

The validated V1--V5 default uses a two-sided L1 loss between each
final-image mean and the LIBERO target `0.48`, so both under- and over-bright
outputs receive gradients. In split mode the auxiliary rendering detaches WB
and CCM: the brightness loss can update denoise/exposure/tone but cannot satisfy
the target by tinting an image. This auxiliary rendering uses a straight-through
clamp: its forward value is the real `[0,1]` clamped image, while its backward
gradient remains non-zero beyond the clamp. Deployment and the image sent to
the frozen VLA still use a hard clamp.

The downstream VLA action loss updates all ISP parameters. A weak chroma loss
updates only the chroma GRU, relative WB, and CCM when the image-level
`mean log(R/G), mean log(B/G)` descriptor leaves the q01--q99 envelope estimated
from unpaired default-lighting LIBERO RGB. The recommended weight is `0.001`
with a 10% envelope margin; it becomes exactly zero inside the envelope.

The total loss is:

```text
L = action_loss_weight * L_action + 0.01 * L_brightness + 0.001 * L_chroma
```

`action_loss_weight` is backbone-specific because the six frozen heads return
very different input-gradient scales. All RAWVLA structural settings remain
identical. The legacy one-sided fourth-power prior remains available only for
controlled comparisons:

```python
from baselines.rawvla import exposure_prior_loss

loss_exp = exposure_prior_loss(out.rgb, target_mean=0.48)
loss = loss_vla + 0.01 * loss_exp
```

The recommended training configuration uses
`2 ** (6 * tanh(raw_exposure))`, an identity-centered `1/64x`--`64x` range that
keeps the commonly needed `+3`--`+4 EV` region away from early tanh saturation.
White balance and CCM remain a separate scale-free fused color matrix. No
Grey-World, tone-identity, or clean-image reconstruction loss is used. Paired
default-lighting RGB is loaded only by the checkpoint visualization hook to
report five-domain ISP-to-target MAE and is never part of backpropagation.

## Current validated defaults

```text
camera-adjacent burst K=6; policy-cadence recurrent window T=6
dual recurrent state = luma 64 + chroma 64
luminance = (R+G+B)/3; luma spatial encoder sees luminance only
scalar exposure = 2 ** (6 * tanh(raw)); fixed update alpha = 1
shared achromatic seven-parameter tone curve
fused color matrix = CCM @ diag(relative_WB)
brightness objective = two-sided L1(target=.48, weight=.01), STE clamp
chroma envelope = weight=.001, margin=10%
local color/tone residuals = disabled
frozen backbone = eval mode, with gradients retained with respect to its image input
```

V6/V7 held this structure fixed and swept only the backbone-specific action
multiplier for 2k steps. Current paired-RGB-MAE selections are:

| Frozen backbone | action weight | step | five-domain paired RGB MAE |
|---|---:|---:|---:|
| Qwen3-PI | .0015 | 2000 | .191205 |
| Qwen3-OFT | .20 | 2000 | .161348 |
| WM4A-Cosmos | .01 | 2000 | .173550 |
| WM4A-Wan | .02 | 2000 | .188693 |
| PI0 | .15 | 2000 | .205859 |
| PI0.5 | .15 | 2000 | .173979 |

The machine-readable version is `optimal_backbone_configs.yaml`. These are
frontend image-similarity selections, not substitutes for LIBERO rollouts.
