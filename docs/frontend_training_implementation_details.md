# Frontend Training Implementation Details for LIBERO

Last updated: 2026-08-26

> Status: repository-grounded implementation record for paper writing.
> Scope: DarkISP, RAM, RAW-Adapter, RAWild, and RAWVLA.
> Important distinction: the RAWVLA trainer, dataset sampler, and final V6/V7
> configurations are available locally. The four baseline checkpoints contain
> training metadata and optimizer states, but the standalone scripts that
> produced them are not present in this checkout. Statements below are therefore
> labelled as **verified**, **checkpoint-recorded**, or **not recoverable**.

## 1. Experimental setting shared by all frontends

The frontends were adapted to pretrained LIBERO policies rather than trained as
standalone RGB reconstruction networks. A separate frontend checkpoint was
maintained for each downstream policy family:

1. Qwen3-VL-PI;
2. Qwen3-VL-OFT;
3. WM4A-Cosmos-GR00T;
4. WM4A-Wan-OFT;
5. PI0;
6. PI0.5.

The input is a three-channel linear pseudo-RAW image in RGB channel order. For
methods originally defined on packed Bayer data, the repository uses an
explicit adapter. DarkISP maps RGB RAW to `[R,G,B,G]`; RAM reduces packed Bayer
to `[R,(G_r+G_b)/2,B]`; RAWild uses the same pseudo-RAW as both its guide and
apply image. Thus, comparisons use the same camera signal while retaining each
method's frontend parameterization.

The downstream policy is initialized from its pretrained LIBERO checkpoint.
Only frontend parameters are intended to be optimized; the policy provides the
task loss and the gradient with respect to its image input. In RAWVLA training,
the frozen modules are additionally forced into evaluation mode to disable
dropout while preserving the differentiable image-to-policy path.

## 2. Four non-RAWVLA baselines

### 2.1 Training protocol recoverable from the checkpoints

All 24 baseline checkpoints are resumable dictionaries containing the frontend
state, Adam-family optimizer state, constant scheduler state, step counter, and a
metadata dictionary. The following settings are directly recoverable from
those files.

| Setting | DarkISP | RAM | RAW-Adapter | RAWild |
|---|---:|---:|---:|---:|
| Optimizer steps | 8,000 | 8,000 | 8,000, except PI0.5 at 4,000 | 8,000 |
| Optimizer state | Adam-family | Adam-family | Adam-family | Adam-family |
| Betas | (0.9, 0.999) | (0.9, 0.999) | (0.9, 0.999) | (0.9, 0.999) |
| Weight decay | 0 | 0 | 0 | 0 |
| LR | `1e-4` | `1e-4` | `2e-5` | `1e-4` |
| Scheduler | constant | constant | constant | constant |
| Recorded window size | 8 | 8 | 8 | 8 |
| Recorded action horizon | 8 | 8 | 8 | 8 |
| World size | 2 | 2 | 2 | 2 |

With zero weight decay, the serialized parameter-group fields do not reliably
distinguish `torch.optim.Adam` from `torch.optim.AdamW`; the original baseline
trainer is needed before naming one of them in the paper.

The policy-dependent per-rank micro-batches are also recorded:

| Policy route | Per-rank micro-batch | Two-rank micro-batch total | Recorded loss family |
|---|---:|---:|---|
| Qwen3-OFT | 8 | 16 | action L1 only |
| Qwen3-PI | 8 | 16 | metadata says `qwen3_oft_action_l1_only` |
| WM4A-Cosmos | 2 | 4 | action-only flow matching |
| WM4A-Wan | 2 | 4 | metadata says `wm4a_cosmos_flow_matching_action_only` |
| PI0 | 1 | 2 | frontend flow matching |
| PI0.5 | 1 | 2 | frontend flow matching |

The metadata labels for Qwen3-PI and WM4A-Wan are internally inconsistent with
their policy names and are most likely stale labels copied from neighboring
trainers. They must not be used to claim that Qwen3-PI was actually optimized
with an OFT head or that Wan used the Cosmos action head. The checkpoint does
verify that the correct pretrained policy checkpoint was selected, but it does
not contain the missing forward-pass source code.

### 2.2 What “window size 8” and “gradient averaging” do *not* yet prove

The baseline checkpoint metadata records `window_size=8`, but the local
standalone baseline trainer is absent. Consequently, the current evidence does
**not** establish all of the following:

- whether the eight samples were strictly consecutive camera frames;
- whether they were sampled with stride one or at the policy replanning cadence;
- whether eight frame losses were averaged before backpropagation, accumulated
  as eight micro-steps, or jointly evaluated as a batch;
- whether DistributedDataParallel used its default cross-rank mean or a custom
  gradient reduction.

For a paper, the safe statement is: “we optimized each baseline for 8k steps
using an eight-sample training window and two workers/ranks,” followed by the
loss and micro-batch information above. Writing “we average gradients over
eight consecutive frames” requires recovering the original trainer or logs.

### 2.3 Method-specific trainable transformations

#### DarkISP

DarkISP learns a dynamic linear mapping from Bayer-like RAW to RGB, followed by
an eight-basis nonlinear tone stretch. Its implementation also exposes the
paper's Self-Boost matrix-consistency regularizer. However, the released LIBERO
checkpoint metadata says **action loss only**. Therefore Self-Boost should not
be claimed as part of these LIBERO runs unless the original trainer is found.

#### RAM

RAM predicts several ISP transformations in parallel—white balance, color
correction, gamma, and brightness—and fuses the resulting branches through the
official reverse-hourglass feature-fusion module. The LIBERO version receives
three-channel RAW-RGB, not four-channel RGGB, matching the preprocessing used by
the official RAM training configuration.

#### RAW-Adapter

RAW-Adapter follows the staged physics-inspired pipeline: learned exposure
gain, Gaussian denoising and sharpening, Shades-of-Gray white balance, a learned
camera color matrix, and an optional implicit 3D LUT. It is the only baseline
whose checkpoints use a smaller learning rate (`2e-5`). The PI0.5 checkpoint
stopped at 4k rather than 8k steps; this exception should be disclosed in any
training-budget comparison.

#### RAWild

RAWild combines a physics-guided monotonic Bezier curve with a conditioned
bilateral-grid adapter. Its original six-channel interface separates a low-bit
guide from a high-bit apply image; in LIBERO, both are constructed from the same
three-channel pseudo-RAW input. Its relatively larger frontend was trained with
the same `1e-4` constant learning rate as DarkISP and RAM.

## 3. RAWVLA training strategy

### 3.1 Frozen-policy, task-driven frontend adaptation

RAWVLA is trained independently for each of the six pretrained policies. The
vision/world-model backbone and action head are frozen and placed in evaluation
mode, but autograd remains enabled with respect to the processed image. Hence,
the action objective updates all differentiable RAWVLA operators without
updating policy parameters. Only frontend weights are saved.

The selected V6/V7 models were trained for 2,000 optimizer steps, with
checkpoints at 1,000 and 2,000 steps. Each run used one GPU, per-device batch
size 1, and eight gradient-accumulation micro-steps, giving an effective batch
of eight sampled trajectory windows per optimizer update. This optimizer-level
accumulation is distinct from both temporal recurrence and diffusion-repeat
averaging.

### 3.2 Two-level causal temporal sampling

RAWVLA deliberately separates camera-rate denoising from policy-rate recurrent
adaptation. Let `c` be the randomly sampled current environment step, `s` the
policy-specific observation stride, `T=6` the recurrent length, and `K=6` the
burst length. The recurrent observation endpoints are

```text
o_t = c - s (T - 1 - t),                 t = 0,...,T-1,
```

and the inner burst for endpoint `o_t` is

```text
b_{t,k} = o_t - (K - 1 - k),             k = 0,...,K-1.
```

This constructs a causal `T x K = 6 x 6` index grid, not 36 uniformly spaced
policy frames. However, the selected implementation does not use all 36 values
symmetrically. At every recurrent call, the operating-point estimator reads
only the burst endpoint `b_{t,K-1}=o_t`. The FFT renderer also processes that
call's full burst, but the wrapper discards intermediate RGB renderings and
retains only the final recurrent call's RGB. Consequently, the loss-relevant
temporal inputs are:

```text
operating-point state:  o_0, ..., o_5       (six policy-stride endpoints)
final FFT denoising:     b_5,0, ..., b_5,5  (six adjacent camera frames)
```

Earlier bursts are loaded and evaluated, but their non-endpoint frames do not
affect the retained state or final loss. This implementation detail should be
reported rather than claiming 36-frame fusion. At episode boundaries, indices
are clipped, which repeats the first or last valid frame rather than crossing
into another episode.

| Frozen policy | Recurrent stride `s` | Action horizon | Action normalization |
|---|---:|---:|---|
| Qwen3-PI | 8 | 8 | StarVLA min-max |
| Qwen3-OFT | 8 | 8 | StarVLA min-max |
| WM4A-Cosmos-GR00T | 8 | 8 | StarVLA min-max |
| WM4A-Wan-OFT | 8 | 8 | StarVLA min-max |
| PI0 | 5 | 50 | PI0 z-score |
| PI0.5 | 5 | 10 | PI0.5 quantile/min-max |

The dataset exposes a virtual length of 100,000 and visits each episode for 32
dataset indices before moving to the next episode slot. For every index it
draws a deterministic pseudo-random current step using `seed + index * 104729`.
This produces diverse causal windows while keeping runs reproducible.

### 3.3 OFT, DiT, and flow-matching gradients

The policy-specific task losses have different reductions and stochasticity:

- **Qwen3-OFT and WM4A-Wan-OFT:** predict an eight-step action chunk and use
  `nn.L1Loss`, whose default reduction averages over batch, horizon, and action
  dimensions. There is one deterministic OFT forward loss per sampled window.
- **Qwen3-PI:** uses a DiT-based action model. Each image/action window is
  repeated eight times with independently sampled diffusion/flow conditions;
  the action head returns a scalar mean loss. These eight repeats are Monte
  Carlo samples of the action objective, not eight video frames.
- **WM4A-Cosmos-GR00T:** similarly repeats the conditioning and action target
  eight times for its DiT/flow-matching action objective and returns a mean
  squared error.
- **PI0 and PI0.5:** use flow matching and minimize the mean squared error
  between the target velocity and predicted velocity, averaged across all
  batch, horizon, and action coordinates.

This distinction matters for the paper: “gradient averaging” occurs at three
different levels—mean reduction inside the action loss, mean reduction across
eight stochastic diffusion samples for the DiT routes, and optimizer gradient
accumulation over eight dataset micro-batches. None of these should be
described as averaging the six recurrent observations; the six endpoint
observations are processed causally to produce the final ISP state, and the
final six-frame burst is rendered for the policy.

### 3.4 Backbone-specific action-loss scaling

The unscaled action-to-frontend gradient magnitude varied by more than two
orders of magnitude across the six frozen action heads. A shared coefficient
therefore caused some frontends to remain near identity and others to collapse
to saturated or color-shifted solutions. V6/V7 fixed the RAWVLA architecture
and swept only the action multiplier. The selected objective is

```text
L = lambda_action * L_action
  + 0.01 * L_brightness
  + 0.001 * L_chroma.
```

| Frozen policy | Selected `lambda_action` |
|---|---:|
| Qwen3-PI | 0.0015 |
| Qwen3-OFT | 0.20 |
| WM4A-Cosmos-GR00T | 0.01 |
| WM4A-Wan-OFT | 0.02 |
| PI0 | 0.15 |
| PI0.5 | 0.15 |

These coefficients balance the action gradient against frontend-specific
regularizers; they do not change the frozen policy's own parameters. During
training, the implementation logs raw and weighted action-to-ISP gradient
norms, brightness/chroma gradient norms for the appropriate parameter groups,
and gradient cosine similarities. These measurements are diagnostics only and
do not modify the backward gradient.

### 3.5 Brightness and chroma objectives

The luminance/exposure branch is regularized with a two-sided L1 penalty on the
mean processed intensity, targeting `0.48`. The auxiliary rendering detaches
white balance and CCM, so brightness cannot be corrected by tinting the image.
It uses a straight-through clamp: the forward image remains in `[0,1]`, while
the backward pass retains a gradient outside the hard-clamp range. The actual
image passed to the frozen VLA and the deployment output still use a normal hard
clamp.

The chroma branch uses a weak, unpaired envelope prior on
`mean(log(R/G))` and `mean(log(B/G))`. A Smooth-L1 penalty is active only when
the output falls outside the q01--q99 envelope computed from unpaired
default-lighting LIBERO RGB, expanded by a 10% margin. This loss updates only
the chroma recurrent state, relative white balance, and CCM; it is exactly zero
inside the envelope. No paired clean RGB reconstruction, Grey-World loss,
tone-identity loss, or per-frame RGB target is used for backpropagation.

### 3.6 Differentiability, clipping, and temporal truncation

Several operations called "clipping" or "truncation" occur in the code, but
they have different gradient behavior. They should not be conflated in the
paper.

#### Straight-through clamp for auxiliary recovery gradients

The brightness and chroma auxiliary renderings use the following
straight-through estimator (STE) when `clamp_mode="ste"`:

```python
clipped = value.clamp(0.0, 1.0)
value_ste = value + (clipped - value).detach()
```

Consequently, the forward value is exactly the physical hard clamp,

```text
value_ste = clip(value, 0, 1),
```

while autograd sees the identity derivative,

```text
d value_ste / d value = 1.
```

This is the mechanism that lets the brightness objective pull an initially
overexposed or underexposed rendering back from saturation. Without the
`detach`, an ordinary clamp has zero derivative for values below zero or above
one, so a fully saturated auxiliary image can stop providing a useful exposure
gradient.

The STE is deliberately restricted to the two auxiliary render paths:

- The exposure rendering detaches `wb_ev` and `ccm_matrix`, but leaves denoise,
  scalar exposure, and shared tone connected. Its STE therefore provides
  recovery gradients only to the achromatic branch.
- The chroma rendering detaches denoise, scalar exposure, shared tone, and the
  denoised RAW reference. Its STE provides gradients only to relative WB, CCM,
  and the chroma recurrent path.
- The actual `output.rgb` consumed by the frozen VLA uses ordinary hard clamps,
  as does deployment. The action loss therefore optimizes the exact deployed
  forward operator, with zero image-gradient outside the hard-clamp interval.
  RAWVLA does **not** use an STE for the action-loss image.

The last point is important: the auxiliary loss creates an escape direction
when the deployed/action path saturates, but it does not pretend that the
deployed hard clamp has an identity derivative.

#### Complete inventory of boundary and discrete operators

| Operation | Forward behavior | Backward behavior in the selected V6/V7 training |
|---|---|---|
| Dataset RAW conversion | LIBERO cached `uint8` pseudo-RAW is converted to float by division by 255 | Offline/discrete preprocessing; no gradient is required with respect to source pixels |
| Input-range clamp | RAW burst is hard-clamped to `[0,1]` before the frontend | Hard-clamp derivative, but the sampled RAW tensor is not trainable |
| FFT denoiser output | Hard-clamped to `[0,1]` | Gradient to the denoise control is zero only where the merged result lies outside the interval |
| Exposure parameter | `e = 6 tanh(e_raw)` EV | Fully differentiable; derivative is `6(1-tanh^2(e_raw))`, approaching zero only near the ±6 EV limits |
| Exposure gain | `g = 2^e` | `dg/de = ln(2) 2^e`; no rounding or discrete gain selection |
| Denoise strength | `eta = sigmoid(eta_raw)` | Differentiable and bounded in `(0,1)` |
| Recurrent update coefficient | Selected configs replace predicted gates with `alpha=1` using `full_like` | Every candidate ISP state is adopted directly; the instantiated update-gate heads receive no loss gradient in V6/V7 |
| Relative WB | two `tanh` coordinates form a zero-sum 3-vector, followed by max-norm projection | Piecewise differentiable; `clamp_min(1)` activates only at the magnitude boundary |
| CCM | six off-diagonals are `0.25 tanh(.)`; selected V6/V7 configs use `gray_preserving_ccm=false` | Differentiable bounded parameterization; no matrix rounding or lookup |
| Tone curve | cumulative `softmax` increments define a monotonic degree-8 Bernstein curve | Differentiable with respect to tone logits and in-range pixels; exact endpoints remain fixed at 0 and 1 |
| VLA/action RGB | Hard clamp before and after the tone curve, plus a redundant hard clamp in the image adapter | Exact deployed forward; zero derivative outside `[0,1]` |
| Exposure auxiliary RGB | Same forward hard clamp implemented with STE | Identity surrogate derivative through both clamp sites |
| Chroma auxiliary RGB | Same forward hard clamp implemented with STE | Identity surrogate derivative, with non-chroma variables explicitly detached |
| Histogram binning | `floor -> long -> scatter_add` into 64 bins | Non-differentiable with respect to bin membership; descriptors are computed from fixed RAW observations, while all following encoder/GRU/head weights remain trainable |
| Quantiles | `torch.quantile` on RAW luminance | Used as a fixed observation descriptor; not a route for gradients to any camera/input parameter |
| Chroma valid-pixel mask | Thresholds detached RAW support (`mean>0.01`, `max<0.995`) | Mask selection is non-differentiable and carries no gradient; accepted chroma pixels remain differentiable through log-ratio statistics |
| Chroma envelope | ReLU distance outside the expanded q01--q99 interval, followed by Smooth-L1 | Exactly zero loss/gradient inside the envelope; piecewise differentiable outside it |

The hard histogram is therefore not an obstacle to training the frontend. It
acts as an observed feature vector: gradients update the histogram encoder,
recurrent modules, and ISP heads downstream of that vector. The method does not
attempt to learn the RAW sensor values or histogram bin boundaries.

#### Full six-step BPTT during training

The temporal slice and recurrent-state handling are also easy to misdescribe.
The wrapper retains only the most recent configured context,

```python
raw = raw[:, -rnn_frames:, -burst_frames:]
```

then iterates over all six recurrent observations. During V6/V7 training it
passes `output.state` and `output.theta` directly into the next observation
without calling `detach()`. Thus the final action and auxiliary losses perform
full backpropagation through all `T=6` operating-point updates. This is a finite
unroll, not truncated BPTT inside the six-step window. Only the final call's
six-frame FFT rendering is connected to the retained RGB loss; earlier calls
contribute through recurrent state/theta, whose update reads their endpoint
frames.

The public `state.detach()` and `theta.detach()` helpers are intended for
streaming inference or an explicitly chosen TBPTT boundary between separate
windows. They are not invoked by the V6/V7 training wrapper. If used between
windows, they stop gradients into earlier windows while keeping all frontend
parameters trainable in the current window; detaching a state tensor does not
freeze the GRU or ISP heads.

Episode-boundary clipping is a data-index operation rather than an autograd
operation. Negative history indices and action indices beyond the episode are
clipped to the first or last valid index, which repeats the boundary frame or
action. Included repeated frames still participate normally in the current
computation graph; there is simply no cross-episode temporal context.

#### Parameter freezing does not block the image gradient

Frozen VLA parameters have `requires_grad=False` and their modules are kept in
`eval()` mode, but their forward pass is not wrapped in `torch.no_grad()` and
the processed image is not detached. Autograd therefore computes the
vector-Jacobian product through the frozen policy to `output.rgb` and then to
RAWVLA. Conceptually, for frozen policy parameters `phi` and trainable frontend
parameters `theta`, the update uses

```text
dL/dtheta = (dL/d PolicyImage) (d PolicyImage/d FrontendRGB)
            (d FrontendRGB/dtheta),
```

while `dL/dphi` is neither stored nor applied. The `detach()` calls used for
metrics, checkpoint serialization, and the branch-isolated auxiliary renders
do not detach the action-loss image.

#### Gradient-norm clipping is different from image clipping

After `accelerator.backward(total_loss)`, the trainer calls global
`clip_grad_norm_` with maximum norm 1.0. This rescales parameter gradients when
their joint L2 norm exceeds one; it does not clamp individual gradient entries
and is unrelated to image-value clipping or the STE. In the checked-in trainer,
this call is inside the accumulation context and has no explicit
`sync_gradients` guard, so it is invoked on each of the eight accumulation
micro-steps. The wrapped optimizer and learning-rate scheduler update only on
the synchronized eighth micro-step.

### 3.7 Recurrent and ISP-specific choices

The final frontend uses two 64-dimensional recurrent states: one for equal
luminance and one for scale-invariant chroma. Luminance is `(R+G+B)/3`, and the
luminance spatial encoder receives only this scalar luminance replicated over
channels, preventing chroma information from bypassing the split descriptors.
The recurrent update coefficient is fixed to one, so each observation directly
adopts its predicted ISP update. The predicted update-gate modules remain
instantiated but are overwritten by a constant-one tensor and therefore are not
trained in V6/V7. Exposure is parameterized as
`2 ** (6 * tanh(e))`, corresponding to a learnable range of `[-6,+6] EV`.
The tone curve is shared across RGB and remains monotonic. Local color and tone
residual branches are disabled.

For the final recurrent observation, RAWVLA performs current-frame-anchored FFT
fusion over its six-frame camera burst using overlapping `32 x 32` patches with
stride 16. The current frame is the reference; historical frames contribute
only when their patchwise frequency evidence is sufficiently consistent. This
final burst is used for noise suppression, whereas the recurrent state tracks
the slower ISP operating point from six endpoint frames. Although the wrapper
also computes FFT renderings for the first five recurrent calls, those RGB
outputs are discarded and do not contribute to the final training objective.

For Cosmos and Wan world-model routes, the VAE posterior is fixed to its mode
rather than sampled. This removes an additional source of stochastic image
gradient variance. Qwen3-PI uses eight repeated diffusion samples per window.

### 3.8 Optimizer and numerical details

The final 2k runs use AdamW with frontend learning rate `1e-4`, betas
`(0.9, 0.95)`, epsilon `1e-8`, and a cosine schedule with 100 warmup steps and
minimum learning rate `1e-6`. Gradients are clipped to global norm 1.0. Forward
passes use bfloat16 autocast where supported, while action computations that
require it are promoted to float32. Frozen backbones remain in `eval()` mode.
The scheduler advances only on synchronized optimizer updates, not on the seven
intermediate gradient-accumulation micro-steps.

The trainer invokes norm clipping after every backward micro-step inside the
accumulation context; it does not place an explicit `sync_gradients` guard
around clipping. Optimizer stepping and scheduler stepping remain synchronized
update events. This exact timing should be preserved if the implementation is
reproduced verbatim.

Weight decay is `1e-8` for the Qwen/WM4A configurations and `1e-10` for the
PI0/PI0.5 configurations. Only frontend state dictionaries are written at
steps 1k and 2k.

### 3.9 Checkpoint selection and use of paired RGB

Paired default-lighting RGB is never read by the training forward pass. A
no-gradient checkpoint hook renders five fixed lighting domains for the agent
and wrist cameras. The 1k/2k candidates and action-loss multipliers were chosen
by first rejecting black/white collapse and then minimizing mean paired-RGB MAE
over these ten diagnostic images. This constitutes model selection, not
training supervision, and should be disclosed separately from the loss.

### 3.10 End-to-end training pseudocode

The following pseudocode summarizes the checked-in V6/V7 computation rather
than proposing a new algorithm:

```python
# One accumulation micro-step; policy parameters phi are frozen.
raw, actions = sample_causal_window()       # [B, T=6, K=6, 3, H, W]
raw = clamp(raw[:, -6:, -6:], 0, 1)

state, theta = None, None
for t in range(6):
    # advance_operating_point reads only raw[:, t, -1]
    out = rawvla(raw[:, t], state=state, theta_prev=theta)
    state, theta = out.state, out.theta      # no detach: full 6-step BPTT

# Only the final call's out.rgb is retained. Its FFT render used raw[:, 5, :].
action_loss = frozen_policy_loss(out.rgb, actions)  # no torch.no_grad()

# Both auxiliary forward values equal a hard-clamped physical rendering.
brightness_rgb = exposure_render(out, clamp_mode="ste")
chroma_rgb = chroma_render(out, clamp_mode="ste")
brightness_loss = mean(abs(mean_per_image(brightness_rgb) - 0.48))
chroma_loss = smooth_l1(outside_envelope(chroma_descriptor(chroma_rgb)))

loss = lambda_action * action_loss + 0.01 * brightness_loss + 0.001 * chroma_loss
accelerator.backward(loss)                  # loss is accumulation-scaled
accelerator.clip_grad_norm_(parameters, 1.0)
optimizer.step()                            # wrapper is a no-op until micro-step 8
if accelerator.sync_gradients:
    scheduler.step()
optimizer.zero_grad()                       # wrapper preserves intermediate accumulation
```

Here `parameters` is passed as `self.model.parameters()` in the trainer, but
frozen policy tensors have no gradients; the effective norm is therefore over
the trainable frontend gradients. The actual policy-specific action-loss
implementation replaces the abstract `frozen_policy_loss` line.

## 4. Paper-ready English draft

The following paragraph is safe for RAWVLA and can be shortened to fit the
paper. The bracketed baseline sentence should be used only with the caveat in
Section 2.2.

> We adapted a separate RAW frontend to each pretrained LIBERO policy while
> freezing both the visual/world-model backbone and action head. Frozen modules
> were kept in evaluation mode, but gradients were retained with respect to the
> processed image, allowing the task objective to optimize only the frontend.
> RAWVLA used a two-scale causal temporal context: the recurrent ISP estimator
> processed six endpoint frames sampled at the policy replanning interval
> (eight environment steps for the Qwen3 and WM4A policies and five for
> PI0/PI0.5), while the final endpoint was rendered from a burst of six
> consecutive camera frames by the FFT denoiser. Qwen3-OFT and Wan-OFT supplied a
> mean-reduced action L1 objective; the DiT/flow routes used mean-reduced
> flow-matching losses, with eight stochastic repeats for Qwen3-PI and
> Cosmos-GR00T. Because input-gradient magnitudes differed substantially across
> frozen heads, we used policy-specific action coefficients of 0.0015, 0.20,
> 0.01, 0.02, 0.15, and 0.15 for Qwen3-PI, Qwen3-OFT, Cosmos-GR00T, Wan-OFT,
> PI0, and PI0.5, respectively. The task loss was combined with a two-sided
> brightness loss (weight 0.01, target mean 0.48) and a weak unpaired chroma
> envelope loss (weight 0.001). We optimized the frontend for 2k steps with
> AdamW (learning rate `1e-4`, eight-step gradient accumulation, gradient norm
> 1.0) and a cosine schedule following 100 warmup steps. Auxiliary exposure and
> chroma renderings used hard-clamped forward values with identity
> straight-through clamp derivatives, while the action-loss and deployed RGB
> paths retained ordinary hard clamps. We backpropagated through the complete
> six-update recurrent unroll without detaching the hidden or ISP state. Paired
> default-lighting RGB was used only in a no-gradient checkpoint-selection hook
> and never as reconstruction supervision.

> [For the four comparison frontends, checkpoint records indicate 8k optimizer
> steps, an eight-sample window, two distributed ranks, action-only objectives,
> and constant learning rates of `1e-4` for DarkISP, RAM, and RAWild and `2e-5`
> for RAW-Adapter; the RAW-Adapter/PI0.5 run terminated at 4k steps.]

## 5. Evidence map

- Baseline interfaces and adaptations:
  [`baselines/darkisp/README.md`](../baselines/darkisp/README.md),
  [`baselines/ram/README.md`](../baselines/ram/README.md),
  [`baselines/raw_adapter/README.md`](../baselines/raw_adapter/README.md), and
  [`baselines/rawild/README.md`](../baselines/rawild/README.md).
- Baseline checkpoint rendering and metric export:
  [`scripts/render_all_baseline_checkpoints_xmlfix.py`](../scripts/render_all_baseline_checkpoints_xmlfix.py).
- Frontend registry and trainability:
  [`starVLA/starVLA/model/modules/raw_frontend/registry.py`](../starVLA/starVLA/model/modules/raw_frontend/registry.py).
- Core recurrent ISP, FFT merge, tone curve, clamps, and state objects:
  [`baselines/rawvla/rawvla.py`](../baselines/rawvla/rawvla.py).
- Differentiable tensor handoff to the frozen Qwen vision path:
  [`starVLA/starVLA/model/modules/raw_frontend/qwen_tensor_vision.py`](../starVLA/starVLA/model/modules/raw_frontend/qwen_tensor_vision.py).
- Causal two-axis dataset sampling:
  [`starVLA/starVLA/dataloader/rawvla_npz_datasets.py`](../starVLA/starVLA/dataloader/rawvla_npz_datasets.py).
- Six-backbone configuration generation:
  [`scripts/rawvla_1k_matrix_common.py`](../scripts/rawvla_1k_matrix_common.py).
- Objective composition, gradient diagnostics, accumulation, and scheduler
  handling:
  [`starVLA/starVLA/training/train_starvla.py`](../starVLA/starVLA/training/train_starvla.py).
- OFT L1, Qwen3-PI repeated DiT loss, Cosmos repeated flow loss, and PI flow
  loss:
  [`QwenOFT.py`](../starVLA/starVLA/model/framework/VLM4A/QwenOFT.py),
  [`QwenPI.py`](../starVLA/starVLA/model/framework/VLM4A/QwenPI.py),
  [`CosmoPredict2GR00T.py`](../starVLA/starVLA/model/framework/WM4A/CosmoPredict2GR00T.py),
  and [`PI0.py`](../starVLA/starVLA/model/framework/VLM4A/PI0.py).
- Final selected RAWVLA hyperparameters:
  [`baselines/rawvla/optimal_backbone_configs.yaml`](../baselines/rawvla/optimal_backbone_configs.yaml)
  and [`ours_test/v6`](../ours_test/v6/README.md) / [`v7`](../ours_test/v7/README.md).

## 6. Items to verify before camera-ready submission

1. Recover the standalone trainer used to create the four baseline checkpoint
   families. It is needed to resolve the exact meaning of `window_size=8` and
   whether temporal/frame losses were averaged.
2. Correct or explain the stale loss labels in the Qwen3-PI and WM4A-Wan
   baseline checkpoint metadata.
3. Decide whether checkpoint selection by paired-RGB MAE belongs in the main
   implementation section or the model-selection/validation paragraph.
4. Report actual hardware and wall-clock time only from the job logs; the
   repository configurations establish GPU allocation but not a portable speed
   measurement.
5. If the camera-ready implementation changes gradient clipping to run only on
   synchronized accumulation steps, report that as a change from the checked-in
   V6/V7 trainer rather than silently describing the new behavior as the
   original one.
