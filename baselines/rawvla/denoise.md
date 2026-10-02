# RAW-VLA Lightweight Denoising Notes

This note records the current candidate denoising path for an adaptive RAW ISP
before VLA inference. The concrete method below is based on the local probes in:

```text
artifacts/rawvla_design_v1_probe/frequency_burst_merge/
```

## Frame Definition

For denoising before a VLM/VLA image query, the intended input frames are adjacent
environment observation frames, not previous VLM query frames.

For a current reference observation at time `t`:

```text
3 frames: t-2, t-1, t
4 frames: t-3, t-2, t-1, t
5 frames: t-4, ..., t
6 frames: t-5, ..., t
```

The current frame `t` is always the reference and remains the anchor of the
output. Older frames are only used to suppress random noise.

## FFT Conservative Temporal Merge

`fft_cons` is a lightweight frequency-domain temporal merge. It is not a CNN and
is not a single-image spatial blur. It compares each historical frame with the
current frame in local Fourier space and only fuses frequency components that
look consistent with the current frame.

For each image:

1. Reflect-pad the input frames to avoid border artifacts.
2. Split each frame into overlapping `32x32` patches with stride `16`.
3. Multiply each patch by a Hann window.
4. Compute a 2D FFT for the current patch and each historical patch.
5. Estimate a reliability gate from the frequency difference.
6. Merge reliable historical frequency components into the current patch.
7. Inverse FFT, overlap-add patches, and crop back to the original size.

For one historical frame:

```text
F_out = F_ref + eta_denoise * base_weight * reliability * (F_alt - F_ref)
```

with:

```text
reliability = sigma2 / (abs(F_alt - F_ref)^2 + sigma2)
0 <= eta_denoise <= 1
```

Intuition:

```text
small frequency difference -> likely random noise -> fuse more
large frequency difference -> likely motion/misalignment/real structure -> fuse less
```

The current probe uses an intermediate setting between the earlier conservative
version and the stronger trial:

```text
FFT_CONS_SIGMA_SCALE = 1.8
FFT_CONS_WEIGHT_SCALE = 1.25
```

`sigma_scale` increases the reliability gate tolerance, so moderately different
frequency components are allowed to contribute more. `weight_scale` multiplies
all historical-frame merge weights. Together they make `fft_cons` stronger than
the first conservative setting, while still keeping the current frame as the
reference anchor.

For RAW-VLA v1, these values should be treated as fixed initialization choices
for the deterministic FFT merge. The learnable denoise control is a single
fusion strength `eta_denoise`, applied inside the merge to scale the historical
correction. The model should not add a second image-space blend after `fft_cons`.

For `fft_4_cons`, using frames `t-3, t-2, t-1, t`:

```text
F_out = F_t
      + eta_denoise * 0.1250 * r_{t-3} * (F_{t-3} - F_t)
      + eta_denoise * 0.1875 * r_{t-2} * (F_{t-2} - F_t)
      + eta_denoise * 0.3750 * r_{t-1} * (F_{t-1} - F_t)
```

The current `fft_cons` weights are:

```text
fft_3_cons: [0.1875, 0.3750]
fft_4_cons: [0.1250, 0.1875, 0.3750]
fft_5_cons: [0.0625, 0.1250, 0.1875, 0.3750]
fft_6_cons: [0.0500, 0.0750, 0.1250, 0.1875, 0.3750]
fft_7_cons: [0.0375, 0.0500, 0.0750, 0.1250, 0.1875, 0.3750]
fft_8_cons: [0.0250, 0.0375, 0.0500, 0.0750, 0.1250, 0.1875, 0.3750]
```

All are current-frame anchored. The listed weights are applied only to
historical frames; the current frame is the reference term.

## Difference From Weighted Averaging

Simple weighted averaging directly mixes pixels:

```text
out = sum_i w_i * frame_i
```

It reduces noise, but it also averages motion, edge displacement, gripper
movement, and object contact changes.

`fft_cons` instead gates the historical contribution per patch and per frequency:

```text
out frequency = current frequency + reliable historical correction
```

Therefore it usually denoises less aggressively than weighted averaging, but
preserves edges and reduces ghosting better.

## Difference From Spatial Wiener

The tested `spatial_wiener_weighted*` variant first performs temporal weighted
averaging and then applies single-image frequency shrinkage. It does not compare
historical frames against the current frame.

This means it cannot reliably distinguish:

```text
sensor noise
fine texture
object boundary
gripper/contact edge
```

In the current probes it looked visibly blurrier than `fft_cons`, so it is not
recommended as the main VLA input denoising path.

## Current Probe Result

The current ExtremeLow probe was generated from real adjacent LIBERO environment
observations, not synthetic shifted frames:

```text
lighting_domain = ExtremeLow
lighting_ev ~= -4.3277
source = artifacts/rawvla_design_v1_probe/extreme_low_env_obs_raw/...
```

Representative no-reference metrics at `ref_t=50`:

```text
agentview:
fft_3_cons: hf↓ 25.0%, edge 0.66, cha 12.9
fft_4_cons: hf↓ 31.1%, edge 0.59, cha 16.8
fft_5_cons: hf↓ 34.1%, edge 0.56, cha 18.5
fft_6_cons: hf↓ 37.2%, edge 0.52, cha 20.2
fft_7_cons: hf↓ 38.9%, edge 0.49, cha 21.2
fft_8_cons: hf↓ 40.2%, edge 0.48, cha 21.8

wrist:
fft_3_cons: hf↓ 24.6%, edge 0.67, cha 11.2
fft_4_cons: hf↓ 31.2%, edge 0.60, cha 14.4
fft_5_cons: hf↓ 34.3%, edge 0.56, cha 15.8
fft_6_cons: hf↓ 37.4%, edge 0.52, cha 17.2
fft_7_cons: hf↓ 39.1%, edge 0.50, cha 18.1
fft_8_cons: hf↓ 40.5%, edge 0.48, cha 18.6
```

Metric meanings:

```text
hf  = high-frequency energy reduction in flat regions; larger means stronger denoising
edge = edge strength retained relative to current frame; closer to 1 is less blurry
cha = mean absolute pixel change from current frame; smaller is more conservative
```

## Practical Recommendation

For VLA input with the current stronger `fft_cons` setting, `fft_3_cons` or
`fft_4_cons` is currently the safer initialization:

```text
fft_3_cons: safest initialization; useful denoising with less structural change
fft_4_cons: stronger initialization when low-light noise dominates
fft_5_cons: visibly stronger denoising, but edge/structure loss is more obvious
fft_6/7/8_cons: diminishing denoising gains with steadily increasing edge loss
```

The denoiser is always available through the same learned control path. The
histogram/state branch predicts `eta_denoise`; when denoising is not useful, the
model can drive `eta_denoise` toward zero, recovering the current RAW reference
without a hand-written exposure rule.

For contact-heavy or fast-motion moments, the same learned control can reduce
`eta_denoise`, which suppresses historical correction and recovers the current
frame anchor.
