# RAWVLA Implementation Details for Reproduction

This document records the exact architectural constants needed to reproduce the
validated global, split-luma/chroma RAWVLA frontend. It complements
`baselines/rawvla/network_structure.md` by replacing suggested dimensions with
the values used by the selected V6/V7 configurations. The canonical
implementation is `baselines/rawvla/rawvla.py`; the machine-readable common
configuration is `baselines/rawvla/optimal_backbone_configs.yaml`.

## 1. Model size and validated global configuration

The validated frontend uses:

| Setting | Value |
|---|---:|
| Input | causal linear-RGB RAW burst, `[B, 6, 3, H, W]` |
| Spatial width | 32 |
| Luminance histogram bins | 64 |
| Chroma histogram bins $B_C$ | 64 |
| Total recurrent hidden size | 128 |
| Luminance/chroma hidden split | 64 / 64 |
| Local color residual | disabled |
| Local tone residual | disabled |
| Maximum exposure $E_{\max}$ | 6 EV |
| Fixed parameter-update coefficient | $\alpha=1$ |
| Trainable parameters (`requires_grad=True`) | **886,966** |

The parameter count includes the two update-gate heads (58,116 parameters).
They remain in the checkpoint for architectural compatibility, but their output
is replaced by the fixed $\alpha=1$ setting and therefore has no active loss
path in the validated configuration. Excluding those inactive heads, 828,850
parameters affect the forward output. The FFT burst merge itself has no
trainable parameters.

## 2. CNN, descriptor encoders, fusion layers, GRUs, and decoder

Every convolutional block below is a 3×3 `Conv2d`, one-group `GroupNorm`, and
`SiLU`. Every hidden layer in an MLP uses `SiLU`.

| Component | Exact dimensions |
|---|---|
| Spatial CNN | $3\to32$ (stride 1), $32\to32$ (1), $32\to64$ (2), $64\to64$ (1), $64\to128$ (2), $128\to128$ (1). The output is `[B,128,H/4,W/4]`. |
| Spatial statistics-attention pool | Concatenates the channel-wise spatial mean, standard deviation, and attention-weighted mean: $3\times128=384$ dimensions. Projection MLP: $384\to256\to128$. Attention score: 1×1 convolution $128\to1$. |
| Luminance descriptor encoder | Descriptor dimension 135; MLP $135\to128\to64$. |
| Chroma descriptor encoder | Descriptor dimension 330; MLP $330\to128\to64$. |
| Previous-luminance-parameter encoder | Nine coordinates; MLP $9\to64\to32$. |
| Previous-chroma-parameter encoder | Eight coordinates; MLP $8\to64\to32$. |
| Luminance fusion | $(128+64+32)=224\to256\to128$. |
| Chroma fusion | $(64+32)=96\to128\to128$. |
| Luminance GRU | one `GRUCell`, input size 128, hidden size 64. |
| Chroma GRU | one `GRUCell`, input size 128, hidden size 64. |
| Luminance output condition | fused feature 128 + hidden state 64 = 192 dimensions. The descriptor embedding is already encoded in the fused feature and is not concatenated again. |
| Chroma output condition | fused feature 128 + hidden state 64 = 192 dimensions. The descriptor embedding is already encoded in the fused feature and is not concatenated again. |
| Denoise head | $192\to128\to1$. |
| Exposure head | $192\to128\to1$. |
| Tone head | $192\to128\to7$. |
| Chroma/WB-CCM head | $192\to128\to8$. |
| Luminance update-gate head | $288\to128\to3$. |
| Chroma update-gate head | $160\to128\to1$. |

The spatial CNN receives equal-channel luminance
$L=(R+G+B)/3$, replicated to three channels. This prevents RGB chroma from
bypassing the separated chroma descriptor. The two GRUs are single-layer cells;
there is no stacked recurrent layer. Their outputs are concatenated into a
128-dimensional streaming state and carried to the next policy observation.

RAWVLA has **no U-Net or convolutional image decoder** in this configuration.
The network predicts structured ISP parameters, and an analytic decoder applies,
in order: FFT burst merging, scalar exposure, relative white balance, a 3×3 CCM,
and a shared monotonic Bernstein tone curve. The tone curve has seven predicted
relative logits plus one fixed-zero reference logit, producing eight monotonic
intervals. Since local color and local tone are disabled, no spatial residual
decoder is active.

## 3. Chroma histogram bin count $B_C$

The chroma branch uses **$B_C=64$ bins per histogram**. Its scale-invariant
descriptor contains five normalized histograms:

1. $r/(r+g+b)$,
2. $g/(r+g+b)$,
3. $b/(r+g+b)$,
4. $\log(R/G)$,
5. $\log(B/G)$.

The two log ratios are clipped to $[-4,4]$ and linearly mapped to $[0,1]$
before binning. The mean and standard deviation of each of the five quantities
are appended, giving

$$
D_C=5B_C+10=5\times64+10=330.
$$

The independent luminance branch uses one 64-bin linear histogram, one 64-bin
log histogram, five quantiles $(q_{.01},q_{.05},q_{.50},q_{.95},q_{.99})$, and
dark/clip ratios, giving $2\times64+7=135$ dimensions.

## 4. Maximum exposure range $E_{\max}$

The exposure head predicts one achromatic scalar:

$$
e=E_{\max}\tanh(z_e),\qquad E_{\max}=6\ \mathrm{EV},\qquad g_e=2^e.
$$

Thus $e\in(-6,6)$ EV and the limiting gain range is $(1/64,64)$. The same gain
is applied to all three channels; channel-relative correction is owned by white
balance and the CCM rather than by three independent exposure gains.

## 5. CCM residual bounds and parameterization

The eight-dimensional chroma head predicts two independent white-balance
coordinates followed by six CCM off-diagonal coordinates. For $i\ne j$,

$$
\Delta C_{ij}=0.25\tanh(z_{ij}),
$$

so each off-diagonal residual lies strictly in $(-0.25,0.25)$. In the default
parameterization, the diagonal is fixed to one:

$$
C=I+\Delta C,\qquad \Delta C_{ii}=0.
$$

The optional `gray_preserving_ccm=true` variant instead sets

$$
\Delta C_{ii}=-\sum_{j\ne i}\Delta C_{ij},
$$

which enforces $C\,[1,1,1]^\top=[1,1,1]^\top$ and gives diagonal entries in
the limiting range $(0.5,1.5)$. Neither variant adds determinant, orthogonality,
nor non-negativity projection.

The two white-balance coordinates generate a three-channel zero-sum EV vector.
Max-norm normalization bounds every residual channel to $[-1,1]$ EV before an
optional fixed WB anchor is added. The final scale-free color transform is
$C\,\mathrm{diag}(2^{\mathrm{WB}_{\mathrm{EV}}})$.

## 6. Burst-denoising patch size $N$, reliability scale $\kappa$, and weights $\omega_i$

The validated RAWVLA burst contains the five preceding camera observations and
the current reference frame. `FFTConservativeMerge` uses:

| Parameter | Value |
|---|---:|
| Burst length $K$ | 6 |
| Square patch size $N$ | **32** |
| Patch stride | 16 |
| Window | separable 2D Hann |
| Reliability scale $\kappa$ | **1.8** |
| Numerical stabilizer | $10^{-8}$ |
| Historical weights, oldest to newest | **$(0.05,0.075,0.125,0.1875,0.375)$** |
| Current-frame anchor coefficient | 1 |

For historical-frame spectrum $F_i$ and current reference $F_t$, define
$d_i=F_i-F_t$. Reliability is computed independently for each patch, channel,
and frequency:

$$
\sigma_i^2=\kappa\operatorname{mean}_{f}|d_i(f)|^2,
\qquad
r_i(f)=\frac{\sigma_i^2}{|d_i(f)|^2+\sigma_i^2+10^{-8}}.
$$

The merged spectrum is

$$
F_{\mathrm{out}}=F_t+\eta_{\mathrm{denoise}}
\sum_{i=1}^{5}\omega_i r_i(F_i-F_t),
$$

where the learned scalar $\eta_{\mathrm{denoise}}\in(0,1)$. Inverse FFT and
Hann-weighted overlap-add reconstruct the current denoised RAW frame. Reflection
padding is used when valid for the input size, otherwise replication padding is
used. The current frame is always the final burst element and no future frame is
accessed.

## 7. Reproduction pointers

- Architecture: `baselines/rawvla/rawvla.py`
- Validated common settings: `baselines/rawvla/optimal_backbone_configs.yaml`
- Higher-level network description: `baselines/rawvla/network_structure.md`
- Minimal construction and streaming-state example: `baselines/rawvla/README.md`

The six selected V6/V7 frontends use the same dimensions above. They differ in
the downstream frozen VLA backbone and the backbone-specific action-loss weight,
not in the RAWVLA architecture.
