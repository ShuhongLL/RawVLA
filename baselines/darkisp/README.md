# Dark-ISP Baseline

This folder implements the core Dark-ISP modules from the ICCV 2025 paper
"Dark-ISP: Enhancing RAW Image Processing for Low-Light Object Detection".

The public arXiv source currently contains the paper, supplementary material,
and figures, but no PyTorch source code. CatalyzeX also lists the code action as
"Request Code". This implementation therefore follows the equations and
architectural description in the paper:

- Dynamic linear mapping: trainable `3 x 4` camera matrix plus local and global
  context residuals.
- Nonlinear stretch: pixel-wise coefficients over eight polynomial tone bases,
  with the skip form described in the supplementary material.
- Self-Boost regularization: least-squares pseudo matrix from the nonlinear
  output, aligned to the learned linear matrix with cosine distance.

The main model accepts Bayer-packed RAW as `[B, 4, H, W]`. For RAW-VLA/LIBERO
experiments, `input_format="rgb_raw"` adapts a three-channel linear pseudo-RAW
tensor to Bayer-packed form by using `[R, G, B, G]`.

```python
from baselines.darkisp import DarkISP, self_boost_loss

isp = DarkISP()
out = isp(raw, input_format="packed_bayer")
rgb = out.rgb
loss_sb = self_boost_loss(raw, out.rgb, out.linear_matrix)
```

When the official repository becomes available, the most likely replacement
points are the local/global attention blocks and the exact polynomial basis
expressions. Both are isolated in `darkisp.py`.
