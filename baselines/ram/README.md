# RAM Baseline

Core implementation of ICCV 2025 RAM:
"Beyond RGB: Adaptive Parallel Processing for RAW Object Detection".

This is a pure PyTorch extraction of the official
`SonyResearch/RawAdaptationModule` implementation, commit
`da0ad0dd607206189efeddf125152532a08dee77`, without MMDetection/MMEngine
dependencies.

## Default Input

The paper says RAW files are preprocessed by reshaping Bayer data into an RGGB
representation. In the official training configs, however, the loader applies
`RGGBtoRGB` before the backbone/RAM:

```python
img_r = img[..., 0]
img_g = (img[..., 1] + img[..., 2]) / 2
img_b = img[..., 3]
img = np.stack((img_r, img_g, img_b), axis=2)
```

The official `RawAdaptationModule` also defaults to `in_channels=3`.
Therefore this baseline's default input is three-channel RAW-RGB:
`input_format="rgb_raw"` with shape `[B, 3, H, W]`.

For a packed Bayer/RGGB tensor `[B, 4, H, W]`, use
`input_format="packed_bayer"`; it is converted to `[R, (Gr+Gb)/2, B]`, matching
the official `RGGBtoRGB` transform.

```python
from baselines.ram import RawAdaptationModule

ram = RawAdaptationModule()
out = ram(raw_rgb)  # default: [B, 3, H, W] RAW-RGB
```
