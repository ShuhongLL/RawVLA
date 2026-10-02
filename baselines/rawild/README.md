# RAWild Baseline

Core implementation of RAWild:
"RAWild: Sensor-Agnostic RAW Object Detection via Physics-Guided Curve and Grid
Modeling".

This is a pure PyTorch extraction of the official `I2WM/RAWild` adapter, without
MMDetection/MMCV registry code or backbone wrappers.

Official source files used for parity:

- `mmdet/models/backbones/rawild_adapter/bilateral_grid_dia.py`
- `mmdet/models/backbones/rawild_adapter/rgbuv_histogram.py`
- `mmdet/models/backbones/RAW_resnet_DIA.py`

The official detection backbone expects a 6-channel tensor:

- channels `0:3`: low-bit guide image, usually 8-bit normalized to `[0, 1]`
- channels `3:6`: high-bit RAW/apply image normalized to `[0, 1]`

For RAW-VLA/LIBERO, `input_format="rgb_raw"` uses the same three-channel
pseudo-RAW as both guide and apply input.

```python
from baselines.rawild import RAWildAdapter

adapter = RAWildAdapter()
rgb = adapter(raw_rgb, input_format="rgb_raw", raw_bit_depth=10)
```
