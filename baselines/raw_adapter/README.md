# RAW-Adapter Baseline

Core implementation of ECCV 2024 RAW-Adapter:
"RAW-Adapter: Adapting Pre-trained Visual Model to Camera RAW Images".

This is a pure PyTorch extraction of the official implementation in
`cuiziteng/ECCV_RAW_Adapter`, without MMDetection/MMCV registry code.

Official source files used for parity:

- `mmdet/models/backbones/RAW_Adapter/input_adapter.py`
- `mmdet/models/backbones/RAW_Adapter/block.py`
- `mmdet/models/backbones/RAW_Adapter/kernel.py`
- `mmdet/models/backbones/RAW_Adapter/model_adapter.py`
- `mmdet/models/backbones/RAW_resnet.py`

The input-level adapter follows the official ISP stages:

1. `I1 -> I2`: learned gain, Gaussian denoise, sharpening
2. `I2 -> I3`: Shades-of-Gray white balance
3. `I3 -> I4`: learned camera color matrix
4. `I4 -> I5`: optional implicit neural 3D LUT

The model-level adapter follows the official behavior: when `w_lut=True`, it
uses stages `[I1, I2, I3, I4]` to create the backbone adapter feature, while
`I5` remains the final input-level adapted image.

```python
from baselines.raw_adapter import RAWAdapter

model = RAWAdapter(mode="low", w_lut=True)
out = model(raw_rgb, input_format="rgb_raw")
rgb = out.image
adapter_feature = out.adapter
```
