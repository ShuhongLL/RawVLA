# Fixed default ISP calibration

The calibration fixes two deterministic camera profiles:

- `agentview`: fitted from `input/head_camera.png`;
- `wrist`: jointly fitted from `input/left_wrist.png` and `input/right_wrist.png`, and used unchanged for both wrists.

This directory is self-contained:

```text
calibration/
├── input/                       # the three untouched calibration captures
├── calibrate_default_isp.py     # ColorChecker fitting
├── default_isp.py               # runtime ISP and image CLI
├── default_isp.json             # fixed runtime parameters
├── calibration_report.json
└── preview/
    └── *_raw_vs_default_isp.png # calibration comparison images
```

The pipeline is intentionally a conventional, non-learned ISP:

```text
three-channel linear RAW
  -> black-level normalization (currently zero)
  -> scalar exposure
  -> per-channel white balance
  -> gray-preserving 3x3 color correction matrix
  -> standard linear-sRGB to sRGB transfer
  -> [0, 1] clipping / uint8 quantization
```

`default_isp.json` contains the runtime parameters. The CCM rows sum to one,
so a neutral value remains neutral after white balance. The head-camera chart
is rotated 180 degrees and is sampled in that orientation. The two wrist
charts are fitted together, not averaged after two independent fits.

Reproduce the fit and preview images with:

```bash
python finetune/calibration/calibrate_default_isp.py
```

Apply the fixed pipeline to an image with:

```bash
python finetune/calibration/default_isp.py \
  --input finetune/calibration/input/left_wrist.png \
  --view left_wrist \
  --output /tmp/left_wrist_default_isp.png
```

For decoded dataset frames, call `apply_default_isp(frame, camera_name)`. Camera
aliases such as `head_camera`, `left_wrist_camera`, and `right_wrist_camera`
are accepted. The calibration assumes the stored RGB triplets are linear RAW
code values, as used by this dataset; do not apply it to already gamma-encoded
RGB images.
