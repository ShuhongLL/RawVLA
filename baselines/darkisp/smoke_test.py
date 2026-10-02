"""Small Dark-ISP smoke tests.

Run with:

    python -m baselines.darkisp.smoke_test
"""

from __future__ import annotations

import numpy as np

from .bayer import numpy_mosaic_to_packed4, numpy_packed4_to_mosaic


def test_numpy_bayer_roundtrip() -> None:
    mosaic = np.arange(8 * 10, dtype=np.float32).reshape(8, 10)
    packed = numpy_mosaic_to_packed4(mosaic, pattern="RGGB")
    recovered = numpy_packed4_to_mosaic(packed, pattern="RGGB")
    assert packed.shape == (4, 5, 4)
    np.testing.assert_array_equal(recovered, mosaic)


def test_torch_darkisp_forward() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("torch is not installed; skipped DarkISP forward smoke test")
        return

    from .darkisp import DarkISP, self_boost_loss

    torch.manual_seed(0)
    model = DarkISP(hidden_dim=16)
    raw = torch.rand(2, 4, 24, 32)
    out = model(raw)
    assert out.rgb.shape == (2, 3, 24, 32)
    assert out.linear_rgb.shape == (2, 3, 24, 32)
    assert out.linear_matrix.shape == (2, 3, 4, 24, 32)
    loss = self_boost_loss(out.raw_packed4, out.rgb, out.linear_matrix)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    pseudo_raw = torch.rand(2, 3, 24, 32)
    out_rgb_raw = model(pseudo_raw, input_format="rgb_raw")
    assert out_rgb_raw.raw_packed4.shape == (2, 4, 24, 32)


if __name__ == "__main__":
    test_numpy_bayer_roundtrip()
    test_torch_darkisp_forward()
    print("Dark-ISP smoke tests passed")
