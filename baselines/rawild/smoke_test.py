"""RAWild smoke tests."""

from __future__ import annotations


def test_import_and_forward() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("torch is not installed; skipped RAWild forward smoke test")
        return

    from .rawild import RAWildAdapter

    torch.manual_seed(0)
    model = RAWildAdapter(transformer_dim=32, num_heads=4, num_layers=1, hist_bins=8, hist_downsample_factor=4)
    x = torch.rand(2, 3, 64, 64)
    out = model(x, input_format="rgb_raw", raw_bit_depth=10, return_details=True)
    assert out.rgb.shape == (2, 3, 64, 64)
    assert out.grid is not None
    assert out.control_points is not None


if __name__ == "__main__":
    test_import_and_forward()
    print("RAWild smoke tests passed")
