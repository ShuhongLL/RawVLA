"""RAM smoke tests."""

from __future__ import annotations


def test_import_and_forward() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("torch is not installed; skipped RAM forward smoke test")
        return

    from .ram import RawAdaptationModule

    torch.manual_seed(0)
    ram = RawAdaptationModule(out_channels=32, ffm_params={"ffm_type": "BN_HG", "mid_channels": 16})
    x = torch.rand(2, 3, 64, 80)
    out = ram(x, return_details=True)
    assert out.image.shape == (2, 3, 64, 80)
    assert set(out.branches) == {"wb", "ccm", "gamma", "brightness"}

    rggb = torch.rand(2, 4, 64, 80)
    out2 = ram(rggb, input_format="packed_bayer")
    assert out2.shape == (2, 3, 64, 80)


if __name__ == "__main__":
    test_import_and_forward()
    print("RAM smoke tests passed")
