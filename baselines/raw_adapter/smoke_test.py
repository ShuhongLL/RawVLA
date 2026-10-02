"""RAW-Adapter smoke tests."""

from __future__ import annotations


def test_import_and_forward() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("torch is not installed; skipped RAW-Adapter forward smoke test")
        return

    from .raw_adapter import RAWAdapter

    torch.manual_seed(0)
    model = RAWAdapter(mode="low", w_lut=True, model_adapter_dim=8)
    x = torch.rand(2, 3, 64, 64)
    out = model(x, input_format="rgb_raw")
    assert out.image.shape == (2, 3, 64, 64)
    assert len(out.stages) == 5
    assert out.adapter is not None
    assert out.adapter.shape[1] == 8


if __name__ == "__main__":
    test_import_and_forward()
    print("RAW-Adapter smoke tests passed")
