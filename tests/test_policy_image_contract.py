import numpy as np
import pytest
import torch

from starVLA.model.modules.raw_frontend.image_contract import (
    as_rgb_chw_float01,
    assert_rgb_float01,
)


def test_uint8_hwc_is_canonicalized_to_float01():
    image = np.full((8, 12, 3), 255, dtype=np.uint8)
    output = as_rgb_chw_float01(image)
    assert output.dtype == torch.float32
    assert output.shape == (3, 8, 12)
    torch.testing.assert_close(output, torch.ones_like(output))


def test_float_255_is_rejected_instead_of_silently_rescaled():
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        as_rgb_chw_float01(torch.full((3, 8, 12), 255.0, dtype=torch.float32))


def test_contract_preserves_gradient():
    image = torch.rand(2, 3, 8, 12, dtype=torch.float32, requires_grad=True)
    output = assert_rgb_float01(image)
    output.mean().backward()
    assert image.grad is not None
