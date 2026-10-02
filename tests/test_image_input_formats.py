import numpy as np
import pytest

from deployment.model_server.tools import image_tools


def test_validate_accepts_uint8_and_float32() -> None:
    uint8_image = np.zeros((8, 9, 3), dtype=np.uint8)
    float_image = np.linspace(0.0, 1.0, 8 * 9 * 3, dtype=np.float32).reshape(8, 9, 3)

    assert image_tools.validate_image_array(uint8_image) is uint8_image
    assert image_tools.validate_image_array(float_image) is float_image


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((8, 9, 3), dtype=np.float64),
        np.full((8, 9, 3), 1.01, dtype=np.float32),
        np.full((8, 9, 3), -0.01, dtype=np.float32),
        np.full((8, 9, 3), np.nan, dtype=np.float32),
    ],
)
def test_validate_rejects_ambiguous_or_invalid_float_images(image: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        image_tools.validate_image_array(image)


def test_resize_preserves_float32_without_uint8_quantization() -> None:
    image = np.linspace(0.0, 1.0, 5 * 7 * 3, dtype=np.float32).reshape(5, 7, 3)
    resized = image_tools.resize_image(image, 11, 13)

    assert resized.shape == (11, 13, 3)
    assert resized.dtype == np.float32
    assert 0.0 <= float(resized.min()) <= float(resized.max()) <= 1.0
    assert np.any((resized * 255.0) % 1.0 != 0.0)


def test_resize_with_pad_preserves_both_supported_dtypes() -> None:
    for dtype in (np.uint8, np.float32):
        value = 127 if dtype == np.uint8 else 0.5
        image = np.full((5, 9, 3), value, dtype=dtype)
        resized = image_tools.resize_with_pad(image, 12, 12)

        assert resized.shape == (12, 12, 3)
        assert resized.dtype == dtype


def test_pil_boundary_maps_equivalent_uint8_and_float32_inputs_equally() -> None:
    uint8_image = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
    float_image = uint8_image.astype(np.float32) / 255.0

    uint8_pil = image_tools.to_pil_preserve(uint8_image)
    float_pil = image_tools.to_pil_preserve(float_image)

    np.testing.assert_array_equal(np.asarray(uint8_pil), np.asarray(float_pil))
