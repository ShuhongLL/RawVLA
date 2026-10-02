"""Dark-ISP baseline modules."""

from .bayer import numpy_mosaic_to_packed4, numpy_packed4_to_mosaic

try:
    from .darkisp import (
        DarkISP,
        DarkISPOutput,
        DynamicLinearMapping,
        PolynomialToneMapping,
        make_camera_matrix,
        self_boost_loss,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise

    DarkISP = None
    DarkISPOutput = None
    DynamicLinearMapping = None
    PolynomialToneMapping = None
    make_camera_matrix = None
    self_boost_loss = None

__all__ = [
    "DarkISP",
    "DarkISPOutput",
    "DynamicLinearMapping",
    "PolynomialToneMapping",
    "make_camera_matrix",
    "numpy_mosaic_to_packed4",
    "numpy_packed4_to_mosaic",
    "self_boost_loss",
]
