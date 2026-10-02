"""RAWild baseline modules."""

try:
    from .rawild import (
        BilateralGridAdapterDIA,
        BilateralGridAdapter_DIA,
        ConditionalTransformerBlock,
        RAWildAdapter,
        RAWildOutput,
        RGBuvHistogram,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise

    BilateralGridAdapterDIA = None
    BilateralGridAdapter_DIA = None
    ConditionalTransformerBlock = None
    RAWildAdapter = None
    RAWildOutput = None
    RGBuvHistogram = None

__all__ = [
    "BilateralGridAdapterDIA",
    "BilateralGridAdapter_DIA",
    "ConditionalTransformerBlock",
    "RAWildAdapter",
    "RAWildOutput",
    "RGBuvHistogram",
]
