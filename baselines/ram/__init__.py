"""RAM baseline modules."""

try:
    from .ram import (
        RAM,
        RAMOutput,
        RPDecoder,
        RPEncoder,
        RawAdaptationModule,
        adapt_ram_input,
        conv_block,
        define_feature_fusion,
        rggb_to_rgb,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise

    RAM = None
    RAMOutput = None
    RPDecoder = None
    RPEncoder = None
    RawAdaptationModule = None
    adapt_ram_input = None
    conv_block = None
    define_feature_fusion = None
    rggb_to_rgb = None

__all__ = [
    "RAM",
    "RAMOutput",
    "RPDecoder",
    "RPEncoder",
    "RawAdaptationModule",
    "adapt_ram_input",
    "conv_block",
    "define_feature_fusion",
    "rggb_to_rgb",
]
