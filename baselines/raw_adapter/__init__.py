"""RAW-Adapter baseline modules."""

try:
    from .raw_adapter import (
        GainDenoise,
        InputLevelAdapter,
        Input_level_Adapeter,
        KernelPredictor,
        MatrixPredictor,
        MergeBlock,
        Merge_block,
        ModelLevelAdapter,
        Model_level_Adapeter,
        NILUT,
        RAWAdapter,
        RAWAdapterOutput,
        gaussian_blur,
        shades_of_gray,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise

    GainDenoise = None
    InputLevelAdapter = None
    Input_level_Adapeter = None
    KernelPredictor = None
    MatrixPredictor = None
    MergeBlock = None
    Merge_block = None
    ModelLevelAdapter = None
    Model_level_Adapeter = None
    NILUT = None
    RAWAdapter = None
    RAWAdapterOutput = None
    gaussian_blur = None
    shades_of_gray = None

__all__ = [
    "GainDenoise",
    "InputLevelAdapter",
    "Input_level_Adapeter",
    "KernelPredictor",
    "MatrixPredictor",
    "MergeBlock",
    "Merge_block",
    "ModelLevelAdapter",
    "Model_level_Adapeter",
    "NILUT",
    "RAWAdapter",
    "RAWAdapterOutput",
    "gaussian_blur",
    "shades_of_gray",
]
