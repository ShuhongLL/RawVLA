"""Load OmegaConf YAML files with relative ``extends`` inheritance."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from omegaconf import DictConfig, ListConfig, OmegaConf


ConfigPath = Union[str, Path]


def load_config(config_path: ConfigPath) -> DictConfig:
    """Load a YAML config and recursively merge its base configs.

    A config may declare either a single base or a list of bases. Relative paths
    are resolved from the directory containing the config that declares them.
    Bases are merged from left to right, then the child config is applied last.
    """

    return _load_config(Path(config_path).expanduser().resolve(), stack=())


def _load_config(config_path: Path, stack: tuple[Path, ...]) -> DictConfig:
    if config_path in stack:
        cycle = " -> ".join(str(path) for path in (*stack, config_path))
        raise ValueError(f"Config inheritance cycle detected: {cycle}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise TypeError(f"Top-level config must be a mapping: {config_path}")

    extends = config.pop("extends", None)
    if extends is None:
        return config
    if isinstance(extends, str):
        base_specs = [extends]
    elif isinstance(extends, (list, ListConfig)) and all(isinstance(item, str) for item in extends):
        base_specs = list(extends)
    else:
        raise TypeError(f"'extends' must be a path or list of paths: {config_path}")

    merged = OmegaConf.create()
    next_stack = (*stack, config_path)
    for base_spec in base_specs:
        base_path = Path(base_spec).expanduser()
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        merged = OmegaConf.merge(merged, _load_config(base_path.resolve(), next_stack))
    return OmegaConf.merge(merged, config)
