# Checkpoint Utilities

- `download_starvla_libero.py` downloads selected public StarVLA LIBERO
  checkpoints and their base models.
- `validate_starvla_libero.py` validates downloaded checkpoint metadata and can
  optionally instantiate each model with `--load`.

Run both commands from any working directory; paths are resolved relative to
the repository unless `STARVLA_ROOT` is set.
