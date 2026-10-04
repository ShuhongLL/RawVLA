# Additional LIBERO training recipes

These configurations are optional and are not the three LIBERO backbones used
for the paper's main RawVLA-Bench comparison. They inherit the same
[`rawvla_libero.yaml`](../../base/rawvla_libero.yaml) base as the main recipes:

- [`qwen3_pi.yaml`](qwen3_pi.yaml): Qwen3-PI. The paper also uses this backbone
  in its ISP perturbation analysis.
- [`wm4a_cosmos.yaml`](wm4a_cosmos.yaml): WM4A-Cosmos experimental recipe.
- [`wm4a_wan.yaml`](wm4a_wan.yaml): WM4A-Wan experimental recipe.

After completing the root README's environment, dataset, and checkpoint setup,
select a recipe explicitly. From the repository root, for example:

```bash
export RAWVLA_ROOT="$PWD"
export LIBERO_CONFIG_PATH="$RAWVLA_ROOT/.local/libero"
conda activate starvla-libero
cd starVLA
accelerate launch --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml ../configs/libero/optional/qwen3_pi.yaml \
  --datasets.vla_data.cache_root "$RAWVLA_ROOT/benchmark_data/RawVLA-Bench/data/libero/raw"
```
