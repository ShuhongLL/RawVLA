# RawVLA 核心组件消融实验复现说明

更新日期：2026-08-31

本文说明如何在另一台服务器上完成 Table 4 的 RawVLA core-component ablation。目标表包含两条独立实验线：

- LIBERO + Qwen3-OFT；
- RoboTwin 2.0 + π0.5。

每条实验线都在 `ExtremeLow / Low / Normal / Over / ExtremeOver` 五个固定光照域上评测，并报告五域宏平均 `Avg.`。

> 重要：本表是**重新训练后的结构消融**，不是在 Full RawVLA checkpoint 上推理时临时把特征置零。每个 variant 都必须从相同初始化规则独立训练，并拥有自己的 checkpoint。Full RawVLA 也必须用与各 ablation 完全相同的训练和评测预算重新跑，不能直接拿历史最优 checkpoint 与新训练的 ablation 比较。

## 1. 当前仓库状态与实施边界

RawVLA 主体实现位于：

```text
baselines/rawvla/rawvla.py
starVLA/starVLA/model/modules/raw_frontend/registry.py
```

训练和数据链路的完整背景见：

```text
docs/rawvla_training_reproduction_qwen3oft_pi0_pi05.md
docs/robotwin2_pi0_pi05_rawvla_reference_reproduction.md
```

当前代码已经实现 Full RawVLA 和旧版 `split_luma_chroma_condition=false`，但尚未为 Table 4 的所有 ablation 提供正交的配置开关。因此，在新服务器开始大规模训练前，应先在上述两个实现文件中加入本文第 5 节定义的 ablation mask。

不要直接使用下列“近似替代”：

- `burst_frames=1` 不是推荐的 `w/o burst denoising` 实现，因为它同时改变 dataset tensor contract 和数据读取量；应保持 `K=6`，只旁路 FFT merge。
- `rnn_frames=1` 不是 `w/o hidden state`，因为它改变训练样本的 policy 时间上下文；应保持 `T=6`，每一步只清零传入 GRU 的 previous hidden。
- 当前 `split_luma_chroma_condition=false` 是 legacy shared architecture，它同时改变 descriptor、tone head 和 conditioning topology。若表格要声称只消融 recurrent estimator，则不能直接把它当作 `Single shared GRU`。
- RoboTwin 的 `run_robotwin2_pi_manifest_default_isp_five_light_x8.sh` 输入是 `default_isp`，不会评测 RawVLA；Table 4 必须使用 native `raw_linear` 和对应 variant 的 frontend checkpoint。

### 1.1 只拉取 `origin/main` 时的已知缺口

截至 2026-09-01，这份工作区中用于 RoboTwin/π0.5 复现的下列内容尚未进入当前 Git index；因此另一台服务器只执行 `git pull origin main` 时看不到它们：

```text
starVLA/starVLA/dataloader/rawvla_npz_datasets.py
docs/rawvla_training_reproduction_qwen3oft_pi0_pi05.md
docs/repro_configs/rawvla/
backbone/robotwin/export_pi05_policy_norm_stats.py
```

不能以“本机文件存在”为依据直接认为远端已经对齐。正式提交前应在目标服务器检查：

```bash
git ls-files starVLA/starVLA/dataloader/rawvla_npz_datasets.py
git ls-files docs/repro_configs/rawvla/pi05_stage1_strong_rgb_chroma_1k.full.yaml
```

两条命令都必须打印路径；若为空，说明当前 checkout 没有得到复现实现/配置。推荐先在源服务器把这些文件审核后提交到专用实验分支，再在目标服务器 checkout 固定 commit。临时 `rsync` 可以用于 smoke，但正式长训必须记录同步文件的 SHA-256 和源 commit，不能只写 `origin/main`。

本工作区已核对的 paired loader SHA-256 为：

```text
9d5d3b2034e46b6c27609b7719ce53f802058735e2d01a27fb93ee257dcc1be8  starVLA/starVLA/dataloader/rawvla_npz_datasets.py
```

π0.5 Stage-1 历史 resolved config SHA-256 为：

```text
abe56365d1a1d9f1e3e75d9911d7d76322c619f97b06e7c4390338e416cdc181  docs/repro_configs/rawvla/pi05_stage1_strong_rgb_chroma_1k.full.yaml
```

这些 hash 只用于确认传输没有拿错版本。目标服务器仍须把 resolved config 中的旧绝对路径改为本机路径，并保存新的 resolved config/hash。

### 1.2 当前 paired loader 已实现的 contract

对齐版 `rawvla_npz_datasets.py` 已支持 `dataset_format: robotwin2`，并读取：

```text
RAW input:
  head_camera_raw_float32
  left_camera_raw_float32
  right_camera_raw_float32

paired clean RGB target:
  head_camera_rgb_uint8
  left_camera_rgb_uint8
  right_camera_rgb_uint8
```

它按照 RAW cache 的相对路径在 `rgb_cache_root` 中查找同名 NPZ；缺少 paired RGB 时会报错。clean RGB 只输出为 layout/chroma target，真正的 frontend 输入仍是 `raw_burst_float32`。它还已实现：

- 分离的 `T_rnn` 与 `K_burst` 时间轴；
- `observation_stride`、`action_horizon` 和 `action_offset`；
- `state_source_key: actions`；
- `normalization: zscore`；
- 仅在 `delta_action_indices` 非空时做 delta conversion；
- episode 内 causal clipping/padding；
- deterministic virtual sampler。

因此，如果目标服务器看到的 loader 只包含 `agentview_raw_uint8/wrist_raw_uint8`，它拿到的是旧版 LIBERO-only 文件，而不是这里已经验证过的 paired loader。

### 1.3 π0.5 policy normalization 不是 `pi05_quantile`

RoboTwin π0.5 的正式配置必须为：

```yaml
dataset_format: robotwin2
normalization: zscore
stats_key: norm_stats
action_stats_key: actions
state_stats_key: state
delta_action_indices: []
observation_stride: 5
action_horizon: 50
action_offset: 1
state_source_key: actions
```

`pi05_quantile` 是早期 LIBERO π0.5 路线使用的 q01/q99 min-max 方式，不适用于本表的 RoboTwin π0.5。对齐版 loader 的 `zscore` 分支从以下 JSON schema 读取 mean/std：

```json
{
  "norm_stats": {
    "actions": {"mean": [], "std": []},
    "state": {"mean": [], "std": []}
  }
}
```

该 JSON 属于 π0.5 policy checkpoint/processor 资产，不是 RawVLA frontend 的 normalization 统计，不应把大模型资产提交进 Git。规范路径为：

```text
<PI05_CKPT>/policy_norm_stats.json
```

可以从 LeRobot normalizer safetensors 导出：

```bash
python "$RAWVLA_ROOT/backbone/robotwin/export_pi05_policy_norm_stats.py" \
  --checkpoint-dir /path/to/pi05_robotwin
```

导出前先用 `safetensors.safe_open` 检查输入文件确实含有 `action.mean/action.std/observation.state.mean/observation.state.std`；不同 LeRobot 版本的 processor 文件名可能不同，不要仅凭文件名猜测。若原训练服务器已有 `policy_norm_stats.json`，优先原样复制并校验 SHA-256，以免使用不同版本 processor 重新导出后产生统计漂移。

只有以下四项同时通过，才能称为 RoboTwin/π0.5 训练链已对齐：

1. 对齐版 paired loader 已进入目标 checkout；
2. 632 个 RAW 相对路径与 632 个 RGB 相对路径完全一致；
3. π0.5 `policy_norm_stats.json` 存在、schema/dimension/数值有限性检查通过；
4. resolved YAML 明确为 `robotwin2 + zscore + delta_action_indices=[] + stride5/horizon50/offset1`。

## 2. 冻结的实验口径

### 2.1 所有 variant 必须相同的量

除被消融组件外，以下内容全部冻结：

- 数据文件、episode 排序、sampler seed 和 normalization stats；
- VLA backbone checkpoint；
- `K_burst=6`、`T_rnn=6` 和 observation stride；
- RawVLA width、histogram bins、总 recurrent state budget 128；
- optimizer、learning rate、warmup、batch size、gradient accumulation 和训练步数；
- action/exposure/chroma loss 权重；
- 五域 manifest、task、初始状态、noise seed 和 evaluator seed；
- checkpoint 选择规则；
- 同一实验线内的代码 commit 和软件环境。

建议每个 variant 至少跑 3 个训练 seed。主表可报告三个 seed 的 success-rate 均值，附录报告标准差。若算力只允许一个 seed，必须在实验设置中明确写出，并让所有 variant 使用同一个 seed。

### 2.2 输入时序契约

```text
dataset: [T_rnn=6, K_burst=6, H, W, C]
model:   [B_view, T_rnn=6, K_burst=6, C=3, H, W]
dtype:   float32
range:   [0, 1]
order:   RGB linear pseudo/native RAW
```

`T_rnn` 与 `K_burst` 是两个不同时间轴，不能合并。episode 开头只允许复制该 episode 的最早帧做 causal padding。

### 2.3 LIBERO 口径

训练使用当前推荐的 action-XML-fixed cache：

```text
benchmark_data/libero/rawvla_light_train_raw_npz_action_xmlfix_20260816T0839Z
benchmark_data/libero/rawvla_light_train_rgb_replay_npz_action_xmlfix_20260816T0839Z
```

评测使用冻结 manifest：

```text
benchmark/rawvla-bench/libero_manifest_50_init_states_10000_rollouts.json
```

它包含：

```text
4 suites × 10 tasks × 50 init states × 5 domains = 10,000 rollouts / variant
```

每个 domain 有 2,000 rollouts。五域范围为：

| Domain | EV range |
|---|---:|
| ExtremeLow | `[-5.5, -4.0]` |
| Low | `[-4.0, -2.5]` |
| Normal | `[-0.5, 0.5]` |
| Over | `[1.5, 3.0]` |
| ExtremeOver | `[6.0, 8.0]` |

### 2.4 RoboTwin 2.0 口径

训练 cache：

```text
benchmark_data/robotwin2/rawvla_light_train_statecopy_633/raw
benchmark_data/robotwin2/rawvla_light_train_statecopy_633/rgb
```

实际有效配对为 632 条。评测使用：

```text
benchmark/rawvla-bench/robotwin2_test_manifest_50_seeds_3250_rollouts.json
```

它包含：

```text
13 tasks × 50 base seeds × 5 domains = 3,250 rollouts / variant
```

每个 domain 有 650 rollouts。五域范围为：

| Domain | EV range |
|---|---:|
| ExtremeLow | `[-9.0, -8.5]` |
| Low | `[-7.0, -6.0]` |
| Normal | `[-0.5, 0.5]` |
| Over | `[1.0, 1.5]` |
| ExtremeOver | `[2.0, 2.5]` |

若先做 smoke test，可从冻结 manifest 中选择每个 `(task, domain)` 的 `base_seed_id=0,1,2`，即 `13×5×3=195` 条；smoke 结果不得填入正式 Table 4。

## 3. Table 4 variant 的精确定义

所有“移除输入”的 ablation 都采用**等维零向量 mask**。这样下游 MLP/heads 的输入维度不变，消融只删除指定信息。被 mask 模块可以跳过计算，但返回张量的 shape、dtype 和 device 必须与 Full 相同。

| ID | 表中名称 | 精确定义 |
|---|---|---|
| `full` | Full RawVLA | split luma/chroma descriptors；独立 64-d luma GRU 和 64-d chroma GRU；FFT burst merge；使用 previous theta conditioning。 |
| `no_burst` | w/o burst denoising | 保持输入 `K=6`，令 `denoised_raw = raw_burst[:, -1]`；其余参数预测和 ISP 不变。FFT merge 与 `eta_denoise` 不参与输出。 |
| `no_spatial` | w/o spatial feature `g_t` | 保留 luma descriptor，令 pooled spatial vector `global_feature=zeros([B,128])`；local color/tone 本来就关闭。 |
| `no_luma_desc` | w/o luminance descriptor `h_t^L` | 令 `luma_embedding=zeros([B,64])`；保留 luma spatial feature、previous luma theta 和 luma recurrent state。 |
| `no_chroma_desc` | w/o chroma descriptor `h_t^C` | 令 `chroma_embedding=zeros([B,64])`；保留 chroma previous theta 和 chroma recurrent path。 |
| `no_hidden` | w/o hidden state `S_{t-1}^{L/C}` | 每个 outer recurrent step 都将传入 luma/chroma GRU 的 previous hidden 置零；仍保留 GRU 对当前 observation 的变换及其 `new_state` 到 heads 的连接。保持 `T=6`。 |
| `no_theta_prev` | w/o `Theta_{t-1}` conditioning | 令 `luma_theta_embedding=zeros([B,32])`、`chroma_theta_embedding=zeros([B,32])`；Full 配置的 `fixed_update_alpha=1.0` 必须保持，因此 previous theta 也不会通过 EMA 混入当前 theta。 |
| `shared_gru` | Single shared GRU `L,C` | 保留分离的 `h^L/h^C` 和各自 heads；将 luma/chroma current conditioning 融合后送入一个 128-d shared GRU，同一 shared state 供 luma 与 chroma heads 使用。不要切换到 legacy descriptor。 |
| `luma_state_only` | Lumin. only (`S_t=S_t^L`) | 只维护一个 128-d luma GRU state。当前 chroma descriptor/theta conditioning仍送入 chroma head，但 chroma head 的 recurrent input 使用该 luma state；不维护 chroma memory。 |
| `chroma_state_only` | Chrom. only (`S_t=S_t^C`) | 只维护一个 128-d chroma GRU state。当前 luma spatial/descriptor/theta conditioning仍送入 luma heads，但 luma heads 的 recurrent input 使用该 chroma state；不维护 luma memory。 |

最后三个 recurrent variants 使用完整 128-d state budget，避免把“状态类型”与“总状态容量减半”混为一谈。它们会改变 GRU/head 参数量，必须在补充材料中同时报告 trainable parameter count。

如果论文作者希望 `luma_state_only/chroma_state_only` 表示“同时删除另一类 descriptor 和输出 head”，应另起名称；不能与上表的 recurrent-estimator ablation 混用。

## 4. 建议的统一配置接口

在 `RAWVLA.__init__` 和 StarVLA registry 中加入以下字段：

```yaml
ablation:
  disable_burst_denoise: false
  disable_spatial_feature: false
  disable_luma_descriptor: false
  disable_chroma_descriptor: false
  disable_previous_hidden: false
  disable_previous_theta: false
  recurrent_mode: split       # split | shared | luma_only | chroma_only
```

配置应做互斥校验：

- `recurrent_mode` 四选一；
- Table 4 每个 config 只能打开一个 core ablation，`full` 全部关闭；
- `split_luma_chroma_condition` 对新矩阵固定为 `true`；legacy shared mode 仅用于旧 checkpoint 兼容；
- `use_local_color=false`、`use_local_tone=false`、`fixed_update_alpha=1.0` 固定不变。

建议将 resolved ablation 写入每个 checkpoint：

```python
checkpoint_payload["rawvla_ablation"] = {
    "variant": variant_name,
    "flags": resolved_ablation_dict,
    "git_commit": git_commit,
}
```

评测加载时必须检查 checkpoint metadata 与 evaluator config 的 variant 相同；不允许 silent fallback 到 `full`。

## 5. 代码改动位置

### 5.1 构造函数与 registry

在 `baselines/rawvla/rawvla.py` 的 `RAWVLA.__init__` 保存上述 flags。在 `starVLA/starVLA/model/modules/raw_frontend/registry.py` 的 `_RAWVLAFrontend` 中用 `_cfg_get` 逐项传入。

不要把完整 `ablation` 字典直接无校验地透传；应给每个字段固定默认值并检查拼写。否则旧 YAML 中的 typo 可能被静默忽略。

### 5.2 `no_spatial/no_luma_desc/no_chroma_desc/no_theta_prev`

修改 `_advance_operating_point()`。先正常得到 tensor，再在进入 fusion/head 前 mask：

```python
if self.disable_spatial_feature:
    global_feature = torch.zeros_like(global_feature)
if self.disable_luma_descriptor:
    luma_embedding = torch.zeros_like(luma_embedding)
if self.disable_chroma_descriptor:
    chroma_embedding = torch.zeros_like(chroma_embedding)
if self.disable_previous_theta:
    luma_theta_embedding = torch.zeros_like(luma_theta_embedding)
    chroma_theta_embedding = torch.zeros_like(chroma_theta_embedding)
```

零向量必须在计算图内保持正确 dtype/device。无需给被旁路 encoder 伪造梯度；该模块无梯度是预期行为。

### 5.3 `no_hidden`

在调用 GRU 前只清零 previous hidden：

```python
luma_hidden_in = torch.zeros_like(luma_state) if self.disable_previous_hidden else luma_state
chroma_hidden_in = torch.zeros_like(chroma_state) if self.disable_previous_hidden else chroma_state
new_luma_state = self.luma_gru(luma_fused, luma_hidden_in)
new_chroma_state = self.chroma_gru(chroma_fused, chroma_hidden_in)
```

adapter 仍然遍历 6 个 outer observations。输出 `new_state` 仍进入当前 step 的 heads，但不会成为下一 step 的有效历史信息。

### 5.4 `no_burst`

在 `forward()` 中保持 control prediction 不变，只旁路 merge：

```python
if self.disable_burst_denoise:
    denoised = raw_burst[:, -1].clamp(0.0, 1.0)
else:
    denoised = self.fft_merge(raw_burst.clamp(0.0, 1.0), theta.eta_denoise)
```

不要把最后一帧复制 6 次再调用 FFT；数值虽可能接近 identity，但会产生不必要的 FFT 误差和开销。

### 5.5 三种 recurrent topology

推荐为三种非 split topology 各自建立明确模块，而不是在 forward 中临时切片：

```text
split:
  luma_gru(128 -> 64), chroma_gru(128 -> 64)

shared:
  shared_input = shared_fusion(concat(luma_fused, chroma_fused))  # -> 128
  shared_state = shared_gru(shared_input, state_128)

luma_only:
  luma_state = luma_gru_128(luma_fused, state_128)

chroma_only:
  chroma_state = chroma_gru_128(chroma_fused, state_128)
```

对 `shared/luma_only/chroma_only`，luma 和 chroma heads 都拼接同一个 128-d state。因此这些 mode 的 head input size 应在构造时按 mode 明确创建，不能通过 pad/truncate 复用 64-d Full head。所有 mode 的 current descriptors 仍按第 3 节保留。

### 5.6 必须新增的单元测试

在 `baselines/rawvla/smoke_test.py` 至少加入：

1. 10 个 variant 均能对 `[2,6,3,24,32]` 完成 forward/backward；
2. `no_burst.denoised_raw` 与最后一帧严格相等；
3. `no_hidden` 在相同当前 observation、不同传入 state 下输出相同；
4. `no_theta_prev` 在相同当前 observation、不同 `theta_prev` 下输出相同；
5. descriptor mask 对对应 encoder 的 gradient 为 `None` 或全零，对未消融 head 的 gradient非零；
6. 四个 recurrent mode 的 state shape 均为 `[B,128]`；
7. 保存再 strict-load 后输出一致；
8. config 中未知 variant/flag 立即报错。

执行：

```bash
export PYTHONPATH="$RAWVLA_ROOT:$RAWVLA_ROOT/starVLA:${PYTHONPATH:-}"
python -m baselines.rawvla.smoke_test
```

## 6. 新服务器准备

### 6.1 路径变量

```bash
export RAWVLA_ROOT=/path/to/RAW-VLA
export ASSET_ROOT=/path/to/rawvla_assets
export RESULT_ROOT=/path/to/results/rawvla_table4
export HF_HOME="$ASSET_ROOT/.cache/huggingface"

export PYTHONPATH="$RAWVLA_ROOT:$RAWVLA_ROOT/starVLA:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
```

不要在 YAML 中保留旧服务器的绝对路径。

### 6.2 环境

训练环境以 `docs/repro_configs/rawvla/environment-observed-20260831.txt` 为优先参考：Python 3.10、PyTorch `2.6.0+cu124`、torchvision `0.21.0+cu124`、Transformers `4.57.0`、Accelerate `1.5.2`、NumPy `1.26.4`、OmegaConf `2.3.0`。

```bash
conda create -n rawvla-ablation python=3.10 -y
conda activate rawvla-ablation
# 根据目标服务器 driver 安装匹配的 PyTorch/CUDA wheel。
pip install -r "$RAWVLA_ROOT/starVLA/requirements.txt"
pip install omegaconf==2.3.0 accelerate==1.5.2 transformers==4.57.0 \
  numpy==1.26.4 einops==0.8.1 timm==1.0.27 h5py==3.16.0
```

LIBERO 和 RoboTwin simulator 环境的部署细节分别见：

```text
benchmark/docs/environment_reproduction.md
docs/robotwin2_pi0_pi05_checkpoint_deployment.md
```

π0.5 policy、StarVLA training 和 RoboTwin simulator 可能需要分离环境；不要为方便而把互相冲突的 Torch/Transformers/SAPIEN 版本强装进一个环境。

### 6.3 数据校验

```bash
test -d "$ASSET_ROOT/benchmark_data/libero/rawvla_light_train_raw_npz_action_xmlfix_20260816T0839Z"
test -d "$ASSET_ROOT/benchmark_data/robotwin2/rawvla_light_train_statecopy_633/raw"
test -f "$RAWVLA_ROOT/benchmark/rawvla-bench/libero_manifest_50_init_states_10000_rollouts.json"
test -f "$RAWVLA_ROOT/benchmark/rawvla-bench/robotwin2_test_manifest_50_seeds_3250_rollouts.json"
```

必须额外审计：

- LIBERO RAW/RGB 两侧各 1771 个相对路径并严格对应；
- RoboTwin RAW/RGB 实际各 632 个相对路径并严格对应；
- 随机 NPZ 的 RAW 为 float32 `[0,1]`；
- lighting-domain 数量来自逐文件 metadata，而不是旧 manifest 顶层缓存统计；
- backbone checkpoint 和 normalization stats 成对。

## 7. 生成 10 个 variant 配置

建议新增一个配置生成器，例如：

```text
scripts/prepare_rawvla_core_ablation_matrix.py
```

variant 列表固定为：

```python
VARIANTS = [
    "full",
    "no_burst",
    "no_spatial",
    "no_luma_desc",
    "no_chroma_desc",
    "no_hidden",
    "no_theta_prev",
    "shared_gru",
    "luma_state_only",
    "chroma_state_only",
]
```

每个生成的 YAML 必须包含唯一的：

```yaml
run_id: <backbone>__<variant>__seed<seed>
run_root_dir: /new/server/path/results/rawvla_table4/<backbone>
framework:
  raw_frontend:
    name: rawvla
    source_key: raw_burst_float32
    input_format: rgb_raw
    trainable: true
    burst_frames: 6
    rnn_frames: 6
    spatial_width: 32
    hist_bins: 64
    state_dim: 128
    luma_state_dim: 64
    fft_patch_size: 32
    fft_stride: 16
    use_local_color: false
    use_local_tone: false
    split_luma_chroma_condition: true
    luminance_weights: [1.0, 1.0, 1.0]
    luma_spatial_input: luma
    fixed_update_alpha: 1.0
    ablation: <variant-specific flags>
```

生成后保存两份文件：

```text
configs/<run_id>.yaml          # 提交前配置
<output_dir>/config.full.yaml  # OmegaConf resolve 后的实际配置
```

必须让生成器检查所有 input/checkpoint/stats 路径存在，并写出 `matrix.json`，至少记录 `variant/seed/config/output_dir/git_commit`。

## 8. Qwen3-OFT / LIBERO 训练

以 `docs/repro_configs/rawvla/qwen3_oft_libero_qoft_aw0200.full.yaml` 为数值模板，但把 cache 换为 action-XML-fixed 版本。关键固定参数：

```yaml
seed: 42                         # 多 seed 时改为预先声明的列表
max_train_steps: 2000
save_interval: 1000
num_warmup_steps: 100
per_device_batch_size: 1
gradient_accumulation_steps: 8
optimizer: AdamW
betas: [0.9, 0.95]
eps: 1.0e-8
weight_decay: 1.0e-8
learning_rate: 1.0e-4
lr_scheduler_type: cosine_with_min_lr
min_lr: 1.0e-6
action_loss_weight: 0.20
exposure_target: 0.48
exposure_loss_weight: 0.01
chroma_prior_weight: 0.001
chroma_prior_margin: 0.10
max_exposure_ev: 6.0
freeze_modules: qwen_vl_interface,action_model
frozen_modules_eval: true
```

单个 run：

```bash
export CUDA_VISIBLE_DEVICES=0
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=30200

cd "$RAWVLA_ROOT/starVLA"
python starVLA/training/train_starvla.py \
  --config_yaml "$RESULT_ROOT/configs/qwen3oft__full__seed42.yaml" \
  2>&1 | tee "$RESULT_ROOT/logs/qwen3oft__full__seed42.log"
```

在 8-GPU 节点上应启动 8 个互相独立的单 GPU run，而不是一个 8-GPU DDP run。每个进程必须使用不同的 `CUDA_VISIBLE_DEVICES`、`MASTER_PORT`、`run_id` 和 `output_dir`。10 个 variant 可分两批完成。

## 9. π0.5 / RoboTwin 2.0 训练

### 9.1 推荐的严格协议

历史 π0.5 Full checkpoint 是五阶段 refinement。若主表要以它为 Full 基准，则每个 variant 都必须从随机 RawVLA 初始化完整走过同样五阶段，并且每一阶段只 warm-start 同一 variant 的上一阶段：

| Stage | 模板 | Steps | 用途 |
|---|---|---:|---|
| 1 | `pi05_stage1_strong_rgb_chroma_1k.full.yaml` | 1000 | 全 frontend 初训 |
| 2 | `pi05_stage2_warm_refine_300.full.yaml` | 300 | warm refinement |
| 3 | `pi05_stage3_highlight_low_200.full.yaml` | 200 | low/highlight balance |
| 4 | `pi05_stage4_gray_ccm_300.full.yaml` | 300 | chroma-only module refinement |
| 5 | `pi05_stage5_extremelow_100.full.yaml` | 100 | exposure-head-only refinement |

禁止让 ablation 从 Full RawVLA 的 stage checkpoint warm-start；这会让不同结构继承不同程度的 Full 信息。

Stage 4 的 `trainable_parameter_prefixes` 必须按 recurrent mode 调整：

- `split`：现有 chroma encoder/theta/fusion/GRU/head/gate prefixes；
- `shared_gru`：chroma current-path modules + shared fusion/GRU + chroma head/gate；
- `luma_state_only`：chroma current-path modules + luma-only GRU + chroma head/gate；
- `chroma_state_only`：chroma current-path modules + chroma-only GRU + chroma head/gate。

每次调整后打印实际 trainable keys 并保存。Stage 5 对所有 mode 只训练 `model.exposure_head.*`。

### 9.2 低成本预实验协议

若目的只是先验证趋势，可以只用 Stage 1 的统一 1000-step recipe 跑全部 variant，同时重新训练一个 Stage-1 Full。该结果只能称为“1k controlled ablation”，不得与五阶段 production Full 混在同一张表。

### 9.3 单阶段执行模板

```bash
export CUDA_VISIBLE_DEVICES=0
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=30300

cd "$RAWVLA_ROOT/starVLA"
python starVLA/training/train_starvla.py \
  --config_yaml "$RESULT_ROOT/configs/pi05__full__seed42__stage1.yaml" \
  2>&1 | tee "$RESULT_ROOT/logs/pi05__full__seed42__stage1.log"
```

下一 stage 的 `framework.raw_frontend.checkpoint` 必须指向同一 variant 上一 stage 的实际产物。脚本应在提交前执行 `test -f`，不能因路径错误退回随机初始化。

## 10. 训练时验收

每个 run 在 step 0、首个 optimizer step、最后一个 step 都检查：

1. 除 `raw_frontend.*` 外 backbone 参数全部 `requires_grad=False`；
2. frozen backbone 处于 eval mode，但 image-input gradient 可回传；
3. 当前 variant 被消融模块无梯度，其余相关 heads/encoders 梯度非零且有限；
4. 输入/输出均为 float32 `[0,1]`，没有 uint8/PIL 往返和二次 gamma；
5. `action_loss/exposure_loss/chroma_loss/total_loss/grad_norm` 无 NaN；
6. state shape 始终为 `[B_view,128]`；
7. 日志打印 variant 和 resolved ablation flags；
8. checkpoint 可在 CPU 上 strict-load，再在固定 sample 上重现相同输出。

建议保存：

```text
results/rawvla_table4/
  configs/
  logs/
  qwen3oft/<variant>/seed_<seed>/
  pi05/<variant>/seed_<seed>/stage_1..5/
  eval/libero/<variant>/seed_<seed>/
  eval/robotwin2/<variant>/seed_<seed>/
  summaries/
```

## 11. Checkpoint 选择

必须在看到 closed-loop test 结果之前冻结选择规则。推荐：

- Qwen3-OFT：固定使用 step 2000；
- π0.5：严格协议固定使用各 variant 的 Stage 5 final；低成本协议固定使用 Stage 1 step 1000；
- 不得针对不同 lighting domain 选择不同 checkpoint；
- 不得以 test success rate 选择 checkpoint；
- 如果训练崩溃，只能按预先定义的 retry 规则用同 seed 重跑，并记录失败原因。

每个 checkpoint 保存 SHA-256：

```bash
sha256sum /path/to/checkpoint.pt >> "$RESULT_ROOT/checkpoints.sha256"
```

## 12. LIBERO 五域闭环评测

先做 dry run，确认只加载 Qwen3-OFT 和目标 frontend：

```bash
ROOT="$RAWVLA_ROOT" \
RAW_FRONTEND=rawvla \
IMAGE_MODE=rawvla_bench \
RAWVLA_BENCH_REPRESENTATION=raw \
LIBERO_MODELS='Qwen3-VL-OFT-LIBERO-4in1' \
RAW_FRONTEND_EVAL_DRY_RUN=1 \
bash "$RAWVLA_ROOT/run_libero_zeroshot_all.sh"
```

现有 runner 的 `frontend_checkpoint_for()` 按固定目录名查找 checkpoint。对 ablation，应扩展 runner 接受显式变量，例如：

```text
RAW_FRONTEND_CHECKPOINT=/absolute/path/to/<variant>.pt
RAW_FRONTEND_CONFIG=/absolute/path/to/<variant>.full.yaml
```

并在 `frontend_overrides()` 中同时传入 checkpoint 和 ablation flags。仅替换 checkpoint、不恢复对应 recurrent topology，会 strict-load 失败或更糟地加载成错误模型。

正式运行：

```bash
export ROOT="$RAWVLA_ROOT"
export RAW_FRONTEND=rawvla
export RAWVLA_BENCH_REPRESENTATION=raw
export RAWVLA_BENCH_MANIFEST="$RAWVLA_ROOT/benchmark/rawvla-bench/libero_manifest_50_init_states_10000_rollouts.json"
export LIBERO_MODELS='Qwen3-VL-OFT-LIBERO-4in1'
export LIBERO_SUITES='libero_spatial libero_object libero_goal libero_10'
export NUM_TRIALS=50
export RAW_FRONTEND_CHECKPOINT=/absolute/path/to/qwen3oft__<variant>.pt
export RAW_FRONTEND_CONFIG=/absolute/path/to/qwen3oft__<variant>.full.yaml
export LOG_ROOT="$RESULT_ROOT/eval/libero/<variant>/seed_<seed>"
export GPU_IDS='0 1 2 3 4 5 6 7'

bash "$RAWVLA_ROOT/benchmark/rawvla-bench/scripts/run_libero_bench.sh"
```

评测 ledger 中每条 rollout 必须记录：`suite/task/init_state_index/lighting_domain/lighting_ev/noise_seed/variant/checkpoint_sha/success`。恢复中断任务时按这些联合键去重，不能只按 episode number 去重。

## 13. RoboTwin 2.0 + π0.5 五域闭环评测

### 13.1 不能直接使用的脚本

```text
scripts/run_robotwin2_pi_manifest_default_isp_five_light_x8.sh
```

该脚本的 `image_input_key=default_isp`，评的是原始 π0.5 policy，不加载 RawVLA。它只能作为 simulator、seed 和 13-task protocol 的参考。

```text
scripts/run_robotwin2_rawvla_five_light_eval.sh
```

当前版本硬编码了 Qwen3-OFT 命名和单一 `grab_roller` 任务，也不能原样用于本表。应基于两者生成一个新的 π0.5 RawVLA manifest runner，例如：

```text
scripts/run_robotwin2_pi05_rawvla_ablation_manifest_x8.sh
```

### 13.2 新 runner 的硬性要求

新 runner 必须：

- 从 `robotwin2_test_manifest_50_seeds_3250_rollouts.json` 读取 13 tasks、5 domains、50 seeds；
- simulator 上传三相机 native `raw_linear` float32 `[0,1]`，不是 `render_rgb/default_isp`；
- policy server 构造 StarVLA π0.5 backbone + variant-specific RawVLA config/checkpoint；
- 使用 head、left wrist、right wrist 三个 view；
- online history 使用 `T=6,K=6`，并明确选择严格 train/test alignment 的 outer stride 5；
- episode 开头复制第一帧 padding，不跨 episode 保存 state；
- 每个 episode reset RawVLA hidden state 和 theta；
- 使用 manifest 固定的 lighting EV、noise seed、dataset seed 和指令；
- 输出逐 episode JSONL，支持按联合键安全 resume；
- 每个 GPU 启动一个 policy server，使用独立 port、结果目录和临时目录。

核心环境变量应至少为：

```bash
export ROBOTWIN_IMAGE_OBS_KEY=raw_linear
export STARVLA_REQUIRE_FLOAT_IMAGE=1
export ROBOTWIN_LIGHTING_STRATEGY=environment_only
export ROBOTWIN_LIGHTING_SAMPLE_UNIT=paired
export ROBOTWIN_LIGHTING_BENCHMARK_SEED=20260824
export ROBOTWIN_RAW_WHITE_LEVEL=3.5
export ROBOTWIN_RAW_SENSOR_NOISE_ENABLED=true
export ROBOTWIN_RAW_SENSOR_NOISE_SEED=20260824
export ROBOTWIN_RAW_SENSOR_SHOT_NOISE=0.000025
export ROBOTWIN_RAW_SENSOR_READ_NOISE=0.0000000390625
export RAWVLA_OBSERVATION_STRIDE=5
```

RawVLA 的 RoboTwin config 固定：

```yaml
max_exposure_ev: 10.0
gray_preserving_ccm: true
output_contract: rgb_chw_float32_0_1
burst_frames: 6
rnn_frames: 6
```

RawVLA 输出进入 π0.5 adapter 前保持 CHW float32 `[0,1]`；adapter resize/pad 后再在 SigLIP/OpenPI 边界转为 `[-1,1]`。不要在中间转 uint8。

### 13.3 运行模板

```bash
export MODEL=pi05
export CODE_ROOT="$RAWVLA_ROOT"
export DATA_ROOT="$ASSET_ROOT"
export ROBOTWIN_MANIFEST="$RAWVLA_ROOT/benchmark/rawvla-bench/robotwin2_test_manifest_50_seeds_3250_rollouts.json"
export RAW_FRONTEND_CHECKPOINT=/absolute/path/to/pi05__<variant>__stage5.pt
export RAW_FRONTEND_CONFIG=/absolute/path/to/pi05__<variant>__stage5.full.yaml
export RESULT_ROOT="$RESULT_ROOT/eval/robotwin2/<variant>/seed_<seed>"
export BASE_PORT=29400

bash "$RAWVLA_ROOT/scripts/run_robotwin2_pi05_rawvla_ablation_manifest_x8.sh"
```

先用 195-rollout smoke selection 验证，再切到 3,250-rollout full selection。正式结果必须确认每个 `(task, domain)` 恰好有 50 条且无重复。

## 14. 结果汇总与 Table 4 填表

对每个 variant/backbone/domain，先累计 episode-level success：

```text
SR_domain = sum(success) / number_of_completed_rollouts
```

主表显示百分数：

```text
display = 100 * SR_domain
```

五域平均使用宏平均：

```text
Avg = mean(SR_ExtremeLow, SR_Low, SR_Normal, SR_Over, SR_ExtremeOver)
```

由于正式 protocol 中五域样本数相同，宏平均与全部 rollout micro average 数值相同；仍建议按上式显式计算并在 metadata 中写 `average=macro_over_domains`。

汇总前执行完整性检查：

| Benchmark | 每 domain | 每 variant 合计 |
|---|---:|---:|
| LIBERO | 2,000 | 10,000 |
| RoboTwin 2.0 | 650 | 3,250 |

若不足，不要把未完成数量当失败填 0；先 resume。若某些 episode 在声明的最大 retry 后仍失败，主表同时报告 completed/expected，并在附录给出 simulator failure 数量。

建议最终生成机器可读 CSV：

```text
benchmark,backbone,variant,seed,domain,successes,trials,success_rate,checkpoint_sha,manifest_sha,git_commit
```

再由 CSV 生成论文表，禁止手工从多个 log 复制百分数。

## 15. 公平性和统计检查

- 五个 domain 必须共享同一组 task/init-state base seeds，只有 lighting/noise 条件按冻结 manifest 变化。
- 所有 variant 的 rollout keys 集合必须完全相同。
- 训练 seed 与 evaluator seed 是不同概念，两者都要记录。
- 报告 Wilson 95% interval 或 bootstrap interval；多训练 seed 时优先对 seed-level aggregate 做均值/标准差。
- 同一 seed 下可利用 paired rollout keys 比较 Full 与 ablation，报告 paired success difference。
- 同时报告 trainable parameter count、单帧 latency 和峰值显存，尤其是三种 recurrent topology。
- 不能根据五域 test success 调 loss weight、训练阶段或 checkpoint；任何调参使用独立 validation set。

## 16. 最终提交前 checklist

- [ ] 代码 commit 已冻结，worktree 无未归档的实验代码修改。
- [ ] 10 个 variant 的语义与第 3 节一致。
- [ ] 单元测试全部通过。
- [ ] 每个 config 的绝对路径已迁移并 resolve 保存。
- [ ] 每个 run 只训练 RawVLA；backbone frozen + eval。
- [ ] Qwen3-OFT 使用相同 2k budget 和 step-2000 checkpoint。
- [ ] π0.5 所有 variant 使用同一种协议：全部五阶段或全部 1k；两者不混用。
- [ ] LIBERO 输入是 manifest RAW，RoboTwin 输入是 native `raw_linear`。
- [ ] RoboTwin 评测未误用 `default_isp` runner。
- [ ] eval config 与 checkpoint metadata 的 variant 完全一致。
- [ ] 每个 episode 开始时 recurrent state/theta 已 reset。
- [ ] LIBERO 每 variant 完成 10,000 条，RoboTwin 每 variant 完成 3,250 条。
- [ ] 结果由 episode JSONL/CSV 自动汇总，五域 Avg 为宏平均。
- [ ] checkpoint、manifest、config、代码 commit 和环境版本均已归档。

## 17. 推荐执行顺序

```text
实现 7 个 ablation flags + 4 个 recurrent modes
  -> 单元测试
  -> 生成 10 个 Qwen3-OFT config + 10 个 π0.5 config chains
  -> 每条线先跑 full/no_burst/no_hidden 的 20-step smoke
  -> 固定样本 strict-load 和图像检查
  -> 全量训练
  -> 冻结 checkpoint 清单和 SHA-256
  -> LIBERO 10k/variant 闭环
  -> RoboTwin 195/variant smoke
  -> RoboTwin 3250/variant 正式闭环
  -> 完整性/paired-key 审计
  -> 自动生成 Table 4
```

按 1 个训练 seed 计算，正式闭环总量为：

```text
LIBERO:   10 variants × 10,000 = 100,000 rollouts
RoboTwin: 10 variants ×  3,250 =  32,500 rollouts
合计:                              132,500 rollouts
```

若使用 3 个训练 seed，则为 397,500 条闭环 rollout。提交任务前应据单条 episode 的实测耗时和失败重试率做 GPU-hour 预算。
