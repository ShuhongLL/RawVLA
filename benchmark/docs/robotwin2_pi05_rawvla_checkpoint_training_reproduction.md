# RoboTwin 2.0 π0.5 RAW-VLA checkpoint 训练与跨服务器复现

本文专门说明下面这个 checkpoint 的来源、训练数据、五阶段训练链、运行命令和验收方法：

```text
baselines/rawvla/checkpoints/robotwin/pi0.5/
└── rawvla_robotwin_pi0.5_extremelow_refined.pt
```

本文依据仓库内保存的五份 resolved training config、训练器、dataset loader、数据生成脚本和最终渲染记录编写。若本文摘要与 resolved YAML 有差异，以
[`docs/repro_configs/rawvla/pi05_*.full.yaml`](../../docs/repro_configs/rawvla/)
为准。

## 1. 先明确 checkpoint 是什么

它是 **RAW-VLA 图像前端的 state dict**，不是完整的 π0.5 policy，也不包含 optimizer、scheduler 或训练步数状态。

训练和推理链路为：

```text
三相机 native linear RAW，float32 [0,1]
  -> RAW-VLA（本 checkpoint）
  -> 三相机 RGB，float32 [0,1]
  -> resize/pad 到 224×224，再映射为 [-1,1]
  -> 冻结的 RoboTwin π0.5 policy
  -> 50-step action chunk
```

最终文件来自第五阶段的：

```text
pi05_elm_lr07/checkpoints/steps_100_raw_frontend_pytorch_model.pt
```

它并非“随机初始化后只训练 100 steps”。完整谱系是：

```text
随机初始化 RAW-VLA
  -> stage 1: 1000 steps，全 frontend
  -> stage 2:  300 steps，全 frontend
  -> stage 3:  200 steps，全 frontend
  -> stage 4:  300 steps，仅 chroma 分支
  -> stage 5:  100 steps，仅 exposure head
  -> rawvla_robotwin_pi0.5_extremelow_refined.pt
```

总计 1900 个 optimizer steps。每个阶段 `is_resume: false`：只从前一阶段加载模型权重，**不会**继承 optimizer/scheduler；每阶段重新创建 AdamW 和学习率调度器。

### 1.1 文件完整性

```bash
sha256sum \
  baselines/rawvla/checkpoints/robotwin/pi0.5/rawvla_robotwin_pi0.5_extremelow_refined.pt
```

期望值：

```text
c5929f5b04fc64a82f2cfc51bffd61dc05466213b2cc1f4921eeba2e45779594
```

配套参考图和指标：

| 文件 | SHA-256 | 用途 |
|---|---|---|
| `rawvla_robotwin_pi0.5_extremelow_refined.pt` | `c5929f5b04fc64a82f2cfc51bffd61dc05466213b2cc1f4921eeba2e45779594` | frontend 权重 |
| `rawvla_robotwin_pi0.5_extremelow_refined.png` | `59b6a95a5e208e4f406299a5133687d6dc0d4d5dad078cf6e7073abd92525626` | 五光照离线渲染参考图 |
| `rawvla_robotwin_pi0.5_extremelow_refined.json` | `fcf5a78004a411ab58c00817aa6f5ea9265724423a7a511ed12ec59359171810` | 参考样本和数值指标 |

若目标只是把**同一个模型**部署到另一台服务器，直接复制 `.pt` 并验证 SHA-256 即可；重新训练通常只能做到协议和数值复现，跨 GPU/CUDA 不保证产生相同 hash。

### 1.2 已知的 bitwise 复现限制

当前历史训练入口的执行顺序是：

```python
vla = build_framework(cfg)       # 这里已经随机初始化 RAW-VLA
...
trainer.prepare_training()       # 这里才执行 set_seed(cfg.seed)
```

因此 stage 1 YAML 中虽然记录了 `seed: 20260906`，这个 seed 会控制训练期随机过程，却**没有固定 stage 1 RAW-VLA 的初始随机权重**。仓库也没有保存 stage 1 初始化前的 state dict 或当时进程的初始 PyTorch RNG state。由此得到两个结论：

- 从发布的最终 `.pt` 复制并验证 hash，可以精确复现发布模型；
- 仅靠源码、数据和五份 YAML 从零重训，不能承诺得到相同 SHA-256，即使所有可见配置均相同。

如果目标是方法/结果复现，可在新实验入口中把 `set_seed(cfg.seed)` 移到 `build_framework(cfg)` 之前，使新的实验自身可重复；但这会定义一条新的、确定性的初始化轨迹，并不等于恢复历史 stage 1 的未知初始化。若能从原服务器取回 stage 1 的 `steps_1000_raw_frontend_pytorch_model.pt`，则可以绕过最早的初始化不确定性，从 stage 2 开始复跑 refinement。

## 2. 必须固定的源码版本

已记录的仓库和 submodule revision 为：

```text
RAW-VLA:              b1dd12c
starVLA:              7de2f3d8baf20bcb51ebd48100c15b0a99296a5c
third_party/RoboTwin: 4b3ba0e926a6cd013ccaa4b79e2ab1e6ad017103
```

建议在新服务器先检查：

```bash
git rev-parse HEAD
git submodule status
git submodule update --init --recursive
```

训练关键实现：

| 内容 | 文件 |
|---|---|
| RAW-VLA 网络 | `baselines/rawvla/rawvla.py` |
| StarVLA frontend wrapper、strict load、定向冻结 | `starVLA/starVLA/model/modules/raw_frontend/registry.py` |
| π0/π0.5 图像和 action objective 接口 | `starVLA/starVLA/model/framework/VLM4A/PI0.py` |
| NPZ window sampler 和 normalization | `starVLA/starVLA/dataloader/rawvla_npz_datasets.py` |
| loss、梯度累积、保存逻辑 | `starVLA/starVLA/training/train_starvla.py` |

## 3. 软件和硬件环境

历史环境中实际观察到的核心版本保存在
[`environment-observed-20260831.txt`](../../docs/repro_configs/rawvla/environment-observed-20260831.txt)：

```text
Python       3.10.20
PyTorch      2.6.0+cu124
torchvision  0.21.0+cu124
Transformers 4.57.0
Accelerate   1.5.2
OmegaConf    2.3.0
NumPy        1.26.4
Pillow       12.3.0
einops       0.8.1
timm         1.0.27
h5py         3.16.0
```

参考安装方式：

```bash
conda create -n rawvla-pi05-repro python=3.10 -y
conda activate rawvla-pi05-repro

# 按服务器驱动安装 torch 2.6.0 / torchvision 0.21.0 的 CUDA 12.4 wheel。
pip install -r starVLA/requirements.txt
pip install \
  omegaconf==2.3.0 \
  accelerate==1.5.2 \
  transformers==4.57.0 \
  numpy==1.26.4 \
  pillow==12.3.0 \
  einops==0.8.1 \
  timm==1.0.27 \
  h5py==3.16.0 \
  sentencepiece \
  safetensors
```

历史路线的每个候选是一个单 GPU 进程，不是 8-GPU DDP。训练节点可用多卡并行跑超参数候选，但最终选中的五个阶段应在同一路径上串行 warm-start。显存不足时不要通过改变 batch、precision、图像分辨率或网络结构来“复现”；这些都会改变优化轨迹。

## 4. 准备冻结的 π0.5 backbone

使用的 policy 来源是 Hugging Face `SidneyXie/pi05_robotwin`，固定 revision：

```text
e49e2ab6c11f07511573b67261bd129e88d0a416
```

下载并导出本项目 dataset loader 所需的 normalization JSON：

```bash
export RAWVLA_ROOT=/path/to/RAW-VLA
export MODEL_ROOT=/path/to/robotwin2_backbones

cd "$RAWVLA_ROOT"
python -m pip install 'huggingface_hub[cli]'
bash backbone/robotwin/download_pi05.sh "$MODEL_ROOT/pi05_robotwin"

python backbone/robotwin/export_pi05_policy_norm_stats.py \
  --checkpoint-dir "$MODEL_ROOT/pi05_robotwin"

sha256sum "$MODEL_ROOT/pi05_robotwin/policy_norm_stats.json"
```

期望 normalization hash：

```text
229a4a2d6cfd1ba4eb2bc0b3aa57f049094032939130e4901485e9ea9910f5f7
```

训练配置使用：

```text
backbone: $MODEL_ROOT/pi05_robotwin/model.safetensors
stats:    $MODEL_ROOT/pi05_robotwin/policy_norm_stats.json
```

不要转换 `model.safetensors`，不要从五光照训练数据重算均值和标准差。还需要一个与该实现配套的 `paligemma_tokenizer.model`，在五份 YAML 中设置 `framework.tokenizer.model_path`。

π0.5 的固定接口参数为：

```yaml
framework:
  name: PI05
  precision: bfloat16
  action_dim: 32
  action_horizon: 50
  max_state_dim: 32
  discrete_state_input: true
  max_token_len: 200
  num_inference_steps: 10
  image_resolution: [224, 224]
  image_keys: [base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb]
```

π0.5 的 VLM 和 action head 全程冻结并保持 eval mode，但不使用 `torch.no_grad()` 包住 backbone；action loss 对图像输入的梯度必须继续穿过冻结 backbone 回传到 RAW-VLA。

## 5. 训练数据如何生成

### 5.1 数据集规模和目录

历史训练 cache：

```text
rawvla_light_train_statecopy_633/
├── raw/<task>/episode_XXXXXX.npz
└── rgb/<task>/episode_XXXXXX.npz
```

计划 633 条，实际成功配对 632 条；缺失的是 `put_bottles_dustbin/episode31`。必须按 NPZ 的实际 `metadata_json` 统计，而不能只相信旧 manifest 顶层的计划计数：

| domain | EV 区间 | 实际 NPZ 数量 |
|---|---:|---:|
| ExtremeLow | `[-9.0, -8.5]` | 127 |
| Low | `[-7.0, -6.0]` | 127 |
| Normal | `[-0.5, 0.5]` | 127 |
| Over | `[1.0, 1.5]` | 126 |
| ExtremeOver | `[2.0, 2.5]` | 125 |

数据覆盖 13 个 RoboTwin2.0 task。固定 manifest 位于：

```text
artifacts/rawvla_reference_renderer_release/train_manifest.json
```

### 5.2 paired state-copy 协议

这些 RAW **不是**由普通 RGB 图像做 inverse ISP 得到的。每条样本使用两个同 seed、同拓扑的 RoboTwin scene：

1. clean scene 在原始光照下 replay 成功轨迹，提供 action、robot state 和 clean RGB；
2. RAW scene 使用抽到的五档光照，不执行动作；
3. 每个拍照时刻，把 clean scene 的 actor pose/velocity、articulation root state、qpos/qvel 完整复制给 RAW scene；
4. 更新渲染后，从 RAW scene 读取 `HdrColor` 并生成 native linear RAW；
5. clean scene 的 `Color` 保存成逐帧配准的 RGB target。

这样不会因为暗光下重新运行 expert 而产生不同物理轨迹。

光照配置：

```yaml
lighting_strategy: environment_only
lighting_distribution: task-stratified_balanced_random_domain_then_uniform_ev
seed: 20260824
```

`environment_only` 将场景原有 ambient/directional/point light 颜色统一乘以 `2**lighting_ev`；不启用额外 key/flood light，也不另加 sensor saturation EV。

native RAW 形成过程：

```python
hdr = camera.get_picture("HdrColor")[..., :3].astype(np.float32)
inverse_wb = np.array([1 / 1.8, 1.0, 1 / 1.7], np.float32)
raw = np.clip(hdr * inverse_wb / 3.5, 0.0, 1.0).astype(np.float32)

variance = 2.5e-5 * raw + 3.90625e-8
raw = np.clip(raw + Normal(0, sqrt(variance)), 0, 1).astype(np.float32)
```

即：

```yaml
representation: three_channel_linear_float32
range: [0.0, 1.0]
black_level_corrected: true
quantized: false
raw_white_level: 3.5
shot_noise: 0.000025
read_noise: 0.0000000390625
```

训练读取的三个 key 是：

```text
head_camera_raw_float32
left_camera_raw_float32
right_camera_raw_float32
```

不要先转 uint8/uint16，不要用 legacy RGB-unprocess/RAW10 路径替代 native HDR RAW。

### 5.3 数据迁移或重建

最可靠的方法是把原 `raw/`、`rgb/` cache 原样复制到新服务器。若必须从源 RoboTwin 数据重建，需先准备 633 条成功 clean replay、对应 HDF5 和 `_traj_data/*.pkl`，然后使用：

```bash
python scripts/build_robotwin2_lighting_train_manifest.py \
  --clean-replay-root /path/to/clean_replay_root \
  --dataset-root /path/to/robotwin_dataset \
  --output /path/to/train_manifest.json \
  --seed 20260824

# 可以把 worker-index 0..N-1 分发到独立进程/容器。
python scripts/replay_robotwin2_paired_lighting_training.py \
  --robotwin-root "$RAWVLA_ROOT/third_party/RoboTwin" \
  --manifest /path/to/train_manifest.json \
  --output-root /path/to/rawvla_light_train_statecopy_633 \
  --num-workers N \
  --worker-index 0 \
  --compression stored

# 所有 worker 完成后汇总并检查缺失项。
python scripts/replay_robotwin2_paired_lighting_training.py \
  --robotwin-root "$RAWVLA_ROOT/third_party/RoboTwin" \
  --manifest /path/to/train_manifest.json \
  --output-root /path/to/rawvla_light_train_statecopy_633 \
  --finalize-only
```

重建数据还依赖 SAPIEN、RoboTwin assets、成功轨迹和 GPU renderer；若目标是 checkpoint 数值复现，优先迁移原 cache。

## 6. dataset window、采样和归一化

每个样本包含两个不能合并的时间轴：

```text
磁盘/loader: [T_rnn=6, K_burst=6, H, W, C=3]
模型:        [B_view, T_rnn=6, K_burst=6, C=3, H, W]
```

- `T_rnn=6`：六个 policy observation endpoint，间隔 `observation_stride=5`；
- `K_burst=6`：每个 observation 内连续六帧，用于 FFT burst denoise；
- 序列开头通过 clip 到 frame 0 做 causal padding，不跨 episode；
- 三个相机 view 共享同一时刻，但作为 frontend batch 的三个元素处理。

对抽到的 `current`：

```python
observation_indices = clip(current - 5 * arange(5, -1, -1))
burst_indices = clip(observation_indices[:, None] - arange(5, -1, -1)[None, :])
action_indices = clip(arange(current + 1, current + 1 + 50))
```

共同 dataset 参数：

```yaml
dataset_format: robotwin2
samples_per_episode: 32
virtual_length: 100000
per_device_batch_size: 1
num_workers: 1
prefetch_factor: 1
persistent_workers: true
pin_memory: true
burst_frames: 6
rnn_frames: 6
observation_stride: 5
action_horizon: 50
action_offset: 1
state_source_key: actions
normalization: zscore
stats_key: norm_stats
action_stats_key: actions
state_stats_key: state
delta_action_indices: []
```

这里 `delta_action_indices: []` 很重要：π0.5 不做 π0 路线使用的 12 个 arm 维 delta 转换。当前帧的 `actions` 向量同时作为低维 state，action 和 state 均使用下载的 π0.5 stats 做 z-score。

loader 的 episode 路径按字典序排序。没有 domain oversampling 时，每 32 个虚拟 index 换一个 episode；每个 index 使用：

```python
rng = np.random.default_rng(dataset_seed + index * 104729)
```

五份配置没有显式设置 `datasets.vla_data.seed`，所以 loader 的实际默认 seed 是 `42`。stage 5 配置了 domain sampling weights，它以同一确定性规则按权重抽 episode。

## 7. RAW-VLA 网络固定结构

五阶段的公共结构：

```yaml
name: rawvla
source_key: raw_burst_float32
input_format: rgb_raw
output_contract: rgb_chw_float32_0_1
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
max_exposure_ev: 10.0
fixed_update_alpha: 1.0
```

主要含义：

- 当前帧锚定的 patchwise FFT burst merge；
- 六层共享 spatial CNN；
- 绝对 luminance descriptor 与 scale-invariant chroma descriptor 分离；
- 64 维 luminance GRU + 64 维 chroma GRU；
- 标量 exposure gain 为 `2 ** (10 * tanh(raw_exposure))`；
- scale-free WB/CCM、单调 Bernstein tone curve；
- local color/tone residual 均关闭；
- `fixed_update_alpha=1`，不再学习跨时刻 update alpha。

stage 4 开始启用并保留：

```yaml
gray_preserving_ccm: true
```

它约束 CCM residual 的每一行和，避免 neutral gray 被 CCM 染色。最终 checkpoint 推理时也必须用 `max_exposure_ev=10.0` 和 `gray_preserving_ccm=true` 构建；这些设置不改变 state-dict key，配错时 strict load 仍可能通过，但输出会错。

## 8. loss 的精确定义

训练器计算：

```text
L_total = λ_action * L_pi05_action
        + λ_exposure * L_exposure
        + λ_chroma_prior * L_chroma_prior
        + λ_chroma_sup * L_chroma_supervision
```

- `L_pi05_action`：冻结 π0.5 原生 action objective，保留对 frontend 图像输入的梯度；
- `L_exposure`：输出平均亮度到目标亮度的 two-sided L1，可按 lighting domain 改 target 和 multiplier；
- `L_chroma_prior`：本五阶段配置权重始终为 0；
- `L_chroma_supervision`：与逐帧配准 clean RGB 的 chroma descriptor 做 Smooth-L1，不监督 target luminance；
- exposure/chroma 辅助渲染采用 STE clamp：forward 是真实 `[0,1]` clamp，backward 在越界处仍保留梯度。

注意 domain multiplier 是 objective 内部的 sample multiplier，外面还要乘该 objective 的全局 loss weight。例如 stage 2 ExtremeLow exposure 项的有效外层系数是 `0.03 × 125`。

## 9. 五阶段训练链

五份原始 resolved YAML 及校验值：

| 阶段 | 配置 | SHA-256 |
|---|---|---|
| 1 | [`pi05_stage1_strong_rgb_chroma_1k.full.yaml`](../../docs/repro_configs/rawvla/pi05_stage1_strong_rgb_chroma_1k.full.yaml) | `abe56365d1a1d9f1e3e75d9911d7d76322c619f97b06e7c4390338e416cdc181` |
| 2 | [`pi05_stage2_warm_refine_300.full.yaml`](../../docs/repro_configs/rawvla/pi05_stage2_warm_refine_300.full.yaml) | `ea6f74dd56daa59280e22af3100391d5e616e0d047a5e45837fb6190da42b658` |
| 3 | [`pi05_stage3_highlight_low_200.full.yaml`](../../docs/repro_configs/rawvla/pi05_stage3_highlight_low_200.full.yaml) | `f6ecddf756cbd565e9c9a874c85b007dd55ee64c61fd9e039e0cf8202c65edf1` |
| 4 | [`pi05_stage4_gray_ccm_300.full.yaml`](../../docs/repro_configs/rawvla/pi05_stage4_gray_ccm_300.full.yaml) | `eb134693552f2fa0a46b4f03829f3c853b4ec498b6c77e1108c23902179f736b` |
| 5 | [`pi05_stage5_extremelow_100.full.yaml`](../../docs/repro_configs/rawvla/pi05_stage5_extremelow_100.full.yaml) | `3a8497598ea3bdae55dbce9428b1fe3a0d2bba3c133149f83227088a7fe15246` |

所有阶段共有：

```yaml
single_gpu: true
per_device_batch_size: 1
gradient_accumulation_steps: 8
effective_batch_size: 8
optimizer: AdamW
betas: [0.9, 0.95]
eps: 1.0e-8
weight_decay: 1.0e-10
gradient_clipping: 1.0
max_grad_norm: 1.0
freeze_modules: vlm,action_head
frozen_modules_eval: true
require_only_trainable_module: raw_frontend
save_trainable_only: true
save_format: pt
```

“step”指 optimizer step，而不是 accumulation microstep；完整链约消费 `1900 × 8 = 15200` 个单样本 window。

### 9.1 Stage 1：strong RGB chroma，1000 steps

```text
run_id: pi05_lr5e5_l10_x100_c300
配置记录的训练期 seed: 20260906（不追溯控制构建时的随机初始化，见 1.2）
warm start: 无，RAW-VLA 随机初始化
trainable: 全部 RAW-VLA frontend
steps / warmup: 1000 / 50
LR: 5e-5，cosine_with_min_lr，min_lr=1e-6
action weight: 0.10
exposure weight: 0.03，默认 target=0.60
chroma supervision weight: 0.30，beta=0.05，有效 target luma=[0.02,0.98]
```

exposure multiplier：

```yaml
Low: 10.0
ExtremeLow: 100.0
```

输出：

```text
<stage1_root>/pi05_lr5e5_l10_x100_c300/checkpoints/
└── steps_1000_raw_frontend_pytorch_model.pt
```

### 9.2 Stage 2：warm refinement，300 steps

```text
run_id: pi05_w_x125_n070_ec4
seed: 20260906
warm start: stage 1 step 1000
trainable: 全部 RAW-VLA frontend
steps / warmup: 300 / 15
LR: 1e-5，cosine_with_min_lr，min_lr=1e-6
action weight: 0.10
exposure weight: 0.03
chroma supervision weight: 0.30
```

domain exposure targets 和 multipliers：

| domain | target | multiplier |
|---|---:|---:|
| ExtremeLow | 0.68 | 125 |
| Low | 0.60 | 10 |
| Normal | 0.70 | 2 |
| Over | 0.58 | 1 |
| ExtremeOver | 0.58 | 2 |

chroma supervision multiplier：`Over=1.5`、`ExtremeOver=4.0`。

### 9.3 Stage 3：highlight/Low refinement，200 steps

```text
run_id: pi05_h_l052_w20_ec8
seed: 20260909
warm start: stage 2 step 300
trainable: 全部 RAW-VLA frontend
steps / warmup: 200 / 10
LR: 5e-6，cosine_with_min_lr，min_lr=1e-6
action weight: 0.10
exposure weight: 0.03
chroma supervision weight: 0.30
chroma supervision target luma max: 1.0
```

domain exposure targets 和 multipliers：

| domain | target | multiplier |
|---|---:|---:|
| ExtremeLow | 0.55 | 125 |
| Low | 0.52 | 20 |
| Normal | 0.70 | 2 |
| Over | 0.58 | 1 |
| ExtremeOver | 0.55 | 2 |

chroma supervision multiplier：`Over=1.5`、`ExtremeOver=8.0`。

### 9.4 Stage 4：gray-preserving CCM/chroma adaptation，300 steps

```text
run_id: pi05_gc_lr3_c100_x2
seed: 20260915
warm start: stage 3 step 200
gray_preserving_ccm: true
steps / warmup: 300 / 15
LR: 3e-5，cosine_with_min_lr，min_lr=1e-6
action weight: 0.05
exposure weight: 0
chroma supervision weight: 1.0
chroma multiplier: Over=1.5，ExtremeOver=2.0
```

只允许以下 chroma 模块更新：

```yaml
trainable_parameter_prefixes:
  - model.chroma_encoder.
  - model.chroma_theta_encoder.
  - model.chroma_fusion.
  - model.chroma_gru.
  - model.chroma_head.
  - model.chroma_update_gate.
```

历史审计值：24 个 tensor、174,121 个参数可训练；π0.5 及 RAW-VLA 的 luma/exposure/tone/denoise/spatial 分支冻结。

### 9.5 Stage 5：ExtremeLow exposure micro-refinement，100 steps

```text
run_id: pi05_elm_lr07
seed: 20261007
warm start: stage 4 step 300
gray_preserving_ccm: true
steps / warmup: 100 / 5
LR: constant 7e-7（min_lr 同为 7e-7）
action weight: 0
exposure weight: 1.0
exposure target: ExtremeLow=0.55
chroma supervision weight: 0
```

只允许：

```yaml
trainable_parameter_prefixes:
  - model.exposure_head.
```

历史审计值：4 个 tensor、33,025 个参数可训练。采样权重：

```yaml
ExtremeLow: 1000.0
Low: 0.001
Normal: 0.001
Over: 0.001
ExtremeOver: 0.001
```

所以最终 100 steps 实质上是 ExtremeLow-only 的 exposure-head 微调；action loss 和 chroma loss 都不贡献梯度。

## 10. 在新服务器串行运行五个阶段

### 10.1 设置路径

```bash
export RAWVLA_ROOT=/path/to/RAW-VLA
export MODEL_ROOT=/path/to/robotwin2_backbones
export DATA_ROOT=/path/to/rawvla_light_train_statecopy_633
export TOKENIZER_PATH=/path/to/paligemma_tokenizer.model
export RUN_ROOT=/path/to/rawvla_pi05_reproduction

export PYTHONPATH="$RAWVLA_ROOT:$RAWVLA_ROOT/starVLA:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=30200

mkdir -p "$RUN_ROOT"
cd "$RAWVLA_ROOT/starVLA"
```

下面使用 CLI override 替换历史 YAML 中的机器绝对路径，而不改动其余超参数。`setup_directories()` 会用 `run_root_dir/run_id` 重新计算真正的 output directory，因此 YAML 尾部的旧 `output_dir` 不生效。

为了便于阅读，定义公共参数数组：

```bash
COMMON_OVERRIDES=(
  --framework.tokenizer.model_path "$TOKENIZER_PATH"
  --datasets.vla_data.cache_root "$DATA_ROOT/raw"
  --datasets.vla_data.rgb_cache_root "$DATA_ROOT/rgb"
  --datasets.vla_data.stats_path "$MODEL_ROOT/pi05_robotwin/policy_norm_stats.json"
  --trainer.pretrained_checkpoint "$MODEL_ROOT/pi05_robotwin/model.safetensors"
)
```

### 10.2 Stage 1

```bash
python starVLA/training/train_starvla.py \
  --config_yaml "$RAWVLA_ROOT/docs/repro_configs/rawvla/pi05_stage1_strong_rgb_chroma_1k.full.yaml" \
  --run_root_dir "$RUN_ROOT/stage1" \
  "${COMMON_OVERRIDES[@]}"

export STAGE1_CKPT="$RUN_ROOT/stage1/pi05_lr5e5_l10_x100_c300/checkpoints/steps_1000_raw_frontend_pytorch_model.pt"
test -f "$STAGE1_CKPT"
```

### 10.3 Stage 2

```bash
python starVLA/training/train_starvla.py \
  --config_yaml "$RAWVLA_ROOT/docs/repro_configs/rawvla/pi05_stage2_warm_refine_300.full.yaml" \
  --run_root_dir "$RUN_ROOT/stage2" \
  --framework.raw_frontend.checkpoint "$STAGE1_CKPT" \
  "${COMMON_OVERRIDES[@]}"

export STAGE2_CKPT="$RUN_ROOT/stage2/pi05_w_x125_n070_ec4/checkpoints/steps_300_raw_frontend_pytorch_model.pt"
test -f "$STAGE2_CKPT"
```

### 10.4 Stage 3

```bash
python starVLA/training/train_starvla.py \
  --config_yaml "$RAWVLA_ROOT/docs/repro_configs/rawvla/pi05_stage3_highlight_low_200.full.yaml" \
  --run_root_dir "$RUN_ROOT/stage3" \
  --framework.raw_frontend.checkpoint "$STAGE2_CKPT" \
  "${COMMON_OVERRIDES[@]}"

export STAGE3_CKPT="$RUN_ROOT/stage3/pi05_h_l052_w20_ec8/checkpoints/steps_200_raw_frontend_pytorch_model.pt"
test -f "$STAGE3_CKPT"
```

### 10.5 Stage 4

```bash
python starVLA/training/train_starvla.py \
  --config_yaml "$RAWVLA_ROOT/docs/repro_configs/rawvla/pi05_stage4_gray_ccm_300.full.yaml" \
  --run_root_dir "$RUN_ROOT/stage4" \
  --framework.raw_frontend.checkpoint "$STAGE3_CKPT" \
  "${COMMON_OVERRIDES[@]}"

export STAGE4_CKPT="$RUN_ROOT/stage4/pi05_gc_lr3_c100_x2/checkpoints/steps_300_raw_frontend_pytorch_model.pt"
test -f "$STAGE4_CKPT"
```

### 10.6 Stage 5 和最终导出

```bash
python starVLA/training/train_starvla.py \
  --config_yaml "$RAWVLA_ROOT/docs/repro_configs/rawvla/pi05_stage5_extremelow_100.full.yaml" \
  --run_root_dir "$RUN_ROOT/stage5" \
  --framework.raw_frontend.checkpoint "$STAGE4_CKPT" \
  "${COMMON_OVERRIDES[@]}"

export FINAL_CKPT="$RUN_ROOT/stage5/pi05_elm_lr07/checkpoints/steps_100_raw_frontend_pytorch_model.pt"
test -f "$FINAL_CKPT"
sha256sum "$FINAL_CKPT"
```

训练器保存的是 `raw_frontend.state_dict()`，文件名为 `steps_<N>_raw_frontend_pytorch_model.pt`。如果要按仓库命名部署：

```bash
cp "$FINAL_CKPT" /path/to/export/rawvla_robotwin_pi0.5_extremelow_refined.pt
```

## 11. 训练前和训练中的强制检查

### 11.1 数据完整性

```bash
python - <<'PY'
from collections import Counter
from pathlib import Path
import json
import numpy as np
import os

root = Path(os.environ["DATA_ROOT"])
raw = {p.relative_to(root / "raw") for p in (root / "raw").glob("**/*.npz")}
rgb = {p.relative_to(root / "rgb") for p in (root / "rgb").glob("**/*.npz")}
assert raw == rgb, (len(raw - rgb), len(rgb - raw))
assert len(raw) == 632, len(raw)

domains = Counter()
for rel in sorted(raw):
    with np.load(root / "raw" / rel, allow_pickle=False) as a:
        m = json.loads(str(a["metadata_json"].item()))
        domains[m["lighting_domain"]] += 1
        for key in (
            "head_camera_raw_float32",
            "left_camera_raw_float32",
            "right_camera_raw_float32",
        ):
            x = a[key]
            assert x.dtype == np.float32, (rel, key, x.dtype)
            assert np.isfinite(x).all(), (rel, key)
            assert x.min() >= 0 and x.max() <= 1, (rel, key, x.min(), x.max())

print(domains)
PY
```

期望：

```text
ExtremeLow=127, Low=127, Normal=127, Over=126, ExtremeOver=125
```

### 11.2 参数冻结

每阶段日志必须出现 only-trainable contract 通过，并检查：

- stage 1–3：只有 `raw_frontend.*` 可训练；
- stage 4：仅六个 `model.chroma_*` prefix；
- stage 5：仅 `model.exposure_head.*`，历史值为 33,025 参数；
- `vlm` 和 `action_head` 参数全部 `requires_grad=False` 且保持 eval；
- action-to-ISP gradient 在 stage 1–4 非零；stage 5 的 weighted action gradient 为零是预期行为。

每 20 optimizer steps 记录：

```text
action_loss
exposure_prior_loss
chroma_supervision_loss
total weighted loss
grad_norm/action_to_isp
grad_norm/weighted_action_to_isp
grad_norm/weighted_exposure_to_isp
```

同时检查 frontend 输出无 NaN、dtype/range 正确，且没有 uint8 往返。

## 12. 最终 checkpoint 离线验收

使用固定 renderer：

```bash
cd "$RAWVLA_ROOT"
export PYTHONPATH="$RAWVLA_ROOT:$RAWVLA_ROOT/starVLA:${PYTHONPATH:-}"

python scripts/render_rawvla_checkpoint_before_after.py \
  --config docs/robotwin2_rawvla_reference_render_config.yaml \
  --checkpoint "$FINAL_CKPT" \
  --cache-root "$DATA_ROOT/raw" \
  --rgb-cache-root "$DATA_ROOT/rgb" \
  --output-dir "$RUN_ROOT/final_render" \
  --device cuda
```

renderer 必须以 strict 方式加载，并使用：

```yaml
rnn_frames: 6
burst_frames: 6
observation_stride: 5
max_exposure_ev: 10.0
gray_preserving_ccm: true
```

参考样本 `grab_roller/episode_000004.npz`、ExtremeLow、current frame 50、head view 的回归值：

```text
RAW mean: 0.00034283509012311697
ISP mean: 0.31113675236701965
```

五档全部参考值在 checkpoint 同目录的 JSON 中。输入索引不是一个单独 6-frame burst，而是：

```text
outer observation endpoints: [25, 30, 35, 40, 45, 50]
每个 endpoint 再取以它结尾的连续 6 帧
```

若 RAW mean 正确而 ISP mean 明显不符，依次检查：

1. checkpoint SHA-256；
2. `max_exposure_ev` 是否误用了默认值 6；
3. 是否传入 `[B,6,6,3,H,W]`，而不是 `[B,6,3,H,W]`；
4. wrapper 是否逐 observation 传递 recurrent `state` 和 `theta`；
5. `gray_preserving_ccm=true`；
6. 是否 strict load；
7. 是否额外加了 gamma、ImageNet normalization、RGB/BGR 交换或 uint8 转换。

保存 PNG 的唯一转换应为：

```python
u8 = (output.float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
```

## 13. “复现成功”的分级标准

建议按以下顺序验收：

1. **资产复现**：源码/submodule revision、π0.5 revision、stats hash、训练 YAML hash 全部一致；
2. **数据复现**：632 对相对路径一致，domain 计数一致，固定样本 RAW mean 一致；
3. **结构复现**：每阶段 strict warm-start 成功，trainable 参数范围一致；
4. **训练复现**：loss/gradient 量级和五阶段趋势一致；
5. **数值复现**：固定样本 ISP mean 接近参考；
6. **bitwise 复现**：最终 checkpoint hash 完全一致；从零重训受第 1.2 节限制。

第 6 项要求最苛刻，而且现有材料不足以恢复历史 stage 1 的初始 RNG state。随机初始化、π0.5 action objective 内部随机量、fused AdamW、CUDA kernel、GPU 型号和库版本都可能破坏 bitwise determinism。因此，新服务器结果 hash 不同不自动代表方法复现失败；应先定位它从上述哪一级开始偏离。

## 14. 最常见的错误

- 把该 `.pt` 当成完整 π0.5 checkpoint；实际上它只能由 RAW-VLA registry adapter 加载。
- 只运行 stage 5 的 100 steps，却没有 stage 4 warm-start。
- 用 `is_resume: true` 或继承 optimizer state；历史流程每阶段都重建 optimizer/scheduler。
- 用 π0 的 normalization 或 delta-action 配置训练 π0.5。
- 用 RGB inverse ISP 伪造训练 RAW；历史数据来自 native `HdrColor`。
- 合并 `T_rnn` 和 `K_burst`，或把 outer stride 从 5 改成 1。
- 使用 `max_exposure_ev=6`；RoboTwin 最终路线是 10。
- stage 4/5 漏掉 `gray_preserving_ccm=true`。
- 把 paired clean RGB 当 reconstruction target；这里只监督 chroma descriptor，不监督 clean luminance/整图重建。
- 在 frontend 和 π0.5 之间转 uint8/PIL，或重复 gamma/normalization。
- 让多个独立 candidate 写入同一个 `run_root_dir/run_id`。

## 15. 相关文档

- [`rawvla_training_reproduction_qwen3oft_pi0_pi05.md`](../../docs/rawvla_training_reproduction_qwen3oft_pi0_pi05.md)：三条 backbone 路线总览；
- [`robotwin2_pi0_pi05_rawvla_reference_reproduction.md`](../../docs/robotwin2_pi0_pi05_rawvla_reference_reproduction.md)：最终 frontend 的逐像素渲染复现；
- [`backbone/robotwin/README.md`](../../backbone/robotwin/README.md)：π0/π0.5 backbone 下载和 normalization；
- [`baselines/rawvla/README.md`](../../baselines/rawvla/README.md)：RAW-VLA 结构和通用 objective；
- [`robotwin2_pi05_rawvla_extremelow_low_results.md`](../../docs/robotwin2_pi05_rawvla_extremelow_low_results.md)：最终 checkpoint 的 ExtremeLow/Low 闭环结果。
