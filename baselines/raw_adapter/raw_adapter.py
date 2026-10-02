"""Pure PyTorch RAW-Adapter core modules.

Class and function behavior follows the official ECCV_RAW_Adapter repository,
with typo-compatible aliases for the original class names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class RAWAdapterOutput:
    image: Tensor
    stages: list[Tensor]
    adapter: Tensor | None


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv7x7(in_planes: int, out_planes: int, stride: int = 3, padding: int = 3) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=padding, bias=False)


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def adapt_raw_input(raw: Tensor, input_format: str = "rgb_raw") -> Tensor:
    fmt = input_format.lower()
    if fmt in {"rgb_raw", "raw_rgb", "pseudo_rgb", "rgb"}:
        if raw.ndim != 4 or raw.shape[1] != 3:
            raise ValueError(f"Expected [B, 3, H, W] for {input_format}, got {tuple(raw.shape)}")
        return raw.float()
    if fmt in {"packed_bayer", "bayer4", "rgbg"}:
        if raw.ndim != 4 or raw.shape[1] != 4:
            raise ValueError(f"Expected [B, 4, H, W] for {input_format}, got {tuple(raw.shape)}")
        red, green_r, blue, green_b = raw[:, 0:1], raw[:, 1:2], raw[:, 2:3], raw[:, 3:4]
        return torch.cat((red, 0.5 * (green_r + green_b), blue), dim=1).float()
    raise ValueError(f"Unknown RAW-Adapter input_format: {input_format!r}")


def _gaussian_kernel1d(kernel_size: int, sigma: Tensor) -> Tensor:
    half = (kernel_size - 1) * 0.5
    x = torch.linspace(-half, half, steps=kernel_size, device=sigma.device, dtype=sigma.dtype)
    pdf = torch.exp(-0.5 * (x / sigma.clamp_min(1.0e-6)).pow(2))
    return pdf / pdf.sum().clamp_min(1.0e-12)


def gaussian_blur(img: Tensor, kernel_size: tuple[int, int], sigma: tuple[Tensor, Tensor]) -> Tensor:
    """Official dynamic Gaussian blur helper, minus torchvision dependencies."""

    need_squeeze = False
    if img.ndim == 3:
        img = img.unsqueeze(0)
        need_squeeze = True
    elif img.ndim != 4:
        raise ValueError(f"Expected CHW or NCHW image tensor, got {tuple(img.shape)}")

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    sigma_x, sigma_y = sigma
    kernel_x = _gaussian_kernel1d(kernel_size[0], sigma_x.to(dtype=dtype))
    kernel_y = _gaussian_kernel1d(kernel_size[1], sigma_y.to(dtype=dtype))
    kernel2d = torch.mm(kernel_y[:, None], kernel_x[None, :])
    kernel = kernel2d.expand(img.shape[-3], 1, kernel_size[1], kernel_size[0])
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    x = F.pad(img.to(dtype), padding, mode="reflect")
    out = F.conv2d(x, kernel, groups=img.shape[-3]).to(dtype=img.dtype)
    return out.squeeze(0) if need_squeeze else out


class QueryAdaptivePredictor(nn.Module):
    def __init__(self, query_count: int, dim: int = 64, num_heads: int = 1) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Parameter(torch.rand((1, query_count, dim)), requires_grad=True)
        self.kv_downsample = nn.Sequential(
            nn.Conv2d(3, dim // 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
        )
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.down = nn.Linear(dim, 1)

    def forward_queries(self, x: Tensor) -> Tensor:
        d_x = self.kv_downsample(x).flatten(2).transpose(1, 2)
        batch, tokens, channels = d_x.shape
        k = self.k(d_x).reshape(batch, tokens, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(d_x).reshape(batch, tokens, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)
        q = self.q.expand(batch, -1, -1).view(batch, -1, self.num_heads, channels // self.num_heads)
        q = q.permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, -1, channels)
        return self.down(self.proj(out)).squeeze(-1)


class KernelPredictor(QueryAdaptivePredictor):
    """Official RAW-Adapter ``Kernel_Predictor``."""

    def __init__(self, dim: int = 64, mode: str = "low", num_heads: int = 1) -> None:
        super().__init__(query_count=4, dim=dim, num_heads=num_heads)
        gain_base = 3.0 if mode == "low" else 1.0
        self.gain_base = nn.Parameter(torch.tensor([gain_base], dtype=torch.float32), requires_grad=(mode != "low"))
        self.r1_base = nn.Parameter(torch.tensor([3.0], dtype=torch.float32), requires_grad=False)
        self.r2_base = nn.Parameter(torch.tensor([2.0], dtype=torch.float32), requires_grad=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r1, r2, gain, sigma = torch.unbind(self.forward_queries(x), dim=1)
        r1 = 0.1 * r1 + self.r1_base
        r2 = 0.1 * r2 + self.r2_base
        gain = gain + self.gain_base
        return r1, r2, gain, torch.sigmoid(sigma)


class MatrixPredictor(QueryAdaptivePredictor):
    """Official RAW-Adapter ``Matrix_Predictor``."""

    def __init__(self, dim: int = 64, num_heads: int = 1) -> None:
        super().__init__(query_count=10, dim=dim, num_heads=num_heads)
        self.register_buffer("ccm_base", torch.eye(3), persistent=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        out = self.forward_queries(x)
        matrix_delta, distance = out[:, :9], out[:, 9:]
        ccm = 0.1 * matrix_delta.view(x.shape[0], 3, 3) + self.ccm_base
        return ccm, F.relu(distance) + 1.0


class NILUT(nn.Module):
    """Implicit neural 3D LUT from the official RAW-Adapter code."""

    def __init__(
        self,
        in_features: int = 3,
        hidden_features: int = 32,
        hidden_layers: int = 3,
        out_features: int = 3,
        res: bool = True,
    ) -> None:
        super().__init__()
        self.res = res
        layers: list[nn.Module] = [nn.Linear(in_features, hidden_features), nn.ReLU()]
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(hidden_features, hidden_features), nn.Tanh()])
        layers.append(nn.Linear(hidden_features, out_features))
        if not res:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, intensity: Tensor) -> Tensor:
        output = self.net(intensity)
        if self.res:
            output = (output + intensity).clamp(0.0, 1.0)
        return output


def shades_of_gray(img: Tensor, p: Tensor) -> Tensor:
    """Shades-of-Gray white balance, batched CHW -> HWC internally."""

    eps = torch.finfo(img.dtype).eps
    p_safe = p.view(-1, 1).clamp_min(1.0e-6)
    flat = img.flatten(2).clamp_min(eps)
    channel_avg = flat.pow(p_safe.unsqueeze(-1)).mean(dim=-1).pow(1.0 / p_safe)
    avg = flat.pow(p_safe.unsqueeze(-1)).mean(dim=(1, 2), keepdim=True).pow(1.0 / p_safe.unsqueeze(-1))
    gains = (channel_avg / avg.squeeze(-1)).clamp_min(eps)
    return (img / gains.unsqueeze(-1).unsqueeze(-1)).permute(0, 2, 3, 1)


def GainDenoise(I1: Tensor, r1: Tensor, r2: Tensor, gain: Tensor, sigma: Tensor, k_size: int = 3) -> Tensor:
    """Official ``Gain_Denoise`` operation."""

    outputs = []
    for i in range(I1.shape[0]):
        image_gain = gain[i] * I1[i]
        blur = gaussian_blur(image_gain, (k_size, k_size), (r1[i], r2[i]))
        sharp = blur + sigma[i] * (I1[i] - blur)
        outputs.append(sharp)
    return torch.stack(outputs, dim=0)


def wb_ccm(I2: Tensor, ccm_matrix: Tensor, distance: Tensor) -> tuple[Tensor, Tensor]:
    I3 = shades_of_gray(I2, distance)
    I4 = torch.einsum("bhwc,boc->bhwo", I3, ccm_matrix).clamp(1.0e-5, 1.0)
    return I3, I4


class InputLevelAdapter(nn.Module):
    def __init__(self, mode: str = "normal", lut_dim: int = 32, out: str = "all", k_size: int = 3, w_lut: bool = True):
        super().__init__()
        self.Predictor_K = KernelPredictor(dim=64, mode=mode)
        self.Predictor_M = MatrixPredictor(dim=64)
        self.w_lut = w_lut
        self.LUT = NILUT(hidden_features=lut_dim) if w_lut else None
        self.out = out
        self.k_size = k_size

    def forward(self, I1: Tensor) -> list[Tensor]:
        r1, r2, gain, sigma = self.Predictor_K(I1)
        I2 = GainDenoise(I1, r1, r2, gain, sigma, k_size=self.k_size).clamp(1.0e-5, 1.0)
        ccm_matrix, distance = self.Predictor_M(I2)
        I3, I4 = wb_ccm(I2, ccm_matrix, distance)
        I3_chw = I3.permute(0, 3, 1, 2)
        I4_chw = I4.permute(0, 3, 1, 2)
        if self.LUT is not None:
            I5 = self.LUT(I4).permute(0, 3, 1, 2)
            return [I1, I2, I3_chw, I4_chw, I5] if self.out == "all" else [I5]
        return [I1, I2, I3_chw, I4_chw] if self.out == "all" else [I4_chw]


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = None if stride == 1 and inplanes == planes else nn.Sequential(_conv1x1(inplanes, planes, stride), norm_layer(planes))

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ModelLevelAdapter(nn.Module):
    def __init__(self, in_c: int = 3, in_dim: int = 24, w_lut: bool = True):
        super().__init__()
        self.conv_1 = _conv3x3(in_c, in_c, 2)
        self.conv_2 = _conv3x3(in_c, in_c, 2)
        self.conv_3 = _conv3x3(in_c, in_c, 2)
        self.w_lut = w_lut
        if w_lut:
            self.conv_4 = _conv3x3(in_c, in_c, 2)
            self.uni_conv = _conv7x7(4 * in_c, in_dim, 2, padding=3)
        else:
            self.uni_conv = _conv7x7(3 * in_c, in_dim, 2, padding=3)
        self.res_1 = BasicBlock(in_dim, in_dim)
        self.res_2 = BasicBlock(in_dim, in_dim)

    def forward(self, images: list[Tensor]) -> Tensor:
        if self.w_lut:
            adapter = torch.cat(
                [self.conv_1(images[0]), self.conv_2(images[1]), self.conv_3(images[2]), self.conv_4(images[3])],
                dim=1,
            )
        else:
            adapter = torch.cat([self.conv_1(images[0]), self.conv_2(images[1]), self.conv_3(images[2])], dim=1)
        return self.res_2(self.res_1(self.uni_conv(adapter)))


class MergeBlock(nn.Module):
    def __init__(self, fea_c: int, ada_c: int, mid_c: int, return_ada: bool = True):
        super().__init__()
        self.conv_1 = _conv1x1(fea_c + ada_c, mid_c)
        self.conv_2 = _conv1x1(mid_c, fea_c)
        self.return_ada = return_ada
        self.conv_3 = _conv3x3(mid_c, ada_c * 2, stride=2) if return_ada else None

    def forward(self, fea: Tensor, adapter: Tensor, ratio: float = 1.0) -> tuple[Tensor, Tensor | None]:
        res = fea
        mixed = self.conv_1(torch.cat([fea, adapter], dim=1))
        fea_out = ratio * self.conv_2(mixed) + res
        if self.conv_3 is None:
            return fea_out, None
        return fea_out, self.conv_3(mixed)


class RAWAdapter(nn.Module):
    """Input-level plus optional model-level RAW-Adapter core."""

    def __init__(
        self,
        mode: str = "normal",
        lut_dim: int = 32,
        k_size: int = 3,
        w_lut: bool = True,
        model_adapter_dim: int | None = 24,
    ) -> None:
        super().__init__()
        self.input_adapter = InputLevelAdapter(mode=mode, lut_dim=lut_dim, k_size=k_size, w_lut=w_lut, out="all")
        self.model_adapter = ModelLevelAdapter(in_dim=model_adapter_dim, w_lut=w_lut) if model_adapter_dim is not None else None
        self.w_lut = w_lut

    def forward(self, raw: Tensor, *, input_format: str = "rgb_raw") -> RAWAdapterOutput:
        x = adapt_raw_input(raw, input_format=input_format).clamp(1.0e-5, 1.0)
        stages = self.input_adapter(x)
        image = stages[-1]
        adapter = None
        if self.model_adapter is not None:
            stage_count = 4 if self.w_lut else 3
            adapter = self.model_adapter(stages[:stage_count])
        return RAWAdapterOutput(image=image, stages=stages, adapter=adapter)


Input_level_Adapeter = InputLevelAdapter
Model_level_Adapeter = ModelLevelAdapter
Merge_block = MergeBlock
