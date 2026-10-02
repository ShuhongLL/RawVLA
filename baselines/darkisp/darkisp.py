"""Core Dark-ISP PyTorch modules.

The implementation follows the Dark-ISP paper equations:

    I' = P' I
    U = F(I')
    L_sb = sum_i |1 - cos(p'_i, p_tilde_i)|

where Bayer RAW is represented as four packed planes ordered ``[R, Gr, B, Gb]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class DarkISPOutput:
    """Outputs from the Dark-ISP plugin."""

    rgb: Tensor
    linear_rgb: Tensor
    raw_packed4: Tensor
    linear_matrix: Tensor


def make_camera_matrix(
    white_balance: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    color_matrix: Tensor | None = None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create the paper's ``P = C B W`` camera matrix with shape ``[3, 4]``."""

    wb = torch.diag(torch.tensor(white_balance, device=device, dtype=dtype))
    binning = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.5], [0.0, 0.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    )
    if color_matrix is None:
        ccm = torch.eye(3, device=device, dtype=dtype)
    else:
        ccm = color_matrix.to(device=device, dtype=dtype)
        if ccm.shape != (3, 3):
            raise ValueError(f"color_matrix must have shape [3, 3], got {tuple(ccm.shape)}")
    return ccm @ binning @ wb


def _zero_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DynamicLinearMapping(nn.Module):
    """Dynamic linear mapping module from Section 3.1.

    The paper defines local and global attention that produce pixel-level
    ``P_l`` and image-level ``P_g`` residual matrices. This module keeps that
    contract explicit: the trainable camera prior ``P`` is combined with a
    local residual map and a global residual vector to produce ``P'`` with shape
    ``[B, 3, 4, H, W]``.
    """

    def __init__(
        self,
        hidden_dim: int = 48,
        initial_matrix: Tensor | None = None,
        max_residual: float = 0.25,
    ) -> None:
        super().__init__()
        self.max_residual = float(max_residual)

        if initial_matrix is None:
            initial_matrix = make_camera_matrix()
        if initial_matrix.shape != (3, 4):
            raise ValueError(f"initial_matrix must have shape [3, 4], got {tuple(initial_matrix.shape)}")

        self.camera_matrix = nn.Parameter(initial_matrix.detach().clone().float())

        self.local_features = nn.Sequential(
            ConvBlock(4, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
        )
        self.local_matrix = nn.Conv2d(hidden_dim, 12, kernel_size=3, padding=1)
        _zero_init(self.local_matrix)

        self.global_features = nn.Sequential(
            ConvBlock(4, hidden_dim, stride=2),
            ConvBlock(hidden_dim, hidden_dim, stride=2),
            ConvBlock(hidden_dim, hidden_dim, stride=2),
            ConvBlock(hidden_dim, hidden_dim, stride=2),
        )
        self.global_matrix = nn.Linear(hidden_dim, 12)
        _zero_init(self.global_matrix)

    def forward(self, raw_packed4: Tensor) -> tuple[Tensor, Tensor]:
        if raw_packed4.ndim != 4 or raw_packed4.shape[1] != 4:
            raise ValueError(f"Expected raw_packed4 shape [B, 4, H, W], got {tuple(raw_packed4.shape)}")

        batch, _, height, width = raw_packed4.shape
        base = self.camera_matrix.view(1, 3, 4, 1, 1)

        local = torch.tanh(self.local_matrix(self.local_features(raw_packed4)))
        local = local.view(batch, 3, 4, height, width) * self.max_residual

        global_features = self.global_features(raw_packed4).mean(dim=(-2, -1))
        global_residual = torch.tanh(self.global_matrix(global_features))
        global_residual = global_residual.view(batch, 3, 4, 1, 1) * self.max_residual

        matrix = base + local + global_residual
        linear_rgb = torch.einsum("bochw,bchw->bohw", matrix, raw_packed4)
        return linear_rgb.clamp(0.0, 1.0), matrix


def polynomial_basis(x: Tensor) -> Tensor:
    """Return eight low-light tone bases, shape ``[..., 8, H, W]``.

    The paper constrains each basis to pass through ``(0, 0)`` and ``(1, 1)``
    and uses first through eighth order bases. The official formulas are only
    shown in a rendered figure in the arXiv source, so the bases are isolated
    here for easy replacement when official code is released.
    """

    x2 = x * x
    one_minus = 1.0 - x
    f1 = x
    f2 = 1.0 - one_minus.pow(2)
    f3 = 1.0 - one_minus.pow(3)
    f4 = 1.0 - one_minus.pow(4)
    f5 = x + 4.0 * x * one_minus * (0.5 - x)
    f6 = 1.0 - one_minus.pow(6)
    f7 = (1.0 - one_minus.pow(2)) + 3.0 * x.pow(3) * one_minus * (2.0 * x - 1.0)
    f8 = 0.35 * (1.0 - one_minus.pow(4)) + 0.65 * (3.0 * x2 - 2.0 * x2 * x)
    return torch.stack((f1, f2, f3, f4, f5, f6, f7, f8), dim=-3)


class PolynomialToneMapping(nn.Module):
    """Nonlinear stretch with polynomial bases from Section 3.2."""

    def __init__(
        self,
        hidden_dim: int = 48,
        order: int = 8,
        coefficient_scale: float = 0.25,
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        if order != 8:
            raise ValueError("The current Dark-ISP basis implementation uses order=8.")
        self.order = int(order)
        self.coefficient_scale = float(coefficient_scale)
        self.use_skip = bool(use_skip)

        self.coefficients = nn.Sequential(
            ConvBlock(3, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, 3 * self.order, kernel_size=3, padding=1),
        )
        _zero_init(self.coefficients[-1])

    def forward(self, linear_rgb: Tensor) -> Tensor:
        if linear_rgb.ndim != 4 or linear_rgb.shape[1] != 3:
            raise ValueError(f"Expected linear_rgb shape [B, 3, H, W], got {tuple(linear_rgb.shape)}")

        x = linear_rgb.clamp(0.0, 1.0)
        batch, channels, height, width = x.shape
        coeff = torch.tanh(self.coefficients(x)).view(batch, channels, self.order, height, width)
        coeff = coeff * self.coefficient_scale
        bases = polynomial_basis(x)

        if self.use_skip:
            delta = bases - x.unsqueeze(2)
            mapped = x + (coeff * delta).sum(dim=2)
        else:
            weights = torch.softmax(coeff, dim=2)
            mapped = (weights * bases).sum(dim=2)
        return mapped.clamp(0.0, 1.0)


def adapt_raw_input(raw: Tensor, input_format: str = "packed_bayer", pattern: str = "RGGB") -> Tensor:
    """Adapt RAW tensors to packed Bayer order ``[R, Gr, B, Gb]``."""

    fmt = input_format.lower()
    if fmt in {"packed_bayer", "bayer4", "rgbg"}:
        if raw.ndim != 4 or raw.shape[1] != 4:
            raise ValueError(f"Expected [B, 4, H, W] for {input_format}, got {tuple(raw.shape)}")
        return raw.float()

    if fmt in {"rgb_raw", "raw_rgb", "pseudo_rgb"}:
        if raw.ndim != 4 or raw.shape[1] != 3:
            raise ValueError(f"Expected [B, 3, H, W] for {input_format}, got {tuple(raw.shape)}")
        red, green, blue = raw[:, 0:1], raw[:, 1:2], raw[:, 2:3]
        return torch.cat((red, green, blue, green), dim=1).float()

    if fmt in {"mosaic", "bayer_mosaic", "single_bayer"}:
        if raw.ndim != 4 or raw.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, H, W] for {input_format}, got {tuple(raw.shape)}")
        height, width = raw.shape[-2:]
        if height % 2 or width % 2:
            raise ValueError(f"Bayer mosaic height and width must be even, got {(height, width)}")
        offsets = {
            "RGGB": ((0, 0), (0, 1), (1, 1), (1, 0)),
            "BGGR": ((1, 1), (1, 0), (0, 0), (0, 1)),
            "GRBG": ((0, 1), (0, 0), (1, 0), (1, 1)),
            "GBRG": ((1, 0), (1, 1), (0, 1), (0, 0)),
        }
        key = pattern.upper()
        if key not in offsets:
            raise ValueError(f"Unsupported Bayer pattern {pattern!r}; expected one of {sorted(offsets)}")
        planes = [raw[:, :, y::2, x::2] for y, x in offsets[key]]
        return torch.cat(planes, dim=1).float()

    raise ValueError(f"Unknown Dark-ISP input_format: {input_format!r}")


class DarkISP(nn.Module):
    """Lightweight Dark-ISP plugin producing RGB tensors for downstream models."""

    def __init__(
        self,
        hidden_dim: int = 48,
        initial_matrix: Tensor | None = None,
        max_linear_residual: float = 0.25,
        tone_coefficient_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.linear = DynamicLinearMapping(
            hidden_dim=hidden_dim,
            initial_matrix=initial_matrix,
            max_residual=max_linear_residual,
        )
        self.nonlinear = PolynomialToneMapping(
            hidden_dim=hidden_dim,
            coefficient_scale=tone_coefficient_scale,
            use_skip=True,
        )

    def forward(
        self,
        raw: Tensor,
        *,
        input_format: str = "packed_bayer",
        pattern: str = "RGGB",
    ) -> DarkISPOutput:
        raw_packed4 = adapt_raw_input(raw, input_format=input_format, pattern=pattern).clamp(0.0, 1.0)
        linear_rgb, matrix = self.linear(raw_packed4)
        rgb = self.nonlinear(linear_rgb)
        return DarkISPOutput(rgb=rgb, linear_rgb=linear_rgb, raw_packed4=raw_packed4, linear_matrix=matrix)


def self_boost_loss(
    raw_packed4: Tensor,
    nonlinear_rgb: Tensor,
    linear_matrix: Tensor,
    *,
    detach_nonlinear_target: bool = False,
    eps: float = 1.0e-6,
) -> Tensor:
    """Compute the Self-Boost regularization from Section 3.3.

    Args:
        raw_packed4: Bayer-packed input ``I`` with shape ``[B, 4, H, W]``.
        nonlinear_rgb: nonlinear output ``U`` with shape ``[B, 3, H, W]``.
        linear_matrix: adaptive ``P'`` as ``[B, 3, 4, H, W]`` or global
            ``[B, 3, 4]``.
        detach_nonlinear_target: If true, treats ``U`` as a fixed pseudo-target.
            The paper's wording allows gradients through ``U``; the default
            keeps that path open.
    """

    if raw_packed4.ndim != 4 or raw_packed4.shape[1] != 4:
        raise ValueError(f"raw_packed4 must have shape [B, 4, H, W], got {tuple(raw_packed4.shape)}")
    if nonlinear_rgb.ndim != 4 or nonlinear_rgb.shape[1] != 3:
        raise ValueError(f"nonlinear_rgb must have shape [B, 3, H, W], got {tuple(nonlinear_rgb.shape)}")

    raw_flat = raw_packed4.flatten(2)
    target = nonlinear_rgb.detach() if detach_nonlinear_target else nonlinear_rgb
    target_flat = target.flatten(2)

    gram = raw_flat @ raw_flat.transpose(1, 2)
    eye = torch.eye(4, device=gram.device, dtype=gram.dtype).unsqueeze(0)
    gram = gram + eps * eye
    pseudo_inv = torch.linalg.pinv(gram)
    p_tilde = (target_flat @ raw_flat.transpose(1, 2)) @ pseudo_inv

    if linear_matrix.ndim == 5:
        p_prime = linear_matrix.mean(dim=(-2, -1))
    elif linear_matrix.ndim == 3:
        p_prime = linear_matrix
    else:
        raise ValueError(
            "linear_matrix must have shape [B, 3, 4, H, W] or [B, 3, 4], "
            f"got {tuple(linear_matrix.shape)}"
        )

    cos = F.cosine_similarity(p_prime, p_tilde, dim=-1, eps=eps)
    return (1.0 - cos).abs().sum(dim=1).mean()
