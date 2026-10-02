"""Pure PyTorch RAM core module.

RAM applies multiple ISP functions in parallel to RAW-RGB input and fuses the
processed representations before a detector backbone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class RAMOutput:
    image: Tensor
    branches: dict[str, Tensor]
    params: dict[str, Tensor]


def conv_block(in_channels: int, out_channels: int, norm: str | None = "BN") -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, 3, padding="same")]
    if norm == "IN":
        layers.append(nn.InstanceNorm2d(out_channels))
    elif norm == "BN":
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm is not None:
        raise ValueError(f"Unsupported norm {norm!r}; expected 'BN', 'IN', or None.")
    layers.append(nn.LeakyReLU())
    return nn.Sequential(*layers)


class RPEncoder(nn.Module):
    """Raw Parameter Encoder from the official RAM implementation."""

    def __init__(self, img_size: int = 256, in_channels: int = 3, out_channels: int = 128) -> None:
        super().__init__()
        self.img_size = img_size
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, 16, 7, padding="same"),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, padding="same"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, out_channels, 3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear")
        return self.seq(x)


class RPDecoder(nn.Module):
    """Raw Parameter Decoder from the official RAM implementation."""

    def __init__(self, out_channels: int, in_channels: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(),
            nn.Linear(in_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


def define_feature_fusion(
    in_channels: int,
    out_channels: int = 3,
    ffm_params: dict | None = None,
) -> nn.Sequential:
    """Build the official feature-fusion module variants.

    Default is `BN_HG`, the reverse-hourglass fusion used in the released RAM
    configs.
    """

    params = dict(ffm_params or {})
    ffm_type = params.pop("ffm_type", "BN_HG")
    mid_channels = params.pop("mid_channels", 64)
    if params:
        raise ValueError(f"Unknown feature-fusion params: {sorted(params)}")

    if ffm_type == "IN":
        return nn.Sequential(
            conv_block(in_channels, 16, "IN"),
            conv_block(16, mid_channels, "IN"),
            nn.Conv2d(mid_channels, out_channels, 1, padding="same"),
        )
    if ffm_type == "HG":
        return nn.Sequential(
            conv_block(in_channels, 16, None),
            conv_block(16, mid_channels, None),
            conv_block(mid_channels, 16, None),
            nn.Conv2d(16, out_channels, 1, padding="same"),
        )
    if ffm_type == "IN_HG":
        return nn.Sequential(
            conv_block(in_channels, 16, "IN"),
            conv_block(16, mid_channels, "IN"),
            conv_block(mid_channels, 16, "IN"),
            nn.Conv2d(16, out_channels, 1, padding="same"),
        )
    if ffm_type == "BN_HG":
        return nn.Sequential(
            conv_block(in_channels, 16, "BN"),
            conv_block(16, mid_channels, "BN"),
            conv_block(mid_channels, 16, "BN"),
            nn.Conv2d(16, out_channels, 1, padding="same"),
        )
    if ffm_type == "BN":
        return nn.Sequential(
            conv_block(in_channels, 16, "BN"),
            conv_block(16, mid_channels, "BN"),
            nn.Conv2d(mid_channels, out_channels, 1, padding="same"),
        )
    raise ValueError(f"Unsupported ffm_type {ffm_type!r}.")


def rggb_to_rgb(raw: Tensor) -> Tensor:
    """Convert official RGGB-packed RAW `[R, Gr, Gb, B]` to RAW-RGB."""

    if raw.ndim != 4 or raw.shape[1] != 4:
        raise ValueError(f"Expected [B, 4, H, W] RGGB-packed RAW, got {tuple(raw.shape)}")
    red = raw[:, 0:1]
    green = 0.5 * (raw[:, 1:2] + raw[:, 2:3])
    blue = raw[:, 3:4]
    return torch.cat((red, green, blue), dim=1)


def adapt_ram_input(raw: Tensor, input_format: str = "rgb_raw") -> Tensor:
    """Adapt inputs to official RAM's default three-channel RAW-RGB tensor."""

    fmt = input_format.lower()
    if fmt in {"rgb_raw", "raw_rgb", "pseudo_rgb", "rgb"}:
        if raw.ndim != 4 or raw.shape[1] != 3:
            raise ValueError(f"Expected [B, 3, H, W] for {input_format}, got {tuple(raw.shape)}")
        return raw.float()
    if fmt in {"packed_bayer", "rggb", "rggb4"}:
        return rggb_to_rgb(raw.float())
    if fmt in {"darkisp_packed", "darkisp_bayer4", "rgbg"}:
        if raw.ndim != 4 or raw.shape[1] != 4:
            raise ValueError(f"Expected [B, 4, H, W] for {input_format}, got {tuple(raw.shape)}")
        red = raw[:, 0:1]
        green = 0.5 * (raw[:, 1:2] + raw[:, 3:4])
        blue = raw[:, 2:3]
        return torch.cat((red, green, blue), dim=1).float()
    raise ValueError(f"Unknown RAM input_format: {input_format!r}")


class RawAdaptationModule(nn.Module):
    """Official RAM core with pure PyTorch dependencies.

    The official module defaults to `in_channels=3`, because its data pipeline
    converts RGGB-packed RAW into `[R, mean(G), B]` before RAM.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 256,
        out_channels: int = 128,
        functions: list[str] | tuple[str, ...] = ("wb", "ccm", "gamma", "brightness"),
        ffm_params: dict | None = None,
        clamp_values: bool = False,
        output_channels: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.functions = list(functions)
        self.encoder = RPEncoder(img_size=img_size, in_channels=in_channels, out_channels=out_channels)
        self.output_channels = output_channels

        for function in self.functions:
            self.define_function(function, rpe_out_channels=out_channels, input_channels=in_channels)

        ffm_in_channels = in_channels * max(len(self.functions), 1)
        self.ffm = define_feature_fusion(in_channels=ffm_in_channels, out_channels=output_channels, ffm_params=ffm_params)
        self.norm_layer = nn.BatchNorm2d(output_channels, affine=True)
        self.clamp_values = clamp_values

    def define_function(self, function: str, rpe_out_channels: int, input_channels: int) -> None:
        if function == "gamma":
            self.gamma = nn.Sequential(RPDecoder(in_channels=rpe_out_channels, out_channels=1), nn.Sigmoid())
        elif function == "ccm":
            self.conv_cc = RPDecoder(in_channels=rpe_out_channels, out_channels=input_channels**2)
        elif function == "wb":
            self.conv_wb = RPDecoder(in_channels=rpe_out_channels, out_channels=input_channels)
        elif function == "brightness":
            self.conv_bright = nn.Sequential(RPDecoder(in_channels=rpe_out_channels, out_channels=1), nn.Sigmoid())
        else:
            raise ValueError(f"Unsupported RAM ISP function {function!r}.")

    def apply_function(self, original: Tensor, function: str, encoded: Tensor) -> tuple[Tensor, Tensor]:
        batch = encoded.size(0)
        if function == "gamma":
            gamma = self.gamma(encoded).view(-1, 1, 1, 1)
            return original.clamp_min(0.0) ** gamma, gamma
        if function == "ccm":
            params = self.conv_cc(encoded).reshape(batch, self.in_channels, self.in_channels)
            return torch.einsum("bcij,bnc->bnij", original, params), params
        if function == "wb":
            params = self.conv_wb(encoded).reshape(batch, self.in_channels, 1, 1)
            return original * params, params
        if function == "brightness":
            params = self.conv_bright(encoded).view(-1, 1, 1, 1)
            return original + params, params
        raise ValueError(f"Unsupported RAM ISP function {function!r}.")

    def forward(
        self,
        raw: Tensor,
        *,
        input_format: str = "rgb_raw",
        return_details: bool = False,
        training: bool | None = None,
    ) -> Tensor | RAMOutput:
        if not self.functions:
            raise AssertionError("functions list is empty")

        x = adapt_ram_input(raw, input_format=input_format)
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"RAM was built with in_channels={self.in_channels}, but adapted input has {x.shape[1]} channels."
            )
        if self.clamp_values:
            x = x.clamp_min(0.0)

        encoded = self.encoder(x)
        branches: dict[str, Tensor] = {}
        params: dict[str, Tensor] = {}
        outputs = []
        for function in self.functions:
            branch, param = self.apply_function(x, function, encoded)
            branches[function] = branch
            params[function] = param
            outputs.append(branch)

        fused = self.ffm(torch.cat(outputs, dim=1))
        image = self.norm_layer(fused)
        if return_details:
            return RAMOutput(image=image, branches=branches, params=params)
        return image


RAM = RawAdaptationModule
