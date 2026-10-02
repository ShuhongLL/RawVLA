"""Pure PyTorch RAWild adapter core."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class RAWildOutput:
    rgb: Tensor
    grid: Tensor | None
    curve_delta: Tensor | None
    control_points: Tensor | None


class RGBuvHistogram(nn.Module):
    """Differentiable RGB-uv histogram used by RAWild conditioning."""

    def __init__(self, num_bins: int = 64, sigma: float = 0.1, val_range: tuple[float, float] = (-3.0, 3.0), downsample_factor: int = 16):
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        self.val_range = val_range
        self.pool = nn.AvgPool2d(downsample_factor, downsample_factor) if downsample_factor > 1 else nn.Identity()
        self.register_buffer("bin_centers", torch.linspace(val_range[0], val_range[1], num_bins), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        eps = 1.0e-6
        flat = self.pool(x).view(batch, 3, -1)
        r = flat[:, 0:1] + eps
        g = flat[:, 1:2] + eps
        b = flat[:, 2:3] + eps
        u = torch.log(r / g)
        v = torch.log(b / g)
        centers = self.bin_centers.to(device=x.device, dtype=x.dtype).view(1, self.num_bins, 1)
        u_weights = torch.exp(-0.5 * ((u - centers) / self.sigma) ** 2)
        v_weights = torch.exp(-0.5 * ((v - centers) / self.sigma) ** 2)
        hist = torch.bmm(u_weights, v_weights.transpose(1, 2))
        hist = hist / hist.sum(dim=(1, 2), keepdim=True).clamp_min(eps)
        return hist.unsqueeze(1)


class ConditionalTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0, cond_injection_type: str = "none"):
        super().__init__()
        self.cond_injection_type = cond_injection_type
        self.cond_token_dim = dim * 2
        self.bit_cond_dim = dim
        affine_norm = cond_injection_type not in {"adaln", "hist_prefix_bit_adaln"}
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=affine_norm)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=affine_norm if cond_injection_type in {"adaln", "hist_prefix_bit_adaln"} else True)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden_dim, dim), nn.Dropout(drop))
        self.cross_norm = self.cond_norm = self.cross_attn = None
        self.adaln_proj = self.film_proj = self.bias_proj = None
        if cond_injection_type == "cross_attn":
            self.cross_norm = nn.LayerNorm(dim)
            self.cond_norm = nn.LayerNorm(dim)
            self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
            nn.init.zeros_(self.cross_attn.out_proj.weight)
            nn.init.zeros_(self.cross_attn.out_proj.bias)
        elif cond_injection_type in {"adaln", "hist_prefix_bit_adaln"}:
            adaln_in_dim = self.cond_token_dim if cond_injection_type == "adaln" else self.bit_cond_dim
            self.adaln_proj = nn.Linear(adaln_in_dim, dim * 6)
            nn.init.zeros_(self.adaln_proj.weight)
            nn.init.zeros_(self.adaln_proj.bias)
        elif cond_injection_type == "film":
            self.film_proj = nn.Linear(self.cond_token_dim, dim * 2)
            nn.init.zeros_(self.film_proj.weight)
            nn.init.zeros_(self.film_proj.bias)
        elif cond_injection_type == "add_bias":
            self.bias_proj = nn.Linear(self.cond_token_dim, dim)
            nn.init.zeros_(self.bias_proj.weight)
            nn.init.zeros_(self.bias_proj.bias)

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    @staticmethod
    def _flatten_cond_tokens(cond_tokens: Tensor | None) -> Tensor:
        if cond_tokens is None:
            raise ValueError("Conditional RAWild block requires cond_tokens.")
        return cond_tokens.flatten(1)

    def forward(self, x: Tensor, cond_tokens: Tensor | None = None, bit_cond: Tensor | None = None) -> Tensor:
        if self.cond_injection_type == "none":
            x2 = self.norm1(x)
            x = x + self.attn(x2, x2, x2, need_weights=False)[0]
            return x + self.mlp(self.norm2(x))

        if self.cond_injection_type in {"adaln", "hist_prefix_bit_adaln"}:
            if self.cond_injection_type == "adaln":
                cond_input = self._flatten_cond_tokens(cond_tokens)
            else:
                if bit_cond is None:
                    raise ValueError("hist_prefix_bit_adaln requires bit_cond.")
                cond_input = bit_cond
            shift1, scale1, gate_attn, shift2, scale2, gate_mlp = self.adaln_proj(cond_input).chunk(6, dim=-1)
            x1 = self._modulate(self.norm1(x), shift1, scale1)
            x = x + torch.tanh(gate_attn).unsqueeze(1) * self.attn(x1, x1, x1, need_weights=False)[0]
            x2 = self._modulate(self.norm2(x), shift2, scale2)
            return x + torch.tanh(gate_mlp).unsqueeze(1) * self.mlp(x2)

        if self.cond_injection_type == "cross_attn":
            if cond_tokens is None:
                raise ValueError("cross_attn requires cond_tokens.")
            x1 = self.norm1(x)
            x = x + self.attn(x1, x1, x1, need_weights=False)[0]
            x = x + self.cross_attn(self.cross_norm(x), self.cond_norm(cond_tokens), self.cond_norm(cond_tokens), need_weights=False)[0]
            return x + self.mlp(self.norm2(x))

        if self.cond_injection_type == "prefix":
            if cond_tokens is None:
                raise ValueError("prefix requires cond_tokens.")
            joint = torch.cat([cond_tokens, x], dim=1)
            joint2 = self.norm1(joint)
            joint = joint + self.attn(joint2, joint2, joint2, need_weights=False)[0]
            joint = joint + self.mlp(self.norm2(joint))
            return joint[:, cond_tokens.shape[1]:]

        if self.cond_injection_type == "film":
            gamma, beta = self.film_proj(self._flatten_cond_tokens(cond_tokens)).chunk(2, dim=-1)
            x1 = self.norm1(x)
            x = x + self.attn(x1, x1, x1, need_weights=False)[0]
            x = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            return x + self.mlp(self.norm2(x))

        if self.cond_injection_type == "add_bias":
            bias = self.bias_proj(self._flatten_cond_tokens(cond_tokens)).unsqueeze(1)
            x1 = self.norm1(x + bias)
            x = x + self.attn(x1, x1, x1, need_weights=False)[0]
            return x + self.mlp(self.norm2(x))

        raise ValueError(f"Unsupported cond_injection_type: {self.cond_injection_type}")


class BilateralGridAdapterDIA(nn.Module):
    """RAWild Bezier curve plus DIA bilateral-grid adapter."""

    def __init__(
        self,
        grid_depth: int = 8,
        transformer_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        curve_n: int = 6,
        use_bezier: bool = True,
        use_grid: bool = True,
        bezier_type: str = "init_bias",
        dia_k: float = 0.05,
        dia_activation: str = "exp_tanh",
        dia_exp_alpha: float = 1.0,
        dia_matrix_mode: str = "d_ia",
        cond_injection_type: str = "adaln",
        hist_bins: int = 64,
        hist_sigma: float = 0.1,
        hist_downsample_factor: int = 16,
        hist_prefix_tokens: int = 8,
        bit_embed_min: int = 8,
        bit_embed_max: int = 24,
        bit_depth_norm_max: float = 24.0,
        hist_condition_mode: str = "quantile",
    ) -> None:
        super().__init__()
        self.grid_depth = grid_depth
        self.transformer_dim = transformer_dim
        self.use_bezier = use_bezier
        self.use_grid = use_grid
        self.bezier_type = bezier_type
        self.curve_n = curve_n
        self.dia_k = dia_k
        self.dia_activation = dia_activation
        self.dia_exp_alpha = dia_exp_alpha
        self.dia_matrix_mode = dia_matrix_mode
        self.cond_injection_type = cond_injection_type
        self.hist_bins = int(hist_bins)
        self.hist_sigma = float(hist_sigma)
        self.hist_downsample_factor = int(hist_downsample_factor)
        self.hist_prefix_tokens = int(hist_prefix_tokens)
        self.bit_embed_min = int(bit_embed_min)
        self.bit_embed_max = int(bit_embed_max)
        self.bit_depth_norm_max = float(bit_depth_norm_max)
        self.hist_condition_mode = hist_condition_mode
        self.use_cond_injection = cond_injection_type != "none"
        self.use_hist_prefix_bit_adaln = cond_injection_type == "hist_prefix_bit_adaln"
        self.C_grid = 9
        self.C_curve = curve_n - 1 if use_bezier and bezier_type == "init_bias" else (curve_n if use_bezier else 0)
        self.register_buffer("hist_quantiles", torch.arange(1, hist_bins + 1, dtype=torch.float32) / float(hist_bins), persistent=False)
        self.register_buffer("hist_value_grid", torch.linspace(0.0, 1.0, steps=hist_bins, dtype=torch.float32), persistent=False)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, transformer_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(transformer_dim),
            nn.GELU(),
        )
        self.transformer_blocks = nn.ModuleList(
            [ConditionalTransformerBlock(transformer_dim, num_heads, mlp_ratio, drop_rate, cond_injection_type) for _ in range(num_layers)]
        )
        self.norm_out = nn.LayerNorm(transformer_dim)
        self.grid_head = nn.Conv2d(transformer_dim, grid_depth * self.C_grid, 1)
        self.curve_head = nn.Linear(transformer_dim, 3 * self.C_curve) if use_bezier else None

        self.histogram = self.hist_encoder = self.hist_token_encoder = self.bit_encoder = None
        if self.use_cond_injection:
            if self.use_hist_prefix_bit_adaln:
                self.histogram = RGBuvHistogram(hist_bins, hist_sigma, downsample_factor=hist_downsample_factor)
                self.hist_encoder = nn.Sequential(
                    nn.Conv2d(1, transformer_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(transformer_dim),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d((hist_prefix_tokens, 1)),
                )
                self.bit_encoder = nn.Embedding(bit_embed_max - bit_embed_min + 1, transformer_dim)
            else:
                if hist_condition_mode == "legacy_conv":
                    self.histogram = RGBuvHistogram(hist_bins, hist_sigma, downsample_factor=hist_downsample_factor)
                    self.hist_encoder = nn.Sequential(
                        nn.Conv2d(1, transformer_dim, 3, padding=1, bias=False),
                        nn.BatchNorm2d(transformer_dim),
                        nn.GELU(),
                        nn.AdaptiveAvgPool2d(1),
                    )
                else:
                    self.hist_token_encoder = nn.Sequential(nn.Linear(3 * hist_bins, transformer_dim), nn.GELU(), nn.Linear(transformer_dim, transformer_dim))
                self.bit_encoder = nn.Sequential(nn.Linear(1, transformer_dim), nn.GELU(), nn.Linear(transformer_dim, transformer_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.grid_head.weight)
        nn.init.zeros_(self.grid_head.bias)
        if self.curve_head is not None:
            nn.init.zeros_(self.curve_head.weight)
            if self.bezier_type == "init_bias":
                nn.init.zeros_(self.curve_head.bias)
            else:
                nn.init.ones_(self.curve_head.bias)

    def _reconstruct_dia(self, raw_grid: Tensor) -> Tensor:
        batch, _, gd, gh, gw = raw_grid.shape
        d_hat = raw_grid[:, 0:3]
        a_hat = raw_grid[:, 3:9]
        d = torch.exp(self.dia_exp_alpha * torch.tanh(d_hat)) if self.dia_activation == "exp_tanh" else F.softplus(d_hat)
        a = self.dia_k * torch.tanh(a_hat)
        matrix = torch.zeros(batch, 3, 3, gd, gh, gw, device=raw_grid.device, dtype=raw_grid.dtype)
        if self.dia_matrix_mode in {"d_ia", "d_only", "full_affine"}:
            matrix[:, 0, 0] = d[:, 0]
            matrix[:, 1, 1] = d[:, 1]
            matrix[:, 2, 2] = d[:, 2]
        else:
            matrix[:, 0, 0] = 1.0
            matrix[:, 1, 1] = 1.0
            matrix[:, 2, 2] = 1.0
        if self.dia_matrix_mode in {"d_ia", "ia_only", "full_affine"}:
            if self.dia_matrix_mode == "d_ia":
                ip_a = torch.zeros_like(matrix)
                ip_a[:, 0, 0] = 1.0
                ip_a[:, 1, 1] = 1.0
                ip_a[:, 2, 2] = 1.0
                ip_a[:, 0, 1] = a[:, 0]
                ip_a[:, 0, 2] = a[:, 1]
                ip_a[:, 1, 0] = a[:, 2]
                ip_a[:, 1, 2] = a[:, 3]
                ip_a[:, 2, 0] = a[:, 4]
                ip_a[:, 2, 1] = a[:, 5]
                matrix = d.unsqueeze(2) * ip_a
            else:
                matrix[:, 0, 1] = a[:, 0]
                matrix[:, 0, 2] = a[:, 1]
                matrix[:, 1, 0] = a[:, 2]
                matrix[:, 1, 2] = a[:, 3]
                matrix[:, 2, 0] = a[:, 4]
                matrix[:, 2, 1] = a[:, 5]
        return matrix.reshape(batch, 9, gd, gh, gw)

    def _get_pos_embed(self, height: int, width: int, device: torch.device) -> Tensor:
        dim = self.transformer_dim
        half = dim // 2
        y_pos = torch.arange(height, device=device, dtype=torch.float32).unsqueeze(1).expand(height, width).reshape(-1)
        x_pos = torch.arange(width, device=device, dtype=torch.float32).unsqueeze(0).expand(height, width).reshape(-1)
        dim_t = 10000 ** (2 * torch.arange(half // 2, device=device, dtype=torch.float32) / half)
        pos_embed = torch.zeros(height * width, dim, device=device)
        pos_embed[:, 0:half:2] = torch.sin(y_pos.unsqueeze(1) / dim_t.unsqueeze(0))
        pos_embed[:, 1:half:2] = torch.cos(y_pos.unsqueeze(1) / dim_t.unsqueeze(0))
        pos_embed[:, half::2] = torch.sin(x_pos.unsqueeze(1) / dim_t.unsqueeze(0))
        pos_embed[:, half + 1::2] = torch.cos(x_pos.unsqueeze(1) / dim_t.unsqueeze(0))
        return pos_embed.unsqueeze(0)

    def _downsample_hist_input(self, x: Tensor) -> Tensor:
        if self.hist_downsample_factor <= 1:
            return x
        height, width = x.shape[-2:]
        return F.adaptive_avg_pool2d(x, (max(1, math.ceil(height / self.hist_downsample_factor)), max(1, math.ceil(width / self.hist_downsample_factor))))

    def _build_soft_quantile_hist_token(self, x_8bit: Tensor) -> Tensor:
        samples = self._downsample_hist_input(x_8bit.clamp(0.0, 1.0)).flatten(2)
        value_grid = self.hist_value_grid.to(dtype=samples.dtype, device=samples.device)
        quantile_levels = self.hist_quantiles.to(dtype=samples.dtype, device=samples.device)
        values = value_grid.view(1, 1, -1, 1)
        cdf = torch.sigmoid((values - samples.unsqueeze(2)) / self.hist_sigma).mean(dim=-1)
        cdf_distance = torch.abs(cdf.unsqueeze(2) - quantile_levels.view(1, 1, -1, 1))
        inverse_temp = max(1.0 / float(self.hist_bins), self.hist_sigma * 0.5)
        weights = torch.softmax(-cdf_distance / inverse_temp, dim=-1)
        quantiles = (weights * value_grid.view(1, 1, 1, -1)).sum(dim=-1)
        return self.hist_token_encoder(quantiles.reshape(x_8bit.shape[0], -1))

    def _build_condition_inputs(self, x_8bit: Tensor, raw_bit_depth: Tensor | int | float | None) -> tuple[Tensor, Tensor | None] | Tensor:
        if raw_bit_depth is None:
            raise ValueError("cond_injection_type is enabled but raw_bit_depth is missing.")
        if not torch.is_tensor(raw_bit_depth):
            raw_bit_depth = torch.full((x_8bit.shape[0], 1), float(raw_bit_depth), device=x_8bit.device, dtype=x_8bit.dtype)
        raw_bit_depth = raw_bit_depth.to(device=x_8bit.device, dtype=x_8bit.dtype).view(x_8bit.shape[0], 1)
        if self.use_hist_prefix_bit_adaln:
            bit_idx = raw_bit_depth.long().view(-1).clamp(self.bit_embed_min, self.bit_embed_max) - self.bit_embed_min
            bit_cond = self.bit_encoder(bit_idx)
            hist = self.histogram(x_8bit)
            hist_prefix = self.hist_encoder(hist).squeeze(-1).transpose(1, 2).contiguous()
            return hist_prefix, bit_cond
        bit_token = self.bit_encoder(raw_bit_depth / self.bit_depth_norm_max)
        if self.hist_condition_mode == "legacy_conv":
            hist_token = self.hist_encoder(self.histogram(x_8bit)).flatten(1)
        else:
            hist_token = self._build_soft_quantile_hist_token(x_8bit)
        return torch.stack([bit_token, hist_token], dim=1)

    def predict_grid(self, x_8bit: Tensor, raw_bit_depth: Tensor | int | float | None = None) -> tuple[Tensor | None, Tensor | None, tuple[int, int]]:
        batch = x_8bit.shape[0]
        feat = self.conv_stem(x_8bit)
        gh, gw = feat.shape[2:]
        tokens = feat.flatten(2).transpose(1, 2) + self._get_pos_embed(gh, gw, feat.device)
        cond_tokens = bit_cond = None
        if self.use_cond_injection:
            cond_inputs = self._build_condition_inputs(x_8bit, raw_bit_depth)
            if self.use_hist_prefix_bit_adaln:
                hist_prefix, bit_cond = cond_inputs
                tokens = torch.cat([hist_prefix, tokens], dim=1)
            else:
                cond_tokens = cond_inputs
        for block in self.transformer_blocks:
            tokens = block(tokens, cond_tokens=cond_tokens, bit_cond=bit_cond)
        tokens = self.norm_out(tokens)
        grid_tokens = tokens[:, self.hist_prefix_tokens:] if self.use_hist_prefix_bit_adaln else tokens
        delta = self.curve_head(grid_tokens.mean(dim=1)) if self.curve_head is not None else None
        feat = grid_tokens.transpose(1, 2).reshape(batch, self.transformer_dim, gh, gw)
        if not self.use_grid:
            return None, delta, (gh, gw)
        matrix_grid = self.grid_head(feat).reshape(batch, self.grid_depth, self.C_grid, gh, gw).permute(0, 2, 1, 3, 4)
        return self._reconstruct_dia(matrix_grid), delta, (gh, gw)

    def build_control_points(self, delta: Tensor) -> Tensor:
        batch = delta.shape[0]
        n = self.curve_n
        if self.bezier_type == "init_bias":
            delta = delta.view(batch, 3, n - 1)
            init_interior = torch.linspace(0, 1, steps=n + 1, device=delta.device, dtype=delta.dtype)[1:-1].view(1, 1, n - 1)
            p_interior = torch.clamp(init_interior + torch.tanh(delta) * 0.5, 0.0, 1.0)
            p0 = torch.zeros(batch, 3, 1, device=delta.device, dtype=delta.dtype)
            pn = torch.ones(batch, 3, 1, device=delta.device, dtype=delta.dtype)
            return torch.cat([p0, p_interior, pn], dim=2)
        delta = F.relu(delta.view(batch, 3, n)) + 1.0e-6
        p_rest = torch.cumsum(delta / delta.sum(dim=2, keepdim=True), dim=2)
        p0 = torch.zeros(batch, 3, 1, device=delta.device, dtype=delta.dtype)
        return torch.cat([p0, p_rest], dim=2)

    def bezier_map(self, raw: Tensor, control_points: Tensor) -> Tensor:
        mapped = torch.zeros_like(raw)
        for i in range(self.curve_n + 1):
            term = math.comb(self.curve_n, i) * ((1 - raw) ** (self.curve_n - i)) * (raw ** i)
            mapped = mapped + term * control_points[:, :, i].unsqueeze(-1).unsqueeze(-1)
        return mapped.clamp(0.0, 1.0)

    def slice_grid(self, grid: Tensor, guide_map: Tensor, full_height: int, full_width: int) -> Tensor:
        batch = grid.shape[0]
        gy = torch.linspace(-1, 1, full_height, device=grid.device)
        gx = torch.linspace(-1, 1, full_width, device=grid.device)
        gy, gx = torch.meshgrid(gy, gx, indexing="ij")
        gy = gy.unsqueeze(0).expand(batch, -1, -1)
        gx = gx.unsqueeze(0).expand(batch, -1, -1)
        gz = guide_map.squeeze(1) * 2 - 1
        sampling_grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(1)
        return F.grid_sample(grid, sampling_grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze(2)

    @staticmethod
    def apply_matrix(coeffs: Tensor, x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        matrix = coeffs.reshape(batch, 3, 3, height, width)
        return torch.einsum("bijhw,bjhw->bihw", matrix, x).clamp(0.0, 1.0)

    def forward(self, x_8bit: Tensor, x_12bit: Tensor, raw_bit_depth: Tensor | int | float | None = None, *, return_details: bool = False) -> Tensor | RAWildOutput:
        _, _, height, width = x_8bit.shape
        grid, delta, _ = self.predict_grid(x_8bit, raw_bit_depth=raw_bit_depth)
        control_points = self.build_control_points(delta) if self.use_bezier else None
        x_mapped = self.bezier_map(x_12bit, control_points) if control_points is not None else x_12bit
        rgb = x_mapped
        if self.use_grid:
            guide_map = x_12bit.mean(dim=1, keepdim=True)
            coeffs = self.slice_grid(grid, guide_map, height, width)
            rgb = self.apply_matrix(coeffs, x_mapped)
        if return_details:
            return RAWildOutput(rgb=rgb, grid=grid, curve_delta=delta, control_points=control_points)
        return rgb


def adapt_rawild_input(raw: Tensor, input_format: str = "six_channel") -> tuple[Tensor, Tensor]:
    fmt = input_format.lower()
    if fmt in {"six_channel", "rawild6", "guide_apply"}:
        if raw.ndim != 4 or raw.shape[1] != 6:
            raise ValueError(f"Expected [B, 6, H, W] for {input_format}, got {tuple(raw.shape)}")
        return raw[:, :3].float(), raw[:, 3:].float()
    if fmt in {"rgb_raw", "raw_rgb", "pseudo_rgb", "rgb"}:
        if raw.ndim != 4 or raw.shape[1] != 3:
            raise ValueError(f"Expected [B, 3, H, W] for {input_format}, got {tuple(raw.shape)}")
        x = raw.float()
        return x, x
    if fmt in {"packed_bayer", "bayer4", "rgbg"}:
        if raw.ndim != 4 or raw.shape[1] != 4:
            raise ValueError(f"Expected [B, 4, H, W] for {input_format}, got {tuple(raw.shape)}")
        red, green_r, blue, green_b = raw[:, 0:1], raw[:, 1:2], raw[:, 2:3], raw[:, 3:4]
        x = torch.cat((red, 0.5 * (green_r + green_b), blue), dim=1).float()
        return x, x
    raise ValueError(f"Unknown RAWild input_format: {input_format!r}")


class RAWildAdapter(BilateralGridAdapterDIA):
    def forward(self, raw: Tensor, *, input_format: str = "six_channel", raw_bit_depth: Tensor | int | float | None = 10, return_details: bool = False) -> Tensor | RAWildOutput:
        guide, apply = adapt_rawild_input(raw, input_format=input_format)
        return super().forward(guide.clamp(0.0, 1.0), apply.clamp(0.0, 1.0), raw_bit_depth=raw_bit_depth, return_details=return_details)


BilateralGridAdapter_DIA = BilateralGridAdapterDIA
