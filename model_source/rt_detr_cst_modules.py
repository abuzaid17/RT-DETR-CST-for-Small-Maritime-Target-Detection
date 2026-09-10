"""
RT-DETR-CST: Custom PyTorch Modules
=====================================
Paper: "A Lightweight Infrared Remote Sensing Architecture for Enhanced Small Target Detection"

This file implements three novel modules from the paper:
  1. TCN  — TetraCore Network (Haar-like wavelet feature decomposition)
  2. SWN  — SynapticWeave Network (lightweight backbone)
  3. CFAN — Cross-Feature Attention Fusion Network (dual-branch SE fusion)

Author note: All modules are self-contained nn.Module subclasses designed
to slot directly into the Ultralytics RT-DETR codebase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath   # pip install timm


# ============================================================
# Shared utility: ConvBN (Conv2d + BatchNorm2d + optional act)
# ============================================================

class ConvBN(nn.Module):
    """
    Conv2d → BatchNorm2d → (optional) Activation.
    This helper is reused in both SWN blocks and CFAN.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,          # BN already has a learnable bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ============================================================
# MODULE 1 — TCN: TetraCore Network
# ============================================================

class TCN(nn.Module):
    """
    TetraCore Network — Haar-like 2-D wavelet decomposition.

    Given input  X ∈ R^{B × C × H × W}, the module computes four
    directional sub-bands (LL, LH, HL, HH) using the 2×2 Haar kernel,
    concatenates them into Y ∈ R^{B × 4C × H/2 × W/2}, then fuses back
    to C channels with a learned 1×1 convolution.

    The four sub-bands are:
        A(i,j) = ½[X(2i,2j) + X(2i,2j+1) + X(2i+1,2j) + X(2i+1,2j+1)]  # LL (low-low)
        B(i,j) = ½[X(2i,2j) + X(2i,2j+1) - X(2i+1,2j) - X(2i+1,2j+1)]  # LH (low-high)
        C(i,j) = ½[X(2i,2j) - X(2i,2j+1) + X(2i+1,2j) - X(2i+1,2j+1)]  # HL (high-low)
        D(i,j) = ½[X(2i,2j) - X(2i,2j+1) - X(2i+1,2j) + X(2i+1,2j+1)]  # HH (high-high)

    Args:
        in_channels (int): Number of input channels C.

    Input:   (B, C, H, W)   — H and W must be even
    Output:  (B, C, H/2, W/2)
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        # ---------- Fixed (non-trainable) Haar filter bank ----------
        # Each filter is stored as a Conv2d with frozen weights.
        # Weight shape: (out_ch, in_ch/groups, kH, kW)
        # We use groups=in_channels so every channel is filtered independently.
        haar_kernels = torch.tensor(
            [
                [[ 1,  1], [ 1,  1]],   # A — LL
                [[ 1,  1], [-1, -1]],   # B — LH
                [[ 1, -1], [ 1, -1]],   # C — HL
                [[ 1, -1], [-1,  1]],   # D — HH
            ],
            dtype=torch.float32
        ) * 0.5   # ½ factor from the paper

        # haar_kernels shape: (4, 1, 2, 2) — one kernel per sub-band
        # Replicate for each input channel → (4*C, 1, 2, 2) then reshape
        # to (4*C, 1, 2, 2) used with groups=C.
        # We build 4 separate depthwise convolutions for clarity.
        self._make_haar_convs(haar_kernels, in_channels)

        # ---------- Learnable 1×1 fusion: 4C → C ----------
        self.fuse = nn.Conv2d(4 * in_channels, in_channels, kernel_size=1, bias=True)

    def _make_haar_convs(self, kernels: torch.Tensor, C: int):
        """Register four frozen depthwise Conv2d layers for sub-bands A–D."""
        for idx, name in enumerate(['conv_A', 'conv_B', 'conv_C', 'conv_D']):
            # kernel shape required by Conv2d: (out_ch, in_ch/groups, kH, kW)
            # depthwise: groups=C, so in_ch/groups=1, out_ch=C
            weight = kernels[idx].unsqueeze(0).unsqueeze(0)  # (1, 1, 2, 2)
            weight = weight.repeat(C, 1, 1, 1)               # (C, 1, 2, 2)

            conv = nn.Conv2d(
                C, C,
                kernel_size=2,
                stride=2,          # halves spatial dims
                padding=0,
                groups=C,          # depthwise
                bias=False,
            )
            with torch.no_grad():
                conv.weight.copy_(weight)

            # Freeze — these are fixed mathematical filters
            for p in conv.parameters():
                p.requires_grad_(False)

            self.add_module(name, conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, C, H, W)  — H, W must be even.

        Returns:
            z: (B, C, H/2, W/2)
        """
        # 1. Compute the four Haar sub-bands — each (B, C, H/2, W/2)
        A = self.conv_A(x)
        B = self.conv_B(x)
        C = self.conv_C(x)
        D = self.conv_D(x)

        # 2. Concatenate along channel dim → (B, 4C, H/2, W/2)
        y = torch.cat([A, B, C, D], dim=1)

        # 3. Fuse with learnable linear weighting: W*Y + b → (B, C, H/2, W/2)
        z = self.fuse(y)
        return z


# ============================================================
# MODULE 2 — SWN: SynapticWeave Network (Backbone)
# ============================================================

class SWNBlock(nn.Module):
    """
    A single SWN block with depthwise separable convolutions and stochastic depth.

    Block equation (from paper):
        X_out = X + DropPath( Conv_post( σ(Conv_f1(X_dw)) ⊙ Conv_f2(X_dw) ) )

    where X_dw is the depthwise-conv output of X, σ is SiLU, and ⊙ is
    element-wise multiplication (gated linear unit style).

    Args:
        channels (int):        Number of input/output channels (block is channel-preserving).
        drop_path_rate (float): Stochastic depth probability (0 = disabled).
        expand_ratio (int):    Hidden-dimension multiplier for the gating branches.
    """

    def __init__(
        self,
        channels: int,
        drop_path_rate: float = 0.0,
        expand_ratio: int = 2,
    ):
        super().__init__()
        hidden = channels * expand_ratio

        # ---------- Depthwise 3×3 conv (spatial mixing) ----------
        self.dw_conv = ConvBN(channels, channels, kernel_size=3, stride=1,
                              padding=1, groups=channels, act=False)

        # ---------- Gating branches (pointwise / 1×1) ----------
        # Conv_f1: activated branch
        self.conv_f1 = ConvBN(channels, hidden, kernel_size=1, act=True)
        # Conv_f2: gate branch (no activation — raw gate values)
        self.conv_f2 = ConvBN(channels, hidden, kernel_size=1, act=False)

        # ---------- Post projection: hidden → channels ----------
        self.conv_post = ConvBN(hidden, channels, kernel_size=1, act=False)

        # ---------- Stochastic depth ----------
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Depthwise spatial mixing
        x_dw = self.dw_conv(x)

        # 2. Gated activation: σ(f1(x_dw)) ⊙ f2(x_dw)
        gate = torch.sigmoid(self.conv_f1(x_dw)) * self.conv_f2(x_dw)

        # 3. Post-projection + stochastic depth residual
        out = x + self.drop_path(self.conv_post(gate))
        return out


class SWNStage(nn.Module):
    """
    One SWN stage: optional stride-2 downsampling → N blocks.

    Args:
        in_channels  (int): Input channel count.
        out_channels (int): Output channel count.
        num_blocks   (int): Number of SWNBlocks in this stage.
        stride       (int): 2 for downsampling stages, 1 for first stage.
        drop_path_rates (list[float]): Per-block stochastic depth rates.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 2,
        drop_path_rates=None,
    ):
        super().__init__()
        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_blocks

        layers = []
        # Downsampling conv (stride-2) to transition channels/spatial dims
        if stride > 1 or in_channels != out_channels:
            layers.append(
                ConvBN(in_channels, out_channels, kernel_size=3,
                       stride=stride, padding=1, act=True)
            )

        # Stacked SWN blocks (all channel-preserving)
        for i in range(num_blocks):
            layers.append(SWNBlock(out_channels, drop_path_rate=drop_path_rates[i]))

        self.stage = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x)


class SWN(nn.Module):
    """
    SynapticWeave Network — lightweight backbone for RT-DETR-CST.

    Architecture:
        Stem  : 3×3 conv, stride 2, BN → 32 channels
        Stage1: stride 2, 32→64,   1 block
        Stage2: stride 2, 64→128,  2 blocks
        Stage3: stride 2, 128→256, 6 blocks
        Stage4: stride 2, 256→512, 2 blocks

    The three feature maps returned (from Stages 2, 3, 4) are sized:
        P3 : C=128,  H/8,  W/8
        P4 : C=256,  H/16, W/16
        P5 : C=512,  H/32, W/32
    These match the FPN/PAN input expectations of RT-DETR.

    Args:
        in_channels (int): Input image channels (default 3 for RGB / 1 for IR).
        drop_path_rate (float): Maximum stochastic depth rate (linearly scaled).

    Input:   (B, in_channels, H, W)
    Output:  tuple of three tensors (P3, P4, P5)
    """

    # Channel widths per stage
    _CHANNELS = [32, 64, 128, 256, 512]
    # Number of blocks per stage
    _BLOCKS   = [1, 2, 6, 2]

    def __init__(self, in_channels: int = 3, drop_path_rate: float = 0.1):
        super().__init__()

        # Build a linearly-spaced drop-path schedule across ALL blocks
        total_blocks = sum(self._BLOCKS)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # ---------- Stem ----------
        self.stem = ConvBN(
            in_channels, self._CHANNELS[0],
            kernel_size=3, stride=2, padding=1, act=True,
        )

        # ---------- Stages ----------
        block_idx = 0
        stages = []
        for stage_i, n_blocks in enumerate(self._BLOCKS):
            in_ch  = self._CHANNELS[stage_i]
            out_ch = self._CHANNELS[stage_i + 1]
            stage_dpr = dpr[block_idx: block_idx + n_blocks]
            stages.append(
                SWNStage(
                    in_ch, out_ch,
                    num_blocks=n_blocks,
                    stride=2,
                    drop_path_rates=stage_dpr,
                )
            )
            block_idx += n_blocks

        self.stage1 = stages[0]   # → 64  ch, H/4
        self.stage2 = stages[1]   # → 128 ch, H/8   (P3)
        self.stage3 = stages[2]   # → 256 ch, H/16  (P4)
        self.stage4 = stages[3]   # → 512 ch, H/32  (P5)

        self._init_weights()

    def _init_weights(self):
        """Kaiming Normal init for conv layers; const init for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            (P3, P4, P5): feature maps at 1/8, 1/16, 1/32 of input resolution.
        """
        x = self.stem(x)      # (B,  32, H/2,  W/2)
        x = self.stage1(x)    # (B,  64, H/4,  W/4)
        P3 = self.stage2(x)   # (B, 128, H/8,  W/8)
        P4 = self.stage3(P3)  # (B, 256, H/16, W/16)
        P5 = self.stage4(P4)  # (B, 512, H/32, W/32)
        return P3, P4, P5


# ============================================================
# MODULE 3 — CFAN: Cross-Feature Attention Fusion Network
# ============================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block (channel attention).

    Implements:
        s = σ(W2 · δ(W1 · GAP(x)))
        output = x * s

    Args:
        channels   (int): Number of input channels.
        reduction  (int): Reduction ratio r for the MLP bottleneck.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)   # guard against tiny channels
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.fc1   = nn.Linear(channels, mid,      bias=True)
        self.act   = nn.ReLU(inplace=True)
        self.fc2   = nn.Linear(mid, channels,      bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        s = self.gap(x).view(B, C)                # (B, C)
        s = self.act(self.fc1(s))                  # (B, mid)
        s = self.sigmoid(self.fc2(s))              # (B, C)
        s = s.view(B, C, 1, 1)                     # broadcast-ready
        return x * s                               # (B, C, H, W)


class CFAN(nn.Module):
    """
    Cross-Feature Attention Fusion Network.

    Takes two feature maps (possibly with different channel counts) and
    fuses them via Squeeze-and-Excitation channel attention with residual
    connections.

    Pipeline (following paper §III-B):

        1. Channel alignment  : X'0 = ConvBN(X0) if C0 ≠ C  else X0
           where C = max(C0, C1)
        2. Concatenation      : X_cat = [X'0, X1]     — (B, 2C, H, W)
        3. SE attention       : X_att = SE(X_cat)      — (B, 2C, H, W)
        4. Split              : X0_att, X1_att = split(X_att, C, dim=1)
        5. Gated features     : X*0 = X'0 ⊗ X0_att
                                X*1 = X1  ⊗ X1_att
        6. Residual fusion    : Y0 = X'0 + X*1
                                Y1 = X1  + X*0
        7. Output             : Y = [Y0, Y1]           — (B, 2C, H, W)

    Args:
        c0 (int): Channel count of first  input X0.
        c1 (int): Channel count of second input X1.
        reduction (int): SE bottleneck reduction ratio.

    Input:  X0 (B, C0, H, W),  X1 (B, C1, H, W)   — same H, W
    Output: Y  (B, 2*max(C0,C1), H, W)
    """

    def __init__(self, c0: int, c1: int, reduction: int = 16):
        super().__init__()
        self.C = C = max(c0, c1)   # unified channel width

        # 1. Align X0 to C channels if needed
        self.align = (
            ConvBN(c0, C, kernel_size=1, act=True) if c0 != C else nn.Identity()
        )

        # 3. SE block on the 2C-channel concatenation
        self.se = SEBlock(2 * C, reduction=reduction)

    @property
    def out_channels(self) -> int:
        """Convenience accessor used when chaining modules."""
        return 2 * self.C

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0: (B, C0, H, W)
            x1: (B, C1, H, W)

        Returns:
            Y : (B, 2C, H, W)  where C = max(C0, C1)
        """
        # Step 1 — align X0 channels
        x0_prime = self.align(x0)           # (B, C, H, W)

        # Step 2 — concatenate
        x_cat = torch.cat([x0_prime, x1], dim=1)   # (B, 2C, H, W)

        # Step 3 — SE channel attention
        x_att = self.se(x_cat)              # (B, 2C, H, W)

        # Step 4 — split back into the two C-channel halves
        x0_att, x1_att = torch.chunk(x_att, 2, dim=1)  # each (B, C, H, W)

        # Step 5 — element-wise gating
        x_star_0 = x0_prime * x0_att       # X*0
        x_star_1 = x1       * x1_att       # X*1

        # Step 6 — residual fusion (cross-branch residuals from paper)
        Y0 = x0_prime + x_star_1           # X'0 attended by X1's SE weights
        Y1 = x1       + x_star_0           # X1  attended by X0's SE weights

        # Step 7 — concatenate fused branches
        return torch.cat([Y0, Y1], dim=1)  # (B, 2C, H, W)
