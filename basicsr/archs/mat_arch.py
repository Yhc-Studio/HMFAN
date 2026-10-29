import math
import torch
import torch.nn.functional as F
from natten.functional import na2d_av, na2d_qk
from torch import Tensor, nn
from torch.nn.init import trunc_normal_
from typing import List, Optional, Sequence

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.hfma import HybridMultiBandFMA

class ChannelAttention(nn.Module):

    def __init__(self, dim, squeeze_factor=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // squeeze_factor, 1),
            nn.SiLU(),
            nn.Conv2d(dim // squeeze_factor, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class LAB(nn.Module):

    def __init__(self, dim, local_dwconv=3, expanded_ratio=1., squeeze_factor=4):
        super().__init__()
        hidden_dim = int(dim * expanded_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                local_dwconv,
                padding=local_dwconv // 2,
                groups=hidden_dim),
            nn.GELU(),
            ChannelAttention(dim=hidden_dim, squeeze_factor=squeeze_factor),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        u = x.clone()
        x = self.net(x)
        return u + x


class NeighborhoodAttention2D(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
        self,
        dim: int,
        num_head: int,
        kernel_sizes: List[int] = [7, 9, 11],
        dilations: List[int] = [1, 1, 1],
        is_causal: List[bool] = [False, False],
        rel_pos_bias: bool = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_range_gate: bool = True,
        gate_mode = "softmax",   # or "softmax"
        gate_hidden: Optional[int] = None,  # None -> auto
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations)
        if any(is_causal) and rel_pos_bias:
            raise NotImplementedError("Causal neighborhood attention is undefined with positional biases."
                                      "Please consider disabling positional biases, or open an issue.")

        self.k = len(kernel_sizes)
        self.channels = []
        for i in range(self.k):
            if i == 0:
                channels = dim * 3 - dim * 3 // len(kernel_sizes) * (len(kernel_sizes) - 1)
            else:
                channels = dim * 3 // len(kernel_sizes)
            assert (channels % (3 * num_head // self.k) == 0)
            self.channels.append(channels)
        self.gate_mode = gate_mode
        self.num_head = num_head
        self.head_dim = dim // self.num_head
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_sizes = tuple((i, i) for i in kernel_sizes)
        self.dilations = tuple((i, i) for i in dilations)
        self.is_causal = is_causal
        self.use_range_gate = use_range_gate
        if self.use_range_gate:
            gh = gate_hidden if gate_hidden is not None else max(dim // 4, 16)
            # input: [B,C] -> output: [B,k]，after applying softmax, weights are assigned to each range branch.
            self.range_gate = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, gh, bias=True),
                nn.SiLU(),
                nn.Linear(gh, self.k, bias=True),
            )
        else:
            self.range_gate = None

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if rel_pos_bias:
            self.rpb = nn.ParameterList()
            for i in range(len(kernel_sizes)):
                temp = nn.Parameter(torch.zeros(
                    num_head // self.k,
                    (2 * kernel_sizes[i] - 1),
                    (2 * kernel_sizes[i] - 1),
                ))
                trunc_normal_(temp, mean=0.0, std=0.02, a=-2.0, b=2.0)
                self.rpb.append(temp)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        self.extra_repr()
        if x.dim() != 4:
            raise ValueError(f"NeighborhoodAttention2D expected a rank-4 input tensor; got {x.dim()=}.")
        # ---- range weights (new) ----
        # Generate the weights for each range using the global statistics (GAP on HW) for the input x.
        if self.use_range_gate:
            gate_mode = getattr(self, "gate_mode", "softmax")
            if gate_mode == "fixed_uniform":
                # C1: constant-uniform weights, shape [B, k]
                B = x.shape[0]
                w = x.new_full((B, self.k), 1.0 / self.k)
            else:
                g = x.mean(dim=(1, 2))          # [B, C]
                w_raw = self.range_gate(g)      # [B, k]

                if gate_mode == "softmax":
                    w = torch.softmax(w_raw, dim=-1)
                elif gate_mode == "sigmoid":
                    w = torch.sigmoid(w_raw)
                    w = w / (w.mean(dim=-1, keepdim=True) + 1e-6)  # mean≈1
                else:
                    raise ValueError(f"Unsupported gate_mode: {gate_mode}")
        else:
            w = None
        x = self.qkv(x)
        x = torch.split(x, split_size_or_sections=self.channels, dim=3)
        attns = []
        for i, x_i in enumerate(x):
            B, H, W, C = x_i.shape
            qkv = (x_i.reshape(B, H, W, 3, self.num_head // self.k, self.head_dim).permute(3, 0, 4, 1, 2, 5))
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * self.scale
            attn = na2d_qk(
                q,
                k,
                kernel_size=self.kernel_sizes[i],
                dilation=self.dilations[i],
                is_causal=self.is_causal,
                rpb=self.rpb[i] if self.rpb is not None else None,
            )
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            y = na2d_av(
                attn,
                v,
                kernel_size=self.kernel_sizes[i],
                dilation=self.dilations[i],
                is_causal=self.is_causal,
            )
            y = y.permute(0, 2, 3, 1, 4).reshape(B, H, W, C // 3)
            if w is not None:
                y = y * w[:, i].view(B, 1, 1, 1)       # Each range has a scalar weight.
            attns.append(y)
        x = torch.cat(attns, dim=3)
        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (f"head_dim={self.head_dim}, num_head={self.num_head}, " + f"kernel_sizes={self.kernel_sizes}, " +
                f"dilations={self.dilations}, " + f"is_causal={self.is_causal}, " + f"has_bias={self.rpb is not None}")


class MSDWConv(nn.Module):

    def __init__(self, dim, dw_sizes=(1, 3, 5, 7)):
        super().__init__()
        self.dw_sizes = dw_sizes
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(dw_sizes)):
            if i == 0:
                channels = dim - dim // len(dw_sizes) * (len(dw_sizes) - 1)
            else:
                channels = dim // len(dw_sizes)
            conv = nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_sizes[i],
                padding=dw_sizes[i] // 2,
                groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


class MSConvStar(nn.Module):

    def __init__(self, dim, mlp_ratio=2., dw_sizes=[1, 3, 5, 7]):
        super().__init__()
        self.dim = dim
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.dwconv = MSDWConv(dim=hidden_dim, dw_sizes=dw_sizes)
        self.fc2 = nn.Conv2d(hidden_dim // 2, dim, 1)
        self.num_head = len(dw_sizes)
        self.act = nn.GELU()

        assert hidden_dim // self.num_head % 2 == 0

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.fc1(x)
        x = x + self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.fc2(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class BDRLite(nn.Module):
    """
    Block Depth Routing Lite

    routing_type:
        - 'softmax': input-dependent softmax routing
        - 'sigmoid': input-dependent sigmoid routing (normalized)
        - 'static': fixed equal-weight fusion
    """

    def __init__(self, dim, num_memory=3, reduction=4, routing_type='static'):
        super().__init__()
        if num_memory < 1:
            raise ValueError(f'num_memory must be >= 1, but got {num_memory}.')
        if routing_type not in ['softmax', 'sigmoid', 'static']:
            raise ValueError(
                f"routing_type must be one of ['softmax', 'sigmoid', 'static'], "
                f"but got {routing_type}."
            )

        self.num_memory = num_memory
        self.routing_type = routing_type

        if self.routing_type != 'static':
            hidden_dim = max(dim // reduction, 4)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
            self.act = nn.GELU()
            self.fc2 = nn.Conv2d(hidden_dim, num_memory, 1, 1, 0)

    def _get_weights(self, x, num_mems):
        if self.routing_type == 'static':
            return x.new_ones(1, num_mems, 1, 1) / float(num_mems)

        logits = self.fc2(self.act(self.fc1(self.pool(x))))

        if self.routing_type == 'softmax':
            weights = torch.softmax(logits[:, :num_mems], dim=1)
        else:
            weights = torch.sigmoid(logits[:, :num_mems])
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        return weights

    def forward(self, x, mems):
        if len(mems) == 0:
            return x

        num_mems = len(mems)
        weights = self._get_weights(x, num_mems)

        fused = torch.zeros_like(x)
        for i, m in enumerate(mems):
            fused = fused + m * weights[:, i:i + 1]

        return x + fused


class InterGroupStaticFusion(nn.Module):
    """
    Static inter-group memory fusion.
    """

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x, mems):
        if len(mems) == 0:
            return x
        fused = torch.zeros_like(x)
        for m in mems:
            fused = fused + m
        fused = fused / float(len(mems))
        return x + self.scale * fused


class MAB(nn.Module):

    def __init__(self,
                 dim,
                 num_head,
                 kernel_sizes=[7, 9, 11],
                 dilations=[1, 1, 1],
                 rel_pos_bias=True,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 mlp_ratio=2.0,
                 dw_sizes=[1, 3, 5, 7],
                 use_range_gate=True,
                 gate_mode="softmax",
                 gate_hidden=None
                 ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NeighborhoodAttention2D(
            dim=dim,
            num_head=num_head,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_range_gate=use_range_gate,
            gate_mode=gate_mode,
            gate_hidden=gate_hidden)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MSConvStar(dim, mlp_ratio=mlp_ratio, dw_sizes=dw_sizes)
        self.dilations = dilations

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RMAG(nn.Module):

    def __init__(self,
                 dim,
                 local_dwconv=3,
                 expanded_ratio=1.,
                 squeeze_factor=4,
                 depth=4,
                 num_head=6,
                 kernel_sizes=[7, 9, 11],
                 dilations=[[1, 1, 1], [4, 4, 4]],
                 rel_pos_bias=True,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_ratio=2.0,
                 dw_sizes=[1, 3, 5, 7],
                 use_bdr=False,
                 memory_indices: Optional[Sequence[int]] = None,
                 routing_reduction=4,
                 routing_type='static',
                 use_range_gate=True,
                 gate_mode="softmax",
                 gate_hidden=None,
                 num_heads=6,
                 fourier_heads=2,
                 num_bands=3):
        super().__init__()
        self.depth = depth
        self.use_bdr = use_bdr

        self.lab = LAB(
            dim=dim,
            local_dwconv=local_dwconv,
            expanded_ratio=expanded_ratio,
            squeeze_factor=squeeze_factor)
        self.fma = HybridMultiBandFMA(
            dim=dim, 
            num_heads=num_heads, 
            fourier_heads=fourier_heads, 
            num_bands=num_bands)
        self.mabs = nn.ModuleList()
        for i_mab in range(depth):
            mab = MAB(
                dim=dim,
                num_head=num_head,
                kernel_sizes=kernel_sizes,
                dilations=dilations[i_mab % 2],
                rel_pos_bias=rel_pos_bias,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                mlp_ratio=mlp_ratio,
                dw_sizes=dw_sizes,
                use_range_gate=use_range_gate,
                gate_mode=gate_mode,
                gate_hidden=gate_hidden)
            self.mabs.append(mab)

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

        if self.use_bdr:
            if self.depth < 2:
                raise ValueError('use_bdr=True requires depth >= 2.')

            if memory_indices is None:
                memory_indices = list(range(min(3, self.depth - 1)))

            memory_indices = sorted(set(int(i) for i in memory_indices))
            if len(memory_indices) == 0:
                raise ValueError('memory_indices must contain at least one valid index when use_bdr=True.')

            max_valid_idx = self.depth - 2
            for idx in memory_indices:
                if idx < 0 or idx > max_valid_idx:
                    raise ValueError(
                        f'All memory_indices must be in [0, {max_valid_idx}] when depth={self.depth}, '
                        f'but got {memory_indices}.'
                    )

            self.memory_indices = tuple(memory_indices)
            self.memory_compress = nn.ModuleList([
                nn.Conv2d(dim, dim, 1, 1, 0) for _ in self.memory_indices
            ])
            self.bdr = BDRLite(
                dim=dim,
                num_memory=len(self.memory_indices),
                reduction=routing_reduction,
                routing_type=routing_type)
        else:
            self.memory_indices = tuple()

    def forward(self, x):
        shortcut = x
        x = self.fma(x)  # After FMA processing, it still maintains [B, C, H, W]
        x = self.lab(x)

        x = x.permute(0, 2, 3, 1).contiguous()

        mems = []
        mem_ptr = 0

        for i, mab in enumerate(self.mabs):
            x = mab(x)

            if self.use_bdr and i in self.memory_indices:
                mem = x.permute(0, 3, 1, 2).contiguous()
                mem = self.memory_compress[mem_ptr](mem)
                mems.append(mem)
                mem_ptr += 1

        x = x.permute(0, 3, 1, 2).contiguous()

        if self.use_bdr and len(mems) > 0:
            x = self.bdr(x, mems)

        x = self.conv(x)
        x = x + shortcut
        return x


def UpsampleOneStep(in_channels, out_channels, upscale_factor=4):
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])


class Upsample(nn.Sequential):
    """Upsample module."""

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class PixelShuffleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_factor=4):
        super().__init__()
        num_feat = 64
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(in_channels, num_feat, 3, 1, 1),
            nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(upscale_factor, num_feat)
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))
        return x


@ARCH_REGISTRY.register()
class MAT(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=60,
                 num_block=4,
                 local_dwconv=3,
                 expanded_ratio=1.0,
                 squeeze_factor=4,
                 depth=4,
                 num_head=6,
                 kernel_sizes=[7, 9, 11],
                 dilations=[[1, 1, 1], [4, 4, 4]],
                 rel_pos_bias=True,
                 qkv_bias=True,
                 qk_scale=False,
                 mlp_ratio=2.0,
                 dw_sizes=[1, 3, 5, 7],
                 # Intra-group Memory Fusion (Intra-MF)
                 use_bdr=False,
                 memory_indices: Optional[Sequence[int]] = None,
                 routing_reduction=4,
                 routing_type='static',
                 # Inter-group Memory Fusion (Inter-MF)
                 use_cgmf=False,
                 cgmf_num_prev=1,
                 cgmf_scale=1.0,
                 upscale=4,
                 # RangeGate
                 use_range_gate=True,
                 gate_mode="softmax",
                 gate_hidden=None,
                 upsampler='',
                 # HFMA
                 num_heads=6,
                 fourier_heads=2,
                 num_bands=3,
                 img_range=1):
        super().__init__()
        self.img_range = img_range
        self.num_block = num_block
        self.use_cgmf = use_cgmf
        self.cgmf_num_prev = cgmf_num_prev

        if num_in_ch == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.body = nn.ModuleList([
            RMAG(
                dim=num_feat,
                local_dwconv=local_dwconv,
                expanded_ratio=expanded_ratio,
                squeeze_factor=squeeze_factor,
                depth=depth,
                num_head=num_head,
                kernel_sizes=kernel_sizes,
                dilations=dilations,
                rel_pos_bias=rel_pos_bias,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                mlp_ratio=mlp_ratio,
                dw_sizes=dw_sizes,
                use_bdr=use_bdr,
                memory_indices=memory_indices,
                routing_reduction=routing_reduction,
                routing_type=routing_type,
                use_range_gate=use_range_gate,
                gate_mode=gate_mode,
                gate_hidden=gate_hidden,
                num_heads=num_heads,
                fourier_heads=fourier_heads,
                num_bands=num_bands
            )
            for _ in range(num_block)
        ])

        if self.use_cgmf:
            if self.num_block < 2:
                raise ValueError('use_cgmf=True requires num_block >= 2.')
            if self.cgmf_num_prev < 1:
                raise ValueError(f'cgmf_num_prev must be >= 1, but got {self.cgmf_num_prev}.')

            # Generate group memories for the first num_block - 1 groups.
            self.group_memory_compress = nn.ModuleList([
                nn.Conv2d(num_feat, num_feat, 1, 1, 0) for _ in range(num_block - 1)
            ])
            self.inter_group_fusion = InterGroupStaticFusion(scale=cgmf_scale)

        self.conv_afterbody = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.upscale = upscale
        if upsampler == 'pixelshuffledirect':
            self.upsampler = UpsampleOneStep(num_feat, num_out_ch, upscale_factor=upscale)
        elif upsampler == 'pixelshuffle':
            self.upsampler = PixelShuffleBlock(num_feat, num_out_ch, upscale_factor=upscale)
        else:
            raise NotImplementedError("Check the Upsampler. None or not support yet.")

    def check_image_size(self, x, min_size=64):
        _, _, H, W = x.size()
        if H >= min_size and W >= min_size:
            return x
        mod_pad_h = max(min_size - H, 0)
        mod_pad_w = max(min_size - W, 0)
        padding = (0, mod_pad_w, 0, mod_pad_h)
        x = F.pad(x, padding, 'reflect')
        return x

    def _select_group_mems(self, group_mems):
        if len(group_mems) == 0:
            return []
        k = min(self.cgmf_num_prev, len(group_mems))
        return group_mems[-k:]

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.fea_conv(x)
        shallow_feat = x

        group_mems = []
        for i, block in enumerate(self.body):
            if self.use_cgmf and len(group_mems) > 0:
                prev_mems = self._select_group_mems(group_mems)
                x = self.inter_group_fusion(x, prev_mems)

            x = block(x)

            # Only the first num_block - 1 groups generate memories for subsequent groups.
            if self.use_cgmf and i < self.num_block - 1:
                mem = self.group_memory_compress[i](x)
                group_mems.append(mem)

        x = self.conv_afterbody(x) + shallow_feat
        x = self.upsampler(x)

        x = x / self.img_range + self.mean
        return x[:, :, :H * self.upscale, :W * self.upscale]

@ARCH_REGISTRY.register()
class HMFAN(MAT):
    """HMFAN with the canonical public/released architecture defaults.

    Explicit values supplied by BasicSR YAML configurations take precedence.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('dilations', [[1, 1, 1], [9, 7, 5]])
        kwargs.setdefault('use_bdr', True)
        kwargs.setdefault('memory_indices', [0, 1, 2])
        kwargs.setdefault('routing_reduction', 8)
        kwargs.setdefault('routing_type', 'static')
        kwargs.setdefault('use_cgmf', True)
        kwargs.setdefault('cgmf_num_prev', 2)
        kwargs.setdefault('cgmf_scale', 1.0)
        kwargs.setdefault('use_range_gate', True)
        kwargs.setdefault('gate_mode', 'softmax')
        kwargs.setdefault('num_heads', 6)
        kwargs.setdefault('fourier_heads', 2)
        kwargs.setdefault('num_bands', 3)
        super().__init__(*args, **kwargs)
