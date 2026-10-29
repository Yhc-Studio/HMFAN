import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================
# LayerNorm（channels_first: N,C,H,W）
# ============================================================
class LayerNorm(nn.Module):
    """
    LayerNorm that supports channels_first tensors in NCHW format.

    This implementation is tailored for vision backbones where feature maps
    are usually represented as [B, C, H, W]. For channels_first mode,
    normalization is performed along the channel dimension manually.

    Args:
        normalized_shape (int): Number of channels.
        eps (float): A small constant for numerical stability.
        data_format (str): Tensor layout, either "channels_first" or "channels_last".
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        assert data_format in ["channels_last", "channels_first"]
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        # channels_first: x is [B,C,H,W]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# ============================================================
# Multi-band Fourier Unit
# ============================================================
class MultiBandFourierUnit(nn.Module):
    """
    Multi-band Fourier unit used in HFMA.

    The input is transformed into the frequency domain with rfft2.
    Real and imaginary parts are concatenated along the channel dimension,
    then the spectrum is split into multiple frequency bands along the frequency
    width axis. Each band is processed by an independent grouped 1x1 convolution.
    Finally, the processed spectrum is transformed back to the spatial domain
    through irfft2.

    This design provides global frequency interaction with grouped band processing.

    Args:
        dim (int): Number of input channels.
        num_bands (int): Number of frequency bands.
        groups (int): Number of groups in grouped 1x1 convolutions.
        fft_norm (str): Normalization mode used in FFT.
    """

    def __init__(self, dim, num_bands=3, groups=2, fft_norm='ortho'):
        super().__init__()
        assert num_bands >= 1
        assert (2 * dim) % groups == 0, f"[FourierUnit] 2*dim={2*dim} must be divisible by groups={groups}"
        self.num_bands = num_bands
        self.fft_norm = fft_norm
        self.groups = groups

        self.band_convs = nn.ModuleList([
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, groups=groups, bias=False)
            for _ in range(num_bands)
        ])
        self.act = nn.GELU()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map of shape [B, C, H, W].

        Returns:
            Tensor: Output feature map of shape [B, C, H, W].
        """

        B, C, H, W = x.shape

        ffted = torch.fft.rfft2(x, norm=self.fft_norm)     # [B,C,H,Wf] complex
        real, imag = ffted.real, ffted.imag
        combined = torch.cat([real, imag], dim=1)          # [B,2C,H,Wf]
        _, _, Hf, Wf = combined.shape

        chunk = math.ceil(Wf / self.num_bands)
        outs = []
        for i, conv in enumerate(self.band_convs):
            start = i * chunk
            end = min((i + 1) * chunk, Wf)
            if start >= end:
                outs.append(torch.zeros(B, 2 * C, Hf, 0, device=x.device, dtype=combined.dtype))
                continue
            band = combined[:, :, :, start:end]
            band = self.act(conv(band))
            outs.append(band)
        # Merge all processed bands back together.
        out_combined = torch.cat(outs, dim=-1)             # [B,2C,H,Wf]
        out_real, out_imag = torch.chunk(out_combined, 2, dim=1)
        out_complex = torch.complex(out_real, out_imag)
        # Transform the processed spectrum back to the spatial domain.
        out = torch.fft.irfft2(out_complex, s=(H, W), norm=self.fft_norm)
        return out


# ============================================================
# HFMA
# ============================================================
class HybridMultiBandFMA(nn.Module):
    """
    Hybrid Fourier Modulated Attention (HFMA) combines:
        1. Fourier-based global modulation
        2. Spatial local modulation
        3. Hybrid multi-head feature interaction

    The Fourier branch is restricted to the channels assigned to Fourier heads,
    while the remaining heads are refined in the spatial domain. This selective
    design avoids full-channel spectral processing while preserving complementary
    frequency-domain and spatial-domain modulation.

    Main components:
        - Layer normalization in channels-first format
        - Lightweight convolutional positional encoding (CPE)
        - Shared value projection
        - Fourier branch for global frequency-aware modulation
        - Spatial branch for local modulation
        - Head-wise gating for adaptive modulation strength
        - Output projection with residual connection

    Args:
        dim (int): Number of input channels.
        num_heads (int): Total number of heads.
        fourier_heads (int): Number of heads assigned to the Fourier branch.
        num_bands (int): Number of frequency bands in the Fourier unit.
        use_bottleneck (bool): Whether to use channel reduction before Fourier modeling.
        bottleneck_ratio (int): Reduction ratio for the bottleneck.
        min_cr (int): Minimum bottleneck channel dimension.
        fft_norm (str): Normalization mode used in FFT.
        gate_hidden (int or None): Hidden dimension in the head-wise gate generator.
    """
    def __init__(
        self,
        dim=60,
        num_heads=6,
        fourier_heads=2,
        num_bands=3,
        use_bottleneck=True,
        bottleneck_ratio=4,
        min_cr=12,
        fft_norm='ortho',
        gate_hidden=None,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert 0 <= fourier_heads <= num_heads

        self.dim = dim
        self.num_heads = num_heads
        self.fourier_heads = fourier_heads
        self.spatial_heads = num_heads - fourier_heads
        self.head_dim = dim // num_heads

        # Only the channels corresponding to Fourier heads are sent to
        # the Fourier branch in the lightweight design.
        self.Cf = self.fourier_heads * self.head_dim  # Channels assigned to Fourier heads.

        # Normalization and lightweight positional encoding.
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.cpe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        # Shared value projection for all heads.
        self.v = nn.Conv2d(dim, dim, 1, 1, 0)

        # Spatial modulation branch for spatial heads.
        if self.spatial_heads > 0:
            self.spatial_dw = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
            self.spatial_pw = nn.Conv2d(dim, dim, 1, 1, 0)

        # Output projection and learnable layer scaling.
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0)
        self.layer_scale = nn.Parameter(1e-6 * torch.ones(1, dim, 1, 1))

        # Head-wise gate generator.
        # A single global descriptor is mapped to per-head gates so that
        # different heads can adaptively control their modulation strength.
        if gate_hidden is None:
            gate_hidden = max(dim // 8, 8)
        self.gate_gen_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, gate_hidden, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(gate_hidden, num_heads, 1, bias=True),
            nn.Sigmoid()
        )

        # Fourier branch used by Fourier heads.
        # A bottleneck can be optionally introduced before Fourier modeling
        # to further reduce computation.
        self.use_bottleneck = use_bottleneck
        self.fft_norm = fft_norm

        if self.fourier_heads > 0:
            # Use grouped processing aligned with Fourier heads.
            self.f_groups = max(self.fourier_heads, 1)

            if use_bottleneck:
                cr0 = max(self.Cf // bottleneck_ratio, min_cr)
                Cr = self._fix_cr_for_groups(cr0, self.f_groups)
                self.Cr = Cr

                self.f_reduce = nn.Conv2d(self.Cf, Cr, 1, 1, 0, bias=False)
                self.f_fourier = MultiBandFourierUnit(
                    dim=Cr,
                    num_bands=num_bands,
                    groups=self.f_groups,
                    fft_norm=fft_norm
                )
                self.f_expand = nn.Conv2d(Cr, self.Cf, 1, 1, 0, bias=False)
            else:
                assert (2 * self.Cf) % self.f_groups == 0
                self.Cr = None
                self.f_fourier = MultiBandFourierUnit(
                    dim=self.Cf,
                    num_bands=num_bands,
                    groups=self.f_groups,
                    fft_norm=fft_norm
                )
        else:
            self.f_groups = 0
            self.Cr = None

    @staticmethod
    def _fix_cr_for_groups(cr, groups):
        """
        Adjust the reduced channel dimension so that (2 * Cr) is divisible by groups.
        This is required because real and imaginary parts are concatenated in the
        Fourier unit.
        """
        Cr = int(cr)
        while (2 * Cr) % groups != 0:
            Cr += 1
        return Cr

    def forward(self, x):
        """
        Forward pass of HFMA.

        Processing steps:
            1. Generate lightweight positional encoding
            2. Normalize input features
            3. Produce head-wise gates from global context
            4. Compute shared value features
            5. Generate local spatial modulation maps
            6. Generate global frequency-aware modulation maps for Fourier heads
            7. Apply hybrid head modulation:
                   - Fourier heads: sigmoid(Fourier map) * gate * value
                   - Spatial heads: tanh(spatial map) * value
            8. Merge all heads and project back
            9. Add residual shortcut

        Args:
            x (Tensor): Input feature map of shape [B, C, H, W].

        Returns:
            Tensor: Output feature map of shape [B, C, H, W].
        """
        B, C, H, W = x.shape
        shortcut = x
        # Lightweight convolutional positional encoding.
        pos = self.cpe(x)

        # Normalize the input before modulation.
        xn = self.norm(x)

        # Generate one gate value for each head.
        gate_h = self.gate_gen_head(xn)  # [B, num_heads, 1, 1]

        # Shared value features.
        v = self.v(xn)                   # [B, C, H, W]

        # Local spatial modulation maps for spatial heads.
        if self.spatial_heads > 0:
            s = self.spatial_pw(self.spatial_dw(xn))     # [B, C, H, W]
        else:
            s = None

        # ------------------------------------------------------------
        # Fourier branch
        # Only channels assigned to Fourier heads are processed here.
        # ------------------------------------------------------------
        # Reshape Fourier modulation to [B, fourier_heads, head_dim, HW].
        if self.fourier_heads > 0:
            xf = xn[:, :self.Cf, :, :]  # [B, Cf, H, W]

            if self.use_bottleneck:
                f = self.f_reduce(xf)              # [B, Cr, H, W]
                f = self.f_fourier(f)              # [B, Cr, H, W]
                a_f_spatial = self.f_expand(f)     # [B, Cf, H, W]
            else:
                a_f_spatial = self.f_fourier(xf)   # [B, Cf, H, W]

            # Reshape Fourier features into head format.
            a_f = rearrange(a_f_spatial, 'b (h c) H W -> b h c (H W)', h=self.fourier_heads)
        else:
            a_f = None

        # Convert value features into multi-head representation.
        v_h = rearrange(v, 'b (h c) H W -> b h c (H W)', h=self.num_heads)

        # Fourier heads:
        # Use frequency-aware maps together with head-wise gates to modulate values.
        if self.fourier_heads > 0:
            v_f = v_h[:, :self.fourier_heads]                  # [B, hf, hd, HW]
            g_f = gate_h[:, :self.fourier_heads].view(B, self.fourier_heads, 1, 1)
            out_f = torch.sigmoid(a_f) * g_f * v_f             # [B, hf, hd, HW]
        else:
            out_f = None

        # Spatial heads:
        # Use local spatial maps to modulate value features.
        if self.spatial_heads > 0:
            v_s = v_h[:, self.fourier_heads:]
            if s is None:
                out_s = v_s
            else:
                s_h = rearrange(s, 'b (h c) H W -> b h c (H W)', h=self.num_heads)
                s_s = s_h[:, self.fourier_heads:]
                out_s = torch.tanh(s_s) * v_s
        else:
            out_s = None

        # Merge Fourier heads and spatial heads.
        if out_f is None:
            out_all = out_s
        elif out_s is None:
            out_all = out_f
        else:
            out_all = torch.cat([out_f, out_s], dim=1)

        # Restore spatial layout [B, C, H, W]
        y = rearrange(out_all, 'b h c (H W) -> b (h c) H W', H=H, W=W)

        # Apply layer scaling, inject positional bias, then project.
        y = y * self.layer_scale + pos
        y = self.proj(y)
        # Residual connection.
        return shortcut + y
