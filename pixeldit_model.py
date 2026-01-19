""" 
PixelDiT: Pixel Diffusion Transformers for Image Generation

Standalone implementation - all dependencies included.
Based on the paper: "PixelDiT: Pixel Diffusion Transformers for Image Generation"

Key innovations:
1. Dual-level DiT Architecture: patch-level for global semantics, pixel-level for texture details
2. Pixel-wise AdaLN: per-pixel modulation parameters instead of patch-wise
3. Pixel Token Compaction: compress pixel tokens for efficient attention
4. MM-DiT blocks for text-to-image generation

Based on LightningDiT codebase
"""

import math
import numpy as np
from math import pi
from typing import Optional, Tuple, Literal, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat

from timm.models.vision_transformer import Mlp


# =============================================================================
# Inlined Dependencies
# =============================================================================

# -----------------------------------------------------------------------------
# RMSNorm (from models/rmsnorm.py)
# -----------------------------------------------------------------------------

class RMSNorm(torch.nn.Module):
    """RMSNorm normalization layer."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# -----------------------------------------------------------------------------
# SwiGLUFFN (from models/swiglu_ffn.py)
# -----------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


# -----------------------------------------------------------------------------
# VisionRotaryEmbeddingFast (from models/pos_embed.py)
# -----------------------------------------------------------------------------

def broadcat(tensors, dim=-1):
    """Broadcast and concatenate tensors."""
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    """Rotate half of the dimensions."""
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbeddingFast(nn.Module):
    """Fast Vision Rotary Embedding."""
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for='lang',
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


# =============================================================================
# Basic Components
# =============================================================================

def modulate(x: torch.Tensor, shift: Optional[torch.Tensor], scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN modulation: x * (1 + scale) + shift"""
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift


def pixel_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply pixel-wise AdaLN modulation.
    Args:
        x: (B*L, p^2, D_pix) pixel tokens
        shift: (B*L, p^2, D_pix) per-pixel shift
        scale: (B*L, p^2, D_pix) per-pixel scale
    """
    return x * (1 + scale) + shift


class Attention(nn.Module):
    """Multi-head self-attention with optional RoPE."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if rope is not None:
            q = rope(q)
            k = rope(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations with dropout for CFG."""
    
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels: torch.Tensor, train: bool, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


# =============================================================================
# Patch-level DiT Block (for global semantic learning)
# =============================================================================

class PatchDiTBlock(nn.Module):
    """Patch-level DiT Block for global semantic learning.
    
    Features: RoPE, QK-Norm, RMSNorm, SwiGLU, AdaLN (with optional shift removal)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        **block_kwargs
    ):
        super().__init__()
        
        # Normalization layers
        if use_rmsnorm:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Attention
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )

        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=lambda: nn.GELU(approximate="tanh"),
                drop=0
            )

        # AdaLN modulation
        self.wo_shift = wo_shift
        num_adaln_params = 4 if wo_shift else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, num_adaln_params * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, rope=None) -> torch.Tensor:
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            shift_msa, shift_mlp = None, None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # unsqueeze(1) adds sequence dimension for broadcasting: [B, D] -> [B, 1, D]
        shift_msa_expand = shift_msa.unsqueeze(1) if shift_msa is not None else None
        shift_mlp_expand = shift_mlp.unsqueeze(1) if shift_mlp is not None else None
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa_expand, scale_msa.unsqueeze(1)), rope=rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp_expand, scale_mlp.unsqueeze(1)))
        return x


# =============================================================================
# Pixel-wise AdaLN and Token Compaction
# =============================================================================

class PixelwiseAdaLN(nn.Module):
    """Pixel-wise AdaLN Modulation.
    
    From Figure 2(C): "Pixel-wise AdaLN applies an MLP to each semantic token 
    to produce per-pixel scale, shift, and gating parameters, enabling fully 
    context-aligned updates at every pixel."
    
    Expands semantic conditioning token into p^2 sets of AdaLN parameters,
    enabling per-pixel modulation instead of patch-wise broadcasting.
    
    MLP: R^D -> R^(p^2 * 6 * D_pix)
    """
    
    def __init__(self, semantic_dim: int, pixel_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.pixel_dim = pixel_dim
        self.num_pixels = patch_size * patch_size
        
        # Output dimension: 6 parameters (shift1, scale1, gate1, shift2, scale2, gate2) 
        # for each of p^2 pixels, each with D_pix dimensions
        out_dim = self.num_pixels * 6 * pixel_dim
        
        # MLP to generate per-pixel AdaLN parameters (following Figure 2C)
        # Structure: Linear -> SiLU -> Linear (standard MLP)
        hidden_dim = semantic_dim  # Use same hidden dim as input
        self.mlp = nn.Sequential(
            # nn.Linear(semantic_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim, bias=True)
        )
        
    def forward(self, s_cond: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            s_cond: (B*L, D) semantic conditioning tokens
        Returns:
            Tuple of 6 tensors, each (B*L, p^2, D_pix):
            (shift1, scale1, gate1, shift2, scale2, gate2)
        """
        # Apply MLP: (B*L, D) -> (B*L, p^2 * 6 * D_pix)
        theta = self.mlp(s_cond)
        
        # Reshape to (B*L, p^2, 6 * D_pix)
        theta = theta.view(-1, self.num_pixels, 6 * self.pixel_dim)
        
        # Split into 6 groups: (B*L, p^2, D_pix) each
        params = theta.chunk(6, dim=-1)
        return params  # (shift1, scale1, gate1, shift2, scale2, gate2)


class PixelTokenCompaction(nn.Module):
    """Pixel Token Compaction mechanism.
    
    From paper Section 3.2:
    "a linear map C: R^(p²×D_pix) → R^D that jointly mixes spatial and channel 
    dimensions, paired with an expansion E: R^D → R^(p²×D_pix)"
    
    Compresses p² pixel tokens into 1 compact token in semantic space (D),
    performs global attention, then expands back to p² pixel tokens.
    This reduces attention sequence length from H×W to L (p²-fold reduction).
    """
    
    def __init__(self, pixel_dim: int, semantic_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.pixel_dim = pixel_dim
        self.semantic_dim = semantic_dim
        self.num_pixels = patch_size * patch_size
        
        # Compress: C: R^(p² × D_pix) → R^D
        # Jointly mixes spatial (p²) and channel (D_pix) dimensions
        self.compress = nn.Linear(self.num_pixels * pixel_dim, semantic_dim)
        
        # Expand: E: R^D → R^(p² × D_pix)
        self.expand = nn.Linear(semantic_dim, self.num_pixels * pixel_dim)
        
    def compact(self, x: torch.Tensor) -> torch.Tensor:
        """Compress pixel tokens to semantic-space patch tokens.
        Args:
            x: (B*L, p², D_pix) pixel tokens
        Returns:
            (B*L, 1, D) compacted patch tokens in semantic space
        """
        B_L = x.shape[0]
        # Flatten pixels: (B*L, p² * D_pix)
        x_flat = x.view(B_L, -1)
        # Compress to semantic space: (B*L, D)
        x_compact = self.compress(x_flat)
        return x_compact.unsqueeze(1)  # (B*L, 1, D)
    
    def decompact(self, x: torch.Tensor) -> torch.Tensor:
        """Expand semantic-space patch tokens back to pixel tokens.
        Args:
            x: (B*L, 1, D) compacted patch tokens in semantic space
        Returns:
            (B*L, p², D_pix) expanded pixel tokens
        """
        B_L = x.shape[0]
        # Remove sequence dim: (B*L, D)
        x = x.squeeze(1)
        # Expand to pixel space: (B*L, p² * D_pix)
        x_expand = self.expand(x)
        # Reshape: (B*L, p², D_pix)
        return x_expand.view(B_L, self.num_pixels, self.pixel_dim)


# =============================================================================
# Pixel Transformer Block (PiT) - Core of pixel-level pathway
# =============================================================================

class PixelTransformerBlock(nn.Module):
    """Pixel Transformer (PiT) Block for pixel-level detail refinement.
    
    Key components:
    1. Pixel-wise AdaLN: per-pixel modulation from semantic tokens
    2. Pixel Token Compaction: compress to semantic space for efficient attention
    
    The block operates on pixel tokens within each patch, conditioned by
    semantic tokens from the patch-level pathway.
    
    From paper Section 3.2:
    - Compaction: C: R^(p²×D_pix) → R^D (compress to semantic space)
    - Attention: operates on compacted tokens in semantic space D
    - Expansion: E: R^D → R^(p²×D_pix) (expand back to pixel space)
    """
    
    def __init__(
        self,
        pixel_dim: int,
        semantic_dim: int,
        num_heads: int,           # num_heads for semantic-space attention
        patch_size: int,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = True,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        self.pixel_dim = pixel_dim
        self.semantic_dim = semantic_dim
        self.patch_size = patch_size
        self.num_pixels = patch_size * patch_size
        
        # Pixel-wise AdaLN modulation (operates in pixel space)
        self.pixelwise_adaln = PixelwiseAdaLN(semantic_dim, pixel_dim, patch_size)
        
        # Pixel token compaction: pixel space (D_pix) <-> semantic space (D)
        self.compaction = PixelTokenCompaction(pixel_dim, semantic_dim, patch_size)
        
        # Normalization (in pixel space, before compaction)
        if use_rmsnorm:
            self.norm1 = RMSNorm(pixel_dim)
            self.norm2 = RMSNorm(pixel_dim)
        else:
            self.norm1 = nn.LayerNorm(pixel_dim, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(pixel_dim, elementwise_affine=False, eps=1e-6)
        
        # Attention operates on compacted tokens in SEMANTIC space (D)
        self.attn = Attention(
            semantic_dim,  # Attention in semantic space!
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
        )
        
        # MLP (in pixel space)
        mlp_hidden_dim = int(pixel_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=pixel_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0
        )
    
    def forward(self, x: torch.Tensor, s_cond: torch.Tensor, rope=None) -> torch.Tensor:
        """
        Forward pass of Pixel Transformer Block.
        
        Paper Section 3.2: "To align with patch-level semantic tokens, we reshape
        into B·L sequences of p² pixel tokens, i.e., X ∈ R^(B·L)×p²×D_pix"
        
        Data flow:
        1. Pixel-wise AdaLN modulation (in pixel space D_pix)
        2. Compaction: (B*L, p², D_pix) -> (B*L, 1, D_semantic) 
        3. Global attention in semantic space D
        4. Expansion: (B*L, 1, D_semantic) -> (B*L, p², D_pix)
        5. Residual + MLP (in pixel space D_pix)
        
        Args:
            x: (B, L, p², D_pix) patch-local pixel tokens
               - B: batch size
               - L: number of patches = (H/p) * (W/p)
               - p²: pixels per patch (e.g., 256 for p=16)
               - D_pix: pixel dimension (16 per paper)
            s_cond: (B, L, D) semantic conditioning tokens from patch-level pathway
        Returns:
            (B, L, p², D_pix) refined pixel tokens
        """
        B, L, p2, D_pix = x.shape
        D_sem = self.semantic_dim
        
        # Reshape to (B*L, p², D_pix) for patch-local processing
        x = x.reshape(B * L, p2, D_pix)
        
        # Flatten s_cond: (B, L, D_semantic) -> (B*L, D_semantic)
        # Each patch gets its corresponding semantic token for modulation
        s_cond_flat = s_cond.reshape(B * L, -1)
        
        # Get pixel-wise AdaLN parameters: each produces (B*L, p², D_pix)
        shift1, scale1, gate1, shift2, scale2, gate2 = self.pixelwise_adaln(s_cond_flat)
        
        # === Attention branch with compaction ===
        # Modulate before attention (pixel-wise modulation in pixel space)
        x_mod = pixel_modulate(self.norm1(x), shift1, scale1)
        
        # Compact pixel tokens to semantic space for efficient global attention
        # C: (B*L, p², D_pix) -> (B*L, 1, D_semantic)
        x_compact = self.compaction.compact(x_mod)
        
        # Reshape for global attention across ALL patches: (B, L, D_semantic)
        x_compact = x_compact.reshape(B, L, D_sem)
        
        # Apply self-attention across all L patches in semantic space (global context)
        x_attn = self.attn(x_compact, rope)
        
        # Reshape back: (B, L, D_semantic) -> (B*L, 1, D_semantic)
        x_attn = x_attn.reshape(B * L, 1, D_sem)
        
        # Expand back to pixel space
        # E: (B*L, 1, D_semantic) -> (B*L, p², D_pix)
        x_attn = self.compaction.decompact(x_attn)
        
        # Residual with per-pixel gating (in pixel space)
        x = x + gate1 * x_attn
        
        # === MLP branch (in pixel space) ===
        x_mod = pixel_modulate(self.norm2(x), shift2, scale2)
        x = x + gate2 * self.mlp(x_mod)
        
        # Reshape back to patch-local view: (B*L, p², D_pix) -> (B, L, p², D_pix)
        x = x.reshape(B, L, p2, D_pix)
        
        return x


# =============================================================================
# Final Layer for PixelDiT
# =============================================================================

class PixelFinalLayer(nn.Module):
    """Final layer that projects pixel tokens to RGB output.
    
    Design: Norm + AdaLN modulation + Linear projection.
    Uses semantic conditioning (like PixelwiseAdaLN) to modulate pixel features.
    """
    
    def __init__(self, semantic_dim: int, pixel_dim: int, out_channels: int = 3, use_rmsnorm: bool = True):
        super().__init__()
        if use_rmsnorm:
            self.norm_final = RMSNorm(pixel_dim)
        else:
            self.norm_final = nn.LayerNorm(pixel_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(pixel_dim, out_channels, bias=True)
        # Project from semantic_dim to pixel_dim AdaLN parameters (like PixelwiseAdaLN)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(semantic_dim, 2 * pixel_dim, bias=True)
        )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, D_pix) pixel tokens
            c: (B, D_semantic) conditioning signal from semantic pathway
        Returns:
            (B, H*W, out_channels)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift.unsqueeze(1), scale.unsqueeze(1))
        x = self.linear(x)
        return x


# =============================================================================
# PixelDiT - Class-conditioned Model
# =============================================================================

class PixelDiT(nn.Module):
    """PixelDiT: Pixel Diffusion Transformer for class-conditioned image generation.
    
    Architecture:
    1. Patch-level pathway: N DiT blocks for global semantic learning
    2. Pixel-level pathway: M PiT blocks for texture detail refinement
    
    Operates directly in pixel space without VAE/autoencoder.
    
    Key insight from paper: D_pix = 16 (much smaller than D) for efficiency.
    This enables dense per-pixel computations while keeping the model tractable.
    """
    
    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 1152,  # D: semantic dimension (XL default)
        pixel_dim: int = 16,      # D_pix: pixel dimension (FIXED at 16 per paper)
        patch_depth: int = 26,    # N: number of patch-level blocks (XL default)
        pixel_depth: int = 4,     # M: number of pixel-level blocks (XL default)
        num_heads: int = 16,
        pixel_num_heads: int = 16, # For semantic-space attention (hidden_size), same as patch-level
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.pixel_dim = pixel_dim
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        
        # Calculate dimensions
        self.num_patches = (input_size // patch_size) ** 2
        self.num_pixels = patch_size * patch_size
        
        # === Patch-level pathway embeddings ===
        # Patch embedding for semantic pathway
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding for patch tokens
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), 
            requires_grad=False
        )
        
        # Timestep and label embeddings
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        
        # RoPE for patch-level attention
        if use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.patch_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.patch_rope = None
        
        # === Pixel-level pathway embeddings ===
        # Pixel embedding (1x1 conv = linear per pixel)
        self.pixel_embed = nn.Conv2d(in_channels, pixel_dim, kernel_size=1, stride=1)
        
        # Pixel position embedding
        self.pixel_pos_embed = nn.Parameter(
            torch.zeros(1, input_size * input_size, pixel_dim),
            requires_grad=False
        )
        
        # === Patch-level blocks (N blocks) ===
        self.patch_blocks = nn.ModuleList([
            PatchDiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm,
                use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm,
                wo_shift=False,
            ) for _ in range(patch_depth)
        ])
        
        # === Pixel-level blocks (M blocks) ===
        self.pixel_blocks = nn.ModuleList([
            PixelTransformerBlock(
                pixel_dim=pixel_dim,
                semantic_dim=hidden_size,
                num_heads=pixel_num_heads,
                patch_size=patch_size,
                mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm,
                use_rmsnorm=use_rmsnorm,
            ) for _ in range(pixel_depth)
        ])
        
        # === Final layer ===
        self.final_layer = PixelFinalLayer(
            semantic_dim=hidden_size,
            pixel_dim=pixel_dim,
            out_channels=in_channels,
            use_rmsnorm=use_rmsnorm,
        )
        
        # Initialize weights
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize model weights."""
        # Basic initialization
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size, 
            int(self.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        pixel_pos_embed = get_2d_sincos_pos_embed(
            self.pixel_dim,
            self.input_size
        )
        self.pixel_pos_embed.data.copy_(torch.from_numpy(pixel_pos_embed).float().unsqueeze(0))
        
        # Initialize patch embedding
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.bias, 0)
        
        # Initialize pixel embedding
        w = self.pixel_embed.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.pixel_embed.bias, 0)
        
        # Initialize embedders
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Zero-out AdaLN layers (DiTBlock has single adaLN_modulation)
        for block in self.patch_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        for block in self.pixel_blocks:
            nn.init.constant_(block.pixelwise_adaln.mlp[-1].weight, 0)
            nn.init.constant_(block.pixelwise_adaln.mlp[-1].bias, 0)
        
        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def patchify_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reorganize pixel tokens into patch-local sequences.
        
        This is the KEY operation described in the paper:
        "To align with patch-level semantic tokens, we reshape into B·L sequences 
        of p² pixel tokens, i.e., X ∈ R^(B·L)×p²×D_pix"
        
        Args:
            x: (B, D_pix, H, W) pixel embeddings
        Returns:
            (B, L, p², D_pix) patch-local pixel tokens
            where L = (H/p) * (W/p) is number of patches
        """
        p = self.patch_size
        # (B, D, H, W) -> (B, L, p², D) where L = (H/p)*(W/p)
        return rearrange(x, 'b d (h p1) (w p2) -> b (h w) (p1 p2) d', p1=p, p2=p)
    
    def unpatchify_pixels(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Reorganize patch-local pixel tokens back to image layout.
        
        Args:
            x: (B, L, p², C) patch-local pixel tokens
            H, W: original image height and width
        Returns:
            (B, C, H, W) image tensor
        """
        p = self.patch_size
        h, w = H // p, W // p
        # (B, L, p², C) -> (B, C, H, W)
        return rearrange(x, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', h=h, w=w, p1=p, p2=p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PixelDiT.
        
        Args:
            x: (B, C, H, W) input images in pixel space
            t: (B,) diffusion timesteps
            y: (B,) class labels
        Returns:
            (B, C, H, W) predicted noise/velocity
        """
        B, C, H, W = x.shape
        p = self.patch_size
        L = (H // p) * (W // p)  # Number of patches
        
        # === Embeddings ===
        t_emb = self.t_embedder(t)                    # (B, D)
        y_emb = self.y_embedder(y, self.training)     # (B, D)
        c = t_emb + y_emb                             # (B, D) conditioning
        
        # === Patch-level pathway ===
        # Embed patches: (B, D, H/p, W/p) -> (B, L, D)
        s = self.patch_embed(x)                       # (B, D, H/p, W/p)
        s = s.flatten(2).transpose(1, 2)              # (B, L, D)
        s = s + self.pos_embed                        # Add position embedding
        
        # Apply patch-level blocks
        for block in self.patch_blocks:
            if self.use_checkpoint:
                s = checkpoint(block, s, c, self.patch_rope, use_reentrant=True)
            else:
                s = block(s, c, self.patch_rope)
        
        # Semantic conditioning: s_cond = s_N + t
        s_cond = s + t_emb.unsqueeze(1)               # (B, L, D)
        
        # === Pixel-level pathway ===
        # Embed pixels: (B, D_pix, H, W)
        p_embed = self.pixel_embed(x)                 # (B, D_pix, H, W)
        
        # KEY: Reorganize into patch-local sequences (paper Section 3.2)
        # (B, D_pix, H, W) -> (B, L, p², D_pix)
        p_tokens = self.patchify_pixels(p_embed)      # (B, L, p², D_pix)
        
        # Add position embedding (need to reshape pos_embed to match)
        # pixel_pos_embed is (1, H*W, D_pix), reshape to (1, L, p², D_pix)
        pos_embed_patches = self.pixel_pos_embed.view(1, L, p * p, self.pixel_dim)
        p_tokens = p_tokens + pos_embed_patches
        
        # Apply pixel-level blocks
        # Each block expects (B, L, p², D_pix) and outputs same shape
        for block in self.pixel_blocks:
            if self.use_checkpoint:
                p_tokens = checkpoint(block, p_tokens, s_cond, self.patch_rope, use_reentrant=True)
            else:
                p_tokens = block(p_tokens, s_cond, self.patch_rope)
        
        # === Final projection ===
        # Reshape for final layer: (B, L, p², D_pix) -> (B, L*p², D_pix)
        p_flat = p_tokens.view(B, -1, self.pixel_dim)
        p_out = self.final_layer(p_flat, c)           # (B, L*p², C)
        
        # Reshape back: (B, L*p², C) -> (B, L, p², C)
        p_out = p_out.view(B, L, p * p, C)
        
        # Unpatchify to image: (B, L, p², C) -> (B, C, H, W)
        output = self.unpatchify_pixels(p_out, H, W)
        
        return output
    
    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, 
                         cfg_scale: float) -> torch.Tensor:
        """Forward with classifier-free guidance."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        
        cond_out, uncond_out = torch.split(model_out, len(model_out) // 2, dim=0)
        half_out = uncond_out + cfg_scale * (cond_out - uncond_out)
        
        return torch.cat([half_out, half_out], dim=0)


# =============================================================================
# MM-DiT Block for Text-to-Image
# =============================================================================

class MMDiTBlock(nn.Module):
    """Multi-Modal DiT Block for text-image fusion.
    
    Following SD3's design: separate QKV projections for image and text streams,
    joint attention, then separate MLPs.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Normalization
        norm_layer = RMSNorm if use_rmsnorm else lambda d: nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        
        # Image stream
        self.norm1_img = norm_layer(hidden_size)
        self.norm2_img = norm_layer(hidden_size)
        self.qkv_img = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj_img = nn.Linear(hidden_size, hidden_size)
        
        # Text stream  
        self.norm1_txt = norm_layer(hidden_size)
        self.norm2_txt = norm_layer(hidden_size)
        self.qkv_txt = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj_txt = nn.Linear(hidden_size, hidden_size)
        
        # QK normalization
        if use_qknorm:
            self.q_norm = norm_layer(self.head_dim)
            self.k_norm = norm_layer(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        
        # MLPs
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp_img = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
            self.mlp_txt = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp_img = Mlp(hidden_size, mlp_hidden_dim, act_layer=lambda: nn.GELU(approximate="tanh"))
            self.mlp_txt = Mlp(hidden_size, mlp_hidden_dim, act_layer=lambda: nn.GELU(approximate="tanh"))
        
        # AdaLN modulation for both streams (from timestep)
        # Image stream: 6 params (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        # Text stream: 6 params (same structure)
        self.adaLN_modulation_img = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.adaLN_modulation_txt = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
    
    def forward(
        self, 
        img: torch.Tensor, 
        txt: torch.Tensor, 
        c: torch.Tensor,
        rope=None,
        txt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img: (B, L_img, D) image tokens
            txt: (B, L_txt, D) text tokens
            c: (B, D) conditioning (timestep embedding)
            rope: optional rotary position embedding
            txt_mask: (B, L_txt) optional attention mask for text (True = valid, False = padding)
        Returns:
            Tuple of (img, txt) with same shapes
        """
        B = img.shape[0]
        L_img = img.shape[1]
        L_txt = txt.shape[1]
        
        # Get AdaLN parameters for both streams
        shift_msa_img, scale_msa_img, gate_msa_img, shift_mlp_img, scale_mlp_img, gate_mlp_img = \
            self.adaLN_modulation_img(c).chunk(6, dim=1)
        shift_msa_txt, scale_msa_txt, gate_msa_txt, shift_mlp_txt, scale_mlp_txt, gate_mlp_txt = \
            self.adaLN_modulation_txt(c).chunk(6, dim=1)
        
        # === Joint Attention ===
        # Normalize and modulate both streams (unsqueeze for broadcasting: [B, D] -> [B, 1, D])
        img_norm = modulate(self.norm1_img(img), shift_msa_img.unsqueeze(1), scale_msa_img.unsqueeze(1))
        txt_norm = modulate(self.norm1_txt(txt), shift_msa_txt.unsqueeze(1), scale_msa_txt.unsqueeze(1))
        
        # Compute QKV for both streams
        qkv_img = self.qkv_img(img_norm).reshape(B, L_img, 3, self.num_heads, self.head_dim)
        qkv_txt = self.qkv_txt(txt_norm).reshape(B, L_txt, 3, self.num_heads, self.head_dim)
        
        # Concatenate for joint attention
        qkv = torch.cat([qkv_img, qkv_txt], dim=1)  # (B, L_img+L_txt, 3, H, D_h)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D_h)
        q, k, v = qkv.unbind(0)
        
        # Apply QK norm
        q, k = self.q_norm(q), self.k_norm(k)
        
        # Apply RoPE if provided (only to image tokens)
        if rope is not None:
            q_img = rope(q[:, :, :L_img, :])
            k_img = rope(k[:, :, :L_img, :])
            q = torch.cat([q_img, q[:, :, L_img:, :]], dim=2)
            k = torch.cat([k_img, k[:, :, L_img:, :]], dim=2)
        
        # Build attention mask if text_mask is provided
        # Mask shape for SDPA: (B, 1, L_total, L_total) or (B, H, L_total, L_total)
        attn_mask = None
        if txt_mask is not None:
            L_total = L_img + L_txt
            # Create combined mask: image tokens are always valid (True)
            # txt_mask: (B, L_txt) where True = valid token, False = padding
            img_mask = torch.ones(B, L_img, dtype=torch.bool, device=img.device)
            combined_mask = torch.cat([img_mask, txt_mask], dim=1)  # (B, L_total)
            
            # For attention: mask[i,j] = True means position j can attend to position i
            # We want: valid tokens can be attended, padding tokens cannot
            # Shape: (B, 1, 1, L_total) for broadcasting with (B, H, L_q, L_k)
            attn_mask = combined_mask[:, None, None, :]  # (B, 1, 1, L_total)
            # Expand to full attention matrix for keys
            # Query positions can attend to all valid key positions
            attn_mask = attn_mask.expand(-1, -1, L_total, -1)  # (B, 1, L_total, L_total)
        
        # Scaled dot-product attention with optional mask
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, L_img + L_txt, self.hidden_size)
        
        # Split back to streams
        attn_img = attn_out[:, :L_img, :]
        attn_txt = attn_out[:, L_img:, :]
        
        # Project and residual (both streams with gating)
        img = img + gate_msa_img.unsqueeze(1) * self.proj_img(attn_img)
        txt = txt + gate_msa_txt.unsqueeze(1) * self.proj_txt(attn_txt)
        
        # === MLPs (both streams with AdaLN modulation, unsqueeze for broadcasting) ===
        img = img + gate_mlp_img.unsqueeze(1) * self.mlp_img(modulate(self.norm2_img(img), shift_mlp_img.unsqueeze(1), scale_mlp_img.unsqueeze(1)))
        txt = txt + gate_mlp_txt.unsqueeze(1) * self.mlp_txt(modulate(self.norm2_txt(txt), shift_mlp_txt.unsqueeze(1), scale_mlp_txt.unsqueeze(1)))
        
        return img, txt


# =============================================================================
# PixelDiT_T2I - Text-to-Image Model
# =============================================================================

class PixelDiT_T2I(nn.Module):
    """PixelDiT for Text-to-Image generation.
    
    Uses MM-DiT blocks in the patch-level pathway for text-image fusion.
    Pixel-level pathway remains unchanged.
    
    From Table 7: D=1536, N=14, M=2, patch_size=16, 1311M params
    D_pix = 16 (fixed for all PixelDiT models)
    """
    
    def __init__(
        self,
        input_size: int = 1024,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 1536,   # D: semantic dimension
        pixel_dim: int = 16,       # D_pix: pixel dimension (FIXED at 16 per paper)
        patch_depth: int = 14,     # N: number of patch-level MM-DiT blocks
        pixel_depth: int = 2,      # M: number of pixel-level PiT blocks
        num_heads: int = 24,
        pixel_num_heads: int = 24,  # For semantic-space attention (hidden_size), same as patch-level
        mlp_ratio: float = 4.0,
        text_hidden_size: int = 2304,  # Gemma-2 hidden size
        max_text_len: int = 256,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.pixel_dim = pixel_dim
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.max_text_len = max_text_len
        
        # Calculate dimensions
        self.num_patches = (input_size // patch_size) ** 2
        self.num_pixels = patch_size * patch_size
        
        # === Patch-level pathway embeddings ===
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size),
            requires_grad=False
        )
        
        # Text embedding projection
        self.text_proj = nn.Linear(text_hidden_size, hidden_size)
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # RoPE
        if use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.patch_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.patch_rope = None
        
        # === Pixel-level pathway embeddings ===
        self.pixel_embed = nn.Conv2d(in_channels, pixel_dim, kernel_size=1, stride=1)
        self.pixel_pos_embed = nn.Parameter(
            torch.zeros(1, input_size * input_size, pixel_dim),
            requires_grad=False
        )
        
        # === Patch-level MM-DiT blocks ===
        self.patch_blocks = nn.ModuleList([
            MMDiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm,
                use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm,
            ) for _ in range(patch_depth)
        ])
        
        # === Pixel-level PiT blocks ===
        self.pixel_blocks = nn.ModuleList([
            PixelTransformerBlock(
                pixel_dim=pixel_dim,
                semantic_dim=hidden_size,
                num_heads=pixel_num_heads,
                patch_size=patch_size,
                mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm,
                use_rmsnorm=use_rmsnorm,
            ) for _ in range(pixel_depth)
        ])
        
        # === Final layer ===
        self.final_layer = PixelFinalLayer(
            semantic_dim=hidden_size,
            pixel_dim=pixel_dim,
            out_channels=in_channels,
            use_rmsnorm=use_rmsnorm,
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize model weights."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            int(self.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        pixel_pos_embed = get_2d_sincos_pos_embed(
            self.pixel_dim,
            self.input_size
        )
        self.pixel_pos_embed.data.copy_(torch.from_numpy(pixel_pos_embed).float().unsqueeze(0))
        
        # Patch/pixel embedding initialization
        for embed in [self.patch_embed, self.pixel_embed]:
            w = embed.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(embed.bias, 0)
        
        # Timestep embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Zero-out AdaLN layers (MMDiTBlock has separate modulation for img and txt streams)
        for block in self.patch_blocks:
            nn.init.constant_(block.adaLN_modulation_img[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation_img[-1].bias, 0)
            nn.init.constant_(block.adaLN_modulation_txt[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation_txt[-1].bias, 0)
        
        for block in self.pixel_blocks:
            nn.init.constant_(block.pixelwise_adaln.mlp[-1].weight, 0)
            nn.init.constant_(block.pixelwise_adaln.mlp[-1].bias, 0)
        
        # Final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def patchify_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reorganize pixel tokens into patch-local sequences.
        
        Args:
            x: (B, D_pix, H, W) pixel embeddings
        Returns:
            (B, L, p², D_pix) patch-local pixel tokens
        """
        p = self.patch_size
        # (B, D, H, W) -> (B, L, p², D) where L = (H/p)*(W/p)
        return rearrange(x, 'b d (h p1) (w p2) -> b (h w) (p1 p2) d', p1=p, p2=p)
    
    def unpatchify_pixels(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Reorganize patch-local pixel tokens back to image layout.
        
        Args:
            x: (B, L, p², C) patch-local pixel tokens
            H, W: original image height and width
        Returns:
            (B, C, H, W) image tensor
        """
        p = self.patch_size
        h, w = H // p, W // p
        # (B, L, p², C) -> (B, C, H, W)
        return rearrange(x, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', h=h, w=w, p1=p, p2=p)

    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        text_embeds: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of PixelDiT-T2I.
        
        Args:
            x: (B, C, H, W) input images in pixel space
            t: (B,) diffusion timesteps
            text_embeds: (B, L_txt, D_txt) text embeddings from Gemma-2
            text_mask: (B, L_txt) optional attention mask for text
        Returns:
            (B, C, H, W) predicted noise/velocity
        """
        B, C, H, W = x.shape
        p = self.patch_size
        L = (H // p) * (W // p)  # Number of patches
        
        # === Embeddings ===
        t_emb = self.t_embedder(t)  # (B, D)
        
        # Project text embeddings to model dimension
        txt = self.text_proj(text_embeds)  # (B, L_txt, D)
        
        # === Patch-level pathway ===
        s = self.patch_embed(x)  # (B, D, H/p, W/p)
        s = s.flatten(2).transpose(1, 2)  # (B, L, D)
        s = s + self.pos_embed
        
        # Apply MM-DiT blocks with text mask
        for block in self.patch_blocks:
            if self.use_checkpoint:
                s, txt = checkpoint(block, s, txt, t_emb, self.patch_rope, text_mask, use_reentrant=True)
            else:
                s, txt = block(s, txt, t_emb, self.patch_rope, text_mask)
        
        # Semantic conditioning: s_cond = s_N + t
        s_cond = s + t_emb.unsqueeze(1)  # (B, L, D)
        
        # === Pixel-level pathway ===
        # Embed pixels: (B, D_pix, H, W)
        p_embed = self.pixel_embed(x)  # (B, D_pix, H, W)
        
        # KEY: Reorganize into patch-local sequences (paper Section 3.2)
        # (B, D_pix, H, W) -> (B, L, p², D_pix)
        p_tokens = self.patchify_pixels(p_embed)  # (B, L, p², D_pix)
        
        # Add position embedding (reshape to match)
        pos_embed_patches = self.pixel_pos_embed.view(1, L, p * p, self.pixel_dim)
        p_tokens = p_tokens + pos_embed_patches
        
        # Apply pixel-level blocks
        for block in self.pixel_blocks:
            if self.use_checkpoint:
                p_tokens = checkpoint(block, p_tokens, s_cond, use_reentrant=True)
            else:
                p_tokens = block(p_tokens, s_cond)
        
        # === Final projection ===
        # Reshape for final layer: (B, L, p², D_pix) -> (B, L*p², D_pix)
        p_flat = p_tokens.view(B, -1, self.pixel_dim)
        p_out = self.final_layer(p_flat, t_emb)  # (B, L*p², C)
        
        # Reshape back: (B, L*p², C) -> (B, L, p², C)
        p_out = p_out.view(B, L, p * p, C)
        
        # Unpatchify to image: (B, L, p², C) -> (B, C, H, W)
        output = self.unpatchify_pixels(p_out, H, W)
        
        return output
    
    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embeds: torch.Tensor,
        cfg_scale: float,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward with classifier-free guidance."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, text_embeds, text_mask)
        
        cond_out, uncond_out = torch.split(model_out, len(model_out) // 2, dim=0)
        half_out = uncond_out + cfg_scale * (cond_out - uncond_out)
        
        return torch.cat([half_out, half_out], dim=0)


# =============================================================================
# Positional Embedding Utilities
# =============================================================================

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """Generate 2D sinusoidal positional embedding."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Generate 2D sinusoidal positional embedding from grid."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D sinusoidal positional embedding from positions."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


# =============================================================================
# Model Configurations (from Table 2 in the paper)
# =============================================================================
# Key insight: D_pix = 16 for ALL models (much smaller than D for efficiency)
# This enables dense per-pixel computations to remain highly efficient.

def PixelDiT_B(input_size: int = 256, **kwargs):
    """PixelDiT-Base: 184M params (Table 2)
    
    Config: N=12, M=2, D=768, D_pix=16, Heads=12
    """
    return PixelDiT(
        input_size=input_size,
        hidden_size=768,       # D
        pixel_dim=16,          # D_pix (fixed at 16 for all models)
        patch_depth=12,        # N
        pixel_depth=2,         # M
        num_heads=12,          # Attention heads for patch-level
        pixel_num_heads=12,    # Same as num_heads (attention in semantic space D=768)
        **kwargs
    )

def PixelDiT_L(input_size: int = 256, **kwargs):
    """PixelDiT-Large: 569M params (Table 2)
    
    Config: N=22, M=4, D=1024, D_pix=16, Heads=16
    """
    return PixelDiT(
        input_size=input_size,
        hidden_size=1024,      # D
        pixel_dim=16,          # D_pix (fixed at 16)
        patch_depth=22,        # N
        pixel_depth=4,         # M
        num_heads=16,          # Attention heads for patch-level
        pixel_num_heads=16,    # Same as num_heads (attention in semantic space D=1024)
        **kwargs
    )

def PixelDiT_XL(input_size: int = 256, **kwargs):
    """PixelDiT-XL: 797M params (Table 2)
    
    Config: N=26, M=4, D=1152, D_pix=16, Heads=16
    """
    return PixelDiT(
        input_size=input_size,
        hidden_size=1152,      # D
        pixel_dim=16,          # D_pix (fixed at 16)
        patch_depth=26,        # N
        pixel_depth=4,         # M
        num_heads=16,          # Attention heads for patch-level
        pixel_num_heads=16,    # Same as num_heads (attention in semantic space D=1152)
        **kwargs
    )

def PixelDiT_T2I_1B(input_size: int = 1024, **kwargs):
    """PixelDiT-T2I: 1311M params for text-to-image generation (Table 7)
    
    Config: N=14, M=2, D=1536, D_pix=16, patch_size=16
    """
    return PixelDiT_T2I(
        input_size=input_size,
        hidden_size=1536,      # D
        pixel_dim=16,          # D_pix (fixed at 16)
        patch_depth=14,        # N
        pixel_depth=2,         # M
        num_heads=24,          # Attention heads for patch-level
        pixel_num_heads=24,    # Same as num_heads (attention in semantic space D=1536)
        **kwargs
    )


# Model registry
PixelDiT_models = {
    'PixelDiT-B': PixelDiT_B,
    'PixelDiT-L': PixelDiT_L,
    'PixelDiT-XL': PixelDiT_XL,
    'PixelDiT-T2I-1B': PixelDiT_T2I_1B,
}


# =============================================================================
# Test Utilities
# =============================================================================

def test_model(
    name: str,
    input_size: int = 256,
    hidden_size: int = 768,
    pixel_dim: int = 16,
    patch_depth: int = 12,
    pixel_depth: int = 2,
    num_heads: int = 12,
    target_params: Optional[int] = None,
    is_t2i: bool = False,
    text_hidden_size: int = 768,
    batch_size: int = 2,
):
    """Test a PixelDiT model with given parameters.
    
    Args:
        name: Model name for display
        input_size: Image size (default: 256)
        hidden_size: Semantic dimension D
        pixel_dim: Pixel dimension D_pix (default: 16)
        patch_depth: Number of patch-level blocks N
        pixel_depth: Number of pixel-level blocks M  
        num_heads: Number of attention heads
        target_params: Expected parameter count in millions (optional)
        is_t2i: Whether this is a text-to-image model
        text_hidden_size: Text embedding dimension (for T2I)
        batch_size: Batch size for testing
    """
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")
    print(f"Config: N={patch_depth}, M={pixel_depth}, D={hidden_size}, D_pix={pixel_dim}, Heads={num_heads}")
    
    # Create model
    if is_t2i:
        model = PixelDiT_T2I(
            input_size=input_size,
            hidden_size=hidden_size,
            pixel_dim=pixel_dim,
            patch_depth=patch_depth,
            pixel_depth=pixel_depth,
            num_heads=num_heads,
            pixel_num_heads=num_heads,
            text_hidden_size=text_hidden_size,
        )
    else:
        model = PixelDiT(
            input_size=input_size,
            hidden_size=hidden_size,
            pixel_dim=pixel_dim,
            patch_depth=patch_depth,
            pixel_depth=pixel_depth,
            num_heads=num_heads,
            pixel_num_heads=num_heads,
            num_classes=1000,
        )
    
    # Count parameters
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    if target_params:
        diff = (params_m - target_params) / target_params * 100
        print(f"Parameters: {params_m:.2f}M (target: {target_params}M, diff: {diff:+.1f}%)")
    else:
        print(f"Parameters: {params_m:.2f}M")
    
    # Test forward pass
    x = torch.randn(batch_size, 3, input_size, input_size)
    t = torch.randint(0, 1000, (batch_size,))
    
    with torch.no_grad():
        if is_t2i:
            text_embeds = torch.randn(batch_size, 77, text_hidden_size)
            out = model(x, t, text_embeds)
        else:
            y = torch.randint(0, 1000, (batch_size,))
            out = model(x, t, y)
    
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print(f"Forward pass: {x.shape} -> {out.shape} ✓")
    
    return params_m


if __name__ == '__main__':
    print("PixelDiT Model Test Suite")
    
    # Test class-conditioned models
    test_model("PixelDiT-B", hidden_size=768, patch_depth=12, pixel_depth=2, num_heads=12, target_params=184)
    test_model("PixelDiT-L", hidden_size=1024, patch_depth=22, pixel_depth=4, num_heads=16, target_params=569)
    test_model("PixelDiT-XL", hidden_size=1152, patch_depth=26, pixel_depth=4, num_heads=16, target_params=797)
    
    # Test T2I model (smaller config for speed)
    test_model("PixelDiT-T2I (test)", hidden_size=768, patch_depth=4, pixel_depth=2, num_heads=12, is_t2i=True)
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
