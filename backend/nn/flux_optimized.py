"""
backend/nn/flux_optimized.py
Optimized flux backend with tiling and normal mode with choice in UI
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from typing import Optional, Tuple, Dict, List
import gc
from collections import OrderedDict
import threading
from modules.shared import cmd_opts

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


try:
    from backend.attention import attention_function
    from backend.utils import fp16_fix, tensor2parameter
except ImportError:
    # Fallback if the backend is not directly accessible
    attention_function = None
    fp16_fix = lambda x: x
    tensor2parameter = lambda x: x


try:
    import modules.shared as shared
except ImportError:
    # Fallback for offline testing
    class FakeOpts:
        flux_tile_size = 512
        flux_tile_overlap = 128
        flux_tiling_mode = "normal"
    
    class FakeShared:
        opts = FakeOpts()
    
    shared = FakeShared()


# TILING CONFIGURATION

class TilingMode:
    AUTO = "auto"      # Automatic selection by VRAM
    NORMAL = "normal"  # Standard generation without tiling
    TILED = "tiled"    # Forced tiling

class TuringConfig:
    """Tiling settings"""
    
    #Default values ​​(fallback)
    DEFAULT_TILE_SIZE = 512
    DEFAULT_TILE_OVERLAP = 128
    DEFAULT_TILING_MODE = "normal"
    CLEAR_CACHE_FREQ = 1
    USE_FP16 = True
    
    @classmethod
    def get_tile_size(cls):
        """Reads the current value from the UI"""
        try:
            return shared.opts.flux_tile_size
        except AttributeError:
            return cls.DEFAULT_TILE_SIZE
    
    @classmethod
    def get_tile_overlap(cls):
        try:
            return shared.opts.flux_tile_overlap
        except AttributeError:
            return cls.DEFAULT_TILE_OVERLAP
    
    @classmethod
    def get_tiling_mode(cls):
        try:
            return shared.opts.flux_tiling_mode
        except AttributeError:
            return cls.DEFAULT_TILING_MODE


# VRAM MONITOR


class VRAMMonitor:
    """Real-time VRAM monitoring with auto-detect"""
    
    def __init__(self):
        self.device = torch.device('cuda')
        self.peak_allocated = 0
        # Autodetect total VRAM size
        self.vram_total = torch.cuda.get_device_properties(self.device).total_memory
        
    def get_free(self):
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(self.device)
        self.peak_allocated = max(self.peak_allocated, allocated)
        return self.vram_total - allocated
    
    def is_critical(self, threshold_mb: int = 500):
        return self.get_free() < (threshold_mb * 1024**2)
    
    def emergency_cleanup(self):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
    
    def get_stats(self):
        """Returns statistics for debugging"""
        free = self.get_free()
        allocated = torch.cuda.memory_allocated(self.device)
        return {
            'total_mb': self.vram_total / (1024**2),
            'allocated_mb': allocated / (1024**2),
            'free_mb': free / (1024**2),
            'peak_mb': self.peak_allocated / (1024**2),
        }


_vram_monitor = VRAMMonitor()


def is_flux_tools_active():
    return getattr(shared.opts, 'flux_tools_enabled', False)


# LRU CACHE TO RAM

class ActivationCache:
    """Global RAM activation cache"""
    
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        
    def get(self, key: str) -> Optional[torch.Tensor]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, tensor: torch.Tensor):
        with self.lock:
            size = tensor.numel() * tensor.element_size()
            
            while self.current_bytes + size > self.max_bytes and self.cache:
                oldest_key, oldest_tensor = self.cache.popitem(last=False)
                self.current_bytes -= oldest_tensor.numel() * oldest_tensor.element_size()
                del oldest_tensor
            
            # Transfer to CPU and save
            cpu_tensor = tensor.cpu()
            self.cache[key] = cpu_tensor
            self.current_bytes += size
            self.cache.move_to_end(key)
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.current_bytes = 0
            gc.collect()


# Global cache
_global_cache = ActivationCache(max_bytes=512 * 1024**2)


# ORIGINAL FUNCTIONS (with optimizations)

def rope(pos, dim, theta):
    """Optimized RoPE - original code, but with FP32 calculations"""
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    
    out = pos.unsqueeze(-1) * omega.unsqueeze(0)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.view(*out.shape[:-1], 2, 2)


def apply_rope(xq, xk, freqs_cis):
    """Using RoPE with Turing Optimization"""
    dtype = xq.dtype
    
    # Turing FP32 computing is more stable and not much slower.
    xq_ = xq.to(torch.float32).view(*xq.shape[:-1], -1, 2).unsqueeze(-2)
    xk_ = xk.to(torch.float32).view(*xk.shape[:-1], -1, 2).unsqueeze(-2)
    
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    
    return xq_out.view_as(xq).to(dtype), xk_out.view_as(xk).to(dtype)


def attention(q, k, v, pe, use_chunked: bool = False):
    """
    Optimized attention with fallback to chunked version
    """
    q, k = apply_rope(q, k, pe)
    
    # If there is an original function from the backend and chunked is not required
    if attention_function is not None and not use_chunked:
        try:
            return attention_function(q, k, v, q.size(1), skip_reshape=True)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            pass  # Fallback on chunked
    
    # Trying standard SDPA
    try:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        pass
    
    # Chunked fallback
    return attention_chunked(q, k, v)


def attention_chunked(q, k, v, chunk_size: int = 256):
    """Chunked attention to save VRAM"""
    B, H, L, D = q.shape
    output = torch.empty_like(q)
    
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        q_chunk = q[:, :, start:end, :]
        
        scores = torch.matmul(q_chunk, k.transpose(-2, -1)) / math.sqrt(D)
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out_chunk = torch.matmul(attn, v)
        
        output[:, :, start:end, :] = out_chunk
        
        del scores, attn, out_chunk
        
        if start % (chunk_size * 2) == 0:
            torch.cuda.empty_cache()
    
    return output


def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
    """Original function unchanged"""
    t = t.float() * time_factor
    half = dim // 2
    
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None, :]
    emb = torch.cat((args.cos(), args.sin()), dim=-1)
    
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    
    return emb.to(t.dtype)



class EmbedND(nn.Module):
    """Original class"""
    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    """Original class"""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        x = self.silu(self.in_layer(x))
        return self.out_layer(x)


# Optimized RMSNorm with in-place operations
if hasattr(torch, 'rms_norm'):
    functional_rms_norm = torch.rms_norm
else:
    def functional_rms_norm(x: Tensor, normalized_shape: int, weight: Tensor, eps: float) -> Tensor:
        if x.dtype == torch.float16:
            norm = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        else:
            x32 = x.to(torch.float32)
            norm = torch.rsqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + eps).to(x.dtype)
        return x * (norm * weight)


class RMSNorm(nn.Module):
    """Optimized RMSNorm with offloading support"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.normalized_shape = (dim,) if isinstance(dim, int) else dim

    def forward(self, x: Tensor) -> Tensor:
        scale = self.scale
        if scale.dtype != x.dtype or scale.device != x.device:
            scale = scale.to(dtype=x.dtype, device=x.device)
        return functional_rms_norm(x, self.normalized_shape, scale, self.eps)


class QKNorm(nn.Module):
    """Original class"""
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        qn = self.query_norm(q)
        kn = self.key_norm(k)
        
        if kn.dtype != qn.dtype or kn.device != qn.device:
            kn = kn.to(dtype=qn.dtype, device=qn.device)
        
        return qn, kn


class SelfAttention(nn.Module):
    """SelfAttention with offloading support"""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor, vram_monitor: Optional[VRAMMonitor] = None) -> Tensor:
        B, L, _ = x.shape

        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        q, k = self.norm(q, k, v)

        use_chunked = vram_monitor.is_critical() if vram_monitor else False
        attn_out = attention(q, k, v, pe, use_chunked=use_chunked)

        return self.proj(attn_out)


class Modulation(nn.Module):
    """Original class"""
    def __init__(self, dim, double):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec):
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return out


class DoubleStreamBlock(nn.Module):
    """
    DoubleStreamBlock with aggressive offloading
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio, qkv_bias=False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        # Image side
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        
        # Text side
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img, txt, vec, pe, vram_monitor: Optional[VRAMMonitor] = None, 
                offload_after: bool = False, cache_key: Optional[str] = None):
        # Image modulation
        img_mod1_shift, img_mod1_scale, img_mod1_gate, \
        img_mod2_shift, img_mod2_scale, img_mod2_gate = self.img_mod(vec)

        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1_scale) * img_modulated + img_mod1_shift
        img_qkv = self.img_attn.qkv(img_modulated)

        B, L, _ = img_qkv.shape
        H = self.num_heads
        D = img_qkv.shape[-1] // (3 * H)
        img_q, img_k, img_v = img_qkv.view(B, L, 3, H, D).permute(2, 0, 3, 1, 4)

        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # Text modulation
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, \
        txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = self.txt_mod(vec)

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1_scale) * txt_modulated + txt_mod1_shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)

        B, L, _ = txt_qkv.shape
        txt_q, txt_k, txt_v = txt_qkv.view(B, L, 3, H, D).permute(2, 0, 3, 1, 4)

        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # Concatenate
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        # Attention
        use_chunked = vram_monitor.is_critical() if vram_monitor else False
        attn = attention(q, k, v, pe, use_chunked=use_chunked)
        txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]

        # Residuals
        img = img + img_mod1_gate * self.img_attn.proj(img_attn)
        img = img + img_mod2_gate * self.img_mlp((1 + img_mod2_scale) * self.img_norm2(img) + img_mod2_shift)

        txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2_gate * self.txt_mlp((1 + txt_mod2_scale) * self.txt_norm2(txt) + txt_mod2_shift)

        txt = fp16_fix(txt)
        
        # Offloading if needed
        if offload_after:
            self.to_cpu()

        return img, txt
    
    def to_cpu(self):
        """Transferring weights to the CPU"""
        for module in [self.img_attn, self.txt_attn, self.img_mlp, self.txt_mlp]:
            module.to('cpu')
        torch.cuda.empty_cache()
    
    def to_gpu(self):
        """Weights back to GPU"""
        for module in [self.img_attn, self.txt_attn, self.img_mlp, self.txt_mlp]:
            module.to('cuda')


class SingleStreamBlock(nn.Module):
    """SingleStreamBlock with offloading"""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_scale=None):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(head_dim)
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x, vec, pe, vram_monitor: Optional[VRAMMonitor] = None,
                offload_after: bool = False, cache_key: Optional[str] = None):
        mod_shift, mod_scale, mod_gate = self.modulation(vec)
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift
        
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        qkv = qkv.view(qkv.size(0), qkv.size(1), 3, self.num_heads, self.hidden_size // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        q, k = self.norm(q, k, v)
        
        use_chunked = vram_monitor.is_critical() if vram_monitor else False
        attn = attention(q, k, v, pe, use_chunked=use_chunked)
        
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=2))
        x = x + mod_gate * output

        x = fp16_fix(x)
        
        if offload_after:
            self.to_cpu()

        return x
    
    def to_cpu(self):
        for module in [self.linear1, self.linear2, self.norm]:
            module.to('cpu')
        torch.cuda.empty_cache()
    
    def to_gpu(self):
        for module in [self.linear1, self.linear2, self.norm]:
            module.to('cuda')


class LastLayer(nn.Module):
    """Original class"""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


# TILLING

class FluxTiler:
    """Tiling for high resolutions without loss of quality"""
    
    def __init__(self, tile_size: int = 128, overlap: int = 32):
        self.tile_size = tile_size  # in px
        self.overlap = overlap
        self.patch_size = 2
        self.effective_tile = tile_size - overlap
        
    def split(self, img: torch.Tensor) -> Tuple[List[torch.Tensor], List[Tuple[int, int]], Tuple[int, int]]:
        """Splits the image into tiles"""
        b, c, h, w = img.shape
        
        # Calculating the number of tiles
        n_tiles_h = max(1, math.ceil((h - self.overlap) / self.effective_tile))
        n_tiles_w = max(1, math.ceil((w - self.overlap) / self.effective_tile))
        
        tiles = []
        positions = []
        
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                y_start = min(i * self.effective_tile, max(0, h - self.tile_size))
                x_start = min(j * self.effective_tile, max(0, w - self.tile_size))
                
                tile = img[:, :, y_start:y_start + self.tile_size, x_start:x_start + self.tile_size]
                tiles.append(tile)
                positions.append((y_start, x_start))
        
        return tiles, positions, (h, w)
    
    def merge(self, tiles: List[torch.Tensor], positions: List[Tuple[int, int]], 
             original_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """Merge tilees with feathered blending"""
        b, c, h, w = original_shape
        result = torch.zeros(original_shape, device='cpu', dtype=tiles[0].dtype)
        weights = torch.zeros((b, 1, h, w), device='cpu', dtype=tiles[0].dtype)
        
        # Determine the maximum coordinates for determining the edges
        max_y = max(pos[0] for pos in positions)
        max_x = max(pos[1] for pos in positions)
        
        for tile, (y, x) in zip(tiles, positions):
            _, _, th, tw = tile.shape
            
            # Determine whether the tile is the outermost one
            is_top = (y == 0)
            is_bottom = (y == max_y)
            is_left = (x == 0)
            is_right = (x == max_x)
            
            # Create a mask taking into account the position
            weight = self._create_blend_mask(
                th, tw, tile.dtype, tile.device,
                is_left_edge=is_left,
                is_right_edge=is_right,
                is_top_edge=is_top,
                is_bottom_edge=is_bottom
            )
            
            weight_cpu = weight.cpu()
            tile_cpu = tile.cpu()
            
            result[:, :, y:y+th, x:x+tw] += tile_cpu * weight_cpu
            weights[:, :, y:y+th, x:x+tw] += weight_cpu
        
        # Normalize
        result = result / (weights + 1e-8)
        
        return result.cuda() if torch.cuda.is_available() else result
    
    def _create_blend_mask(self, h: int, w: int, dtype, device, 
                           is_left_edge=False, is_right_edge=False, 
                           is_top_edge=False, is_bottom_edge=False):
        """Creates a mask for a smooth transition based on the tile's position."""
        weight = torch.ones((1, 1, h, w), dtype=dtype, device=device)
        fade = min(self.overlap // 2, h // 4, w // 4)
        
        if fade > 0:
            # Left border (fade only if not left edge of image)
            if not is_left_edge:
                weight[:, :, :, :fade] *= torch.linspace(0, 1, fade, device=device)[None, None, None, :]
            # Right border (fade only if not right edge of image)
            if not is_right_edge:
                weight[:, :, :, -fade:] *= torch.linspace(1, 0, fade, device=device)[None, None, None, :]
            # Top border (fade only if not the top edge of the image)
            if not is_top_edge:
                weight[:, :, :fade, :] *= torch.linspace(0, 1, fade, device=device)[None, None, :, None]
            # Bottom border (fade only if not bottom edge of image)
            if not is_bottom_edge:
                weight[:, :, -fade:, :] *= torch.linspace(1, 0, fade, device=device)[None, None, :, None]
        
        return weight


# MAIN MODEL

class IntegratedFluxTransformer2DModel(nn.Module):
    """
    Drop-in replacement for the original IntegratedFluxTransformer2DModel
    with optimizations for 6GB VRAM and support for all resolutions
    """
    
    def __init__(self, in_channels: int, vec_in_dim: int, context_in_dim: int, 
                 hidden_size: int, mlp_ratio: float, num_heads: int, 
                 depth: int, depth_single_blocks: int, axes_dim: list[int], 
                 theta: int, qkv_bias: bool, guidance_embed: bool):
        super().__init__()

        self.guidance_embed = guidance_embed
        self.in_channels = in_channels * 4
        self.out_channels = self.in_channels

        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")

        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Embeddings
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(vec_in_dim, self.hidden_size)
        self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if guidance_embed else nn.Identity()
        self.txt_in = nn.Linear(context_in_dim, self.hidden_size)

        # Blocks
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
            for _ in range(depth)
        ])

        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth_single_blocks)
        ])

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
 
        # State
        self._initialized = False

    def _move_all_to_cpu(self):
        """Initialization: all on CPU"""
        for module in [self.img_in, self.time_in, self.vector_in, self.guidance_in, self.txt_in]:
            module.to('cpu')
        for block in self.double_blocks:
            block.to_cpu()
        for block in self.single_blocks:
            block.to_cpu()
        self.final_layer.to('cpu')
        torch.cuda.empty_cache()

    def inner_forward(self, img, img_ids, txt, txt_ids, timesteps, y, guidance=None):
        """Forward with aggressive layer offloading"""
        
        # Initial projections
        self.img_in.to('cuda')
        img = self.img_in(img.to('cuda'))
        self.img_in.to('cpu')
        
        # Time embeddings (CPU)
        self.time_in.to('cpu')
        t_emb = timestep_embedding(timesteps, 256)
        vec = self.time_in(t_emb.cpu())
        
        if self.guidance_embed and guidance is not None:
            self.guidance_in.to('cpu')
            g_emb = timestep_embedding(guidance, 256)
            vec = vec + self.guidance_in(g_emb.cpu())
            
        self.vector_in.to('cpu')
        vec = vec + self.vector_in(y.cpu())
        vec = vec.to('cuda')
        
        # Text
        self.txt_in.to('cpu')
        txt = self.txt_in(txt.cpu())
        txt = txt.to('cuda')
        
        # PE
        ids = torch.cat((txt_ids, img_ids), dim=1)
        self.pe_embedder.to('cuda')
        pe = self.pe_embedder(ids)
        
        # Double blocks with offloading - NO CACHING
        for i, block in enumerate(self.double_blocks):
            block.to_gpu()
            torch.cuda.synchronize()
            
            # VRAM Check
            if _vram_monitor.is_critical(300):
                _vram_monitor.emergency_cleanup()
            
            # Just offload, no cache
            img, txt = block(img, txt, vec, pe, _vram_monitor, offload_after=True)
            
            # Cleaning every block
            torch.cuda.empty_cache()
            gc.collect()
        
        # Concatenate — WITHOUT restoring from cache
        img = torch.cat((txt, img), dim=1)
        del txt  # explicitly remove text features
        
        # Single blocks - NO CACHING
        for i, block in enumerate(self.single_blocks):
            block.to_gpu()
            torch.cuda.synchronize()
            
            if _vram_monitor.is_critical(300):
                _vram_monitor.emergency_cleanup()
            
            img = block(img, vec, pe, _vram_monitor, offload_after=True)
            
            torch.cuda.empty_cache()
            gc.collect()
        
        # Final layer
        self.final_layer.to('cuda')

        # Get the length of txt from txt_ids
        txt_len = txt_ids.shape[1]
        img_out = img[:, txt_len:, ...]  # we take only the image part
        
        img = self.final_layer(img_out, vec)
        
        return img

    def forward(self, x, timestep, context, y, guidance=None, 
                tiling_mode: str = None, image_height: int = None, 
                image_width: int = None, **kwargs):
        """
        Args:
            tiling_mode: "auto", "normal", or "tiled" (from UI dropdown)
            image_height, image_width: dimensions for tiling (from UI)
        """
        if is_flux_tools_active():
            tiling_mode = "normal"
        
        with torch.no_grad():
            if getattr(shared.opts, 'flux_tools_enabled', False):
                tiling_mode = "normal"
                print("[FluxOptimized] Flux Tools active: forcing NORMAL mode")
            else:
                tiling_mode = tiling_mode or TuringConfig.get_tiling_mode()
            bs, c, h, w = x.shape
            input_dtype = x.dtype
            
            # Read from the UI via TuringConfig methods
            mode = tiling_mode or TuringConfig.get_tiling_mode()
            tile_size = TuringConfig.get_tile_size()
            tile_overlap = TuringConfig.get_tile_overlap()
            
            # Create tiler with actual parameters from UI
            self.tiler = FluxTiler(tile_size=tile_size, overlap=tile_overlap)
            
            # Image sizes for logs
            img_h = image_height or h * 8
            img_w = image_width or w * 8
            
            # NORMAL MODE: always standard
            if mode == "normal":
                print(f"[FluxOptimized] Mode: NORMAL | {img_h}x{img_w}")
                return self._forward_standard(x, timestep, context, y, guidance, input_dtype)
            
            # TILED MODE: always tiling
            if mode == "tiled":
                print(f"[FluxOptimized] Mode: TILED | {img_h}x{img_w} (tile={tile_size}, overlap={tile_overlap})")
                return self._forward_tiled(x, timestep, context, y, guidance, input_dtype)
            
            # AUTO MODE: VRAM selection
            estimated_vram = (bs * (h//2) * (w//2) * self.hidden_size * 4 * 4)
            vram_usable = _vram_monitor.vram_total * 0.8
            
            use_tiling = estimated_vram > vram_usable * 0.6
            
            if use_tiling:
                print(f"[FluxOptimized] Mode: AUTO -> TILED | {img_h}x{img_w} "
                      f"(est:{estimated_vram/1024**3:.1f}GB > limit:{vram_usable*0.6/1024**3:.1f}GB) "
                      f"tile={tile_size}, overlap={tile_overlap}")
                return self._forward_tiled(x, timestep, context, y, guidance, input_dtype)
            else:
                print(f"[FluxOptimized] Mode: AUTO -> NORMAL | {img_h}x{img_w} "
                      f"(est:{estimated_vram/1024**3:.1f}GB <= limit:{vram_usable*0.6/1024**3:.1f}GB)")
                return self._forward_standard(x, timestep, context, y, guidance, input_dtype)
        
    def _forward_standard(self, x, timestep, context, y, guidance, input_dtype):
        """Standard forward WITHOUT offloading"""
        
        # moving model to GPU
        if next(self.parameters()).device.type != 'cuda':
            print("[FluxOptimized] Moving model to GPU for NORMAL mode...")
            self.cuda()
            torch.cuda.synchronize()
        
        bs, c, h, w = x.shape
        input_device = x.device
        
        patch_size = 2
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="circular")
        
        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        
        h_len = (h + pad_h) // patch_size
        w_len = (w + pad_w) // patch_size
        
        img_ids = torch.zeros((h_len, w_len, 3), device=input_device, dtype=input_dtype)
        img_ids[..., 1] = torch.linspace(0, h_len - 1, steps=h_len, device=input_device, dtype=input_dtype)[:, None]
        img_ids[..., 2] = torch.linspace(0, w_len - 1, steps=w_len, device=input_device, dtype=input_dtype)[None, :]
        img_ids = img_ids.expand(bs, -1, -1, -1).reshape(bs, h_len * w_len, 3)
        
        txt_ids = torch.zeros((bs, context.shape[1], 3), device=input_device, dtype=input_dtype)
        
        # Original forward without offloading
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timestep, 256).to(img.dtype))
        
        if self.guidance_embed and guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
        
        vec = vec + self.vector_in(y)
        txt = self.txt_in(context)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
        
        img = torch.cat((txt, img), dim=1)
        
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        
        img = img[:, txt.shape[1]:, ...]
        img = self.final_layer(img, vec)
        
        out = rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", 
                       h=h_len, w=w_len, ph=patch_size, pw=patch_size)
        return out[:, :, :h, :w].to(input_dtype)
    
    def _forward_tiled(self, x, timestep, context, y, guidance, input_dtype):
        """STREAM tiling - one tile in memory, immediately written to the result"""
        bs, c, h, w = x.shape
        
        print(f"[FluxOptimized] Tiling: {x.shape[2]}x{x.shape[3]}")
        
        # Break it into tiles
        tiles, positions, original_shape = self.tiler.split(x)
        print(f"[FluxOptimized] Tiles: {len(tiles)}")
        
        # Result and weights immediately on the CPU (do not accumulate in RAM!)
        result = torch.zeros((bs, c, h, w), device='cpu', dtype=input_dtype)
        weights = torch.zeros((bs, 1, h, w), device='cpu', dtype=input_dtype)
        
        # Context once per CPU
        context_cpu = context.cpu()
        txt_len = context.shape[1]
        
        for idx, (tile, (y_pos, x_pos)) in enumerate(zip(tiles, positions)):
            print(f"[FluxOptimized] Tile {idx+1}/{len(tiles)}", end='\r')
            
            # Preparing the tile
            bs_t, c_t, th, tw = tile.shape
            patch_size = 2
            
            # Padding
            pad_h = (-th) % patch_size
            pad_w = (-tw) % patch_size
            if pad_h or pad_w:
                tile = F.pad(tile, (0, pad_w, 0, pad_h))
            
            # In patches
            img = rearrange(tile, "b c (h ph) (w pw) -> b (h w) (c ph pw)", 
                           ph=patch_size, pw=patch_size)
            
            # PE with global coordinates
            h_len = (th + pad_h) // patch_size
            w_len = (tw + pad_w) // patch_size
            
            # GLOBAL coordinates
            global_y = y_pos // patch_size
            global_x = x_pos // patch_size
            
            img_ids = torch.zeros((h_len, w_len, 3), device='cuda', dtype=input_dtype)
            img_ids[..., 1] = (global_y + torch.arange(h_len, device='cuda', dtype=input_dtype))[:, None]
            img_ids[..., 2] = (global_x + torch.arange(w_len, device='cuda', dtype=input_dtype))[None, :]
            img_ids = img_ids.expand(bs_t, -1, -1, -1).reshape(bs_t, h_len * w_len, 3)
            
            txt_ids = torch.zeros((bs_t, txt_len, 3), device='cuda', dtype=input_dtype)
            
            # Processing
            with torch.no_grad():
                out_patches = self.inner_forward(img.cuda(), img_ids, 
                                                context_cpu, txt_ids, 
                                                timestep, y, guidance)
            
            # Back to spatial
            out_h = (th + pad_h) // patch_size
            out_w = (tw + pad_w) // patch_size
            out_tile = rearrange(out_patches, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                                h=out_h, w=out_w, ph=patch_size, pw=patch_size)
            out_tile = out_tile[:, :, :th, :tw]  # remove padding
            
            # Stream merge (KEY!)
            # Blend mask
            is_top = (y_pos == 0)
            is_bottom = (y_pos + th >= h)
            is_left = (x_pos == 0)
            is_right = (x_pos + tw >= w)

            mask = self.tiler._create_blend_mask(
                th, tw, out_tile.dtype, out_tile.device,
                is_left_edge=is_left,
                is_right_edge=is_right,
                is_top_edge=is_top,
                is_bottom_edge=is_bottom
            )
            
            # Directly to the CPU and get results!
            result[:, :, y_pos:y_pos+th, x_pos:x_pos+tw] += (out_tile * mask).cpu()
            weights[:, :, y_pos:y_pos+th, x_pos:x_pos+tw] += mask.cpu()
            
            # Aggressive cleaning
            del img, out_patches, out_tile, mask, img_ids, txt_ids, tile
            _vram_monitor.emergency_cleanup()
        
        # Final normalization
        print(f"\n[FluxOptimized] Finalizing...")
        result = result / weights.clamp_min(1e-8)
        
        # Cleaning
        del weights, tiles  # tiles are also deleted!
        _global_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()
        
        return result.cuda().to(input_dtype) if torch.cuda.is_available() else result.to(input_dtype)



# EXPORT

__all__ = [
    'IntegratedFluxTransformer2DModel',
    'attention',
    'rope',
    'apply_rope',
    'timestep_embedding',
    'EmbedND',
    'MLPEmbedder',
    'RMSNorm',
    'QKNorm',
    'SelfAttention',
    'Modulation',
    'DoubleStreamBlock',
    'SingleStreamBlock',
    'LastLayer',
    'TuringConfig',
    'VRAMMonitor',
]