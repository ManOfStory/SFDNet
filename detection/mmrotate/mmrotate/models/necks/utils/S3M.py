import math
import copy
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import flop_count, parameter_count
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

try:
    from .utils import selective_scan_state_flop_jit, selective_scan_fn, Stem, DownSampling
except:
    from utils import selective_scan_state_flop_jit, selective_scan_fn, Stem, DownSampling

try:
    from Dwconv.dwconv_layer import DepthwiseFunction
except:
    DepthwiseFunction = None


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpectrumAwareSSM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        spectrum = 'low',
        hw = (200, 200),
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.spectrum = spectrum
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds = self.D_init(self.d_inner, dt_init)

        self.selective_scan = selective_scan_fn

        if self.spectrum == 'low':
            self.register_buffer("scan_order", self.low_spectrum_scan_order(hw[0], hw[1]))
        elif self.spectrum == 'mid':
            self.register_buffer("scan_order", self.mid_spectrum_scan_order(hw[0], hw[1]))
        elif self.spectrum == 'high':
            self.register_buffer("scan_order", self.high_spectrum_scan_order(hw[0], hw[1]))
        else:
            assert False,"spectrum error"

        self.register_buffer("inv_scan_order", torch.argsort(self.scan_order))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, bias=True,**factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)

        if bias:
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner)))
                dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner)))
                dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init=="random" or "constant":
            # S4D real initialization
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        elif init=="simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state)))
        elif init=="zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state)))
        else:
            raise NotImplementedError
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        if init=="random" or "constant":
            # D "skip" parameter
            D = torch.ones(d_inner, device=device)
            D = nn.Parameter(D) 
            D._no_weight_decay = True
        elif init == "simple" or "zero":
            D = nn.Parameter(torch.ones(d_inner))
        else:
            raise NotImplementedError
        return D

    
        """
        ys: list of (B, C, L)
        return: (B, C, L)
        """
        # concat on channel
        y = torch.cat(ys, dim=1)   # (B, 4C, L)

        # 1x1 conv merge
        y = self.merge(y)          # (B, C, L)

        return y

    def low_spectrum_scan_order(self, H, W):
        """
        外→内螺旋扫描
        从 (0,0) 向右开始
        返回: [H*W] 扁平索引 tensor
        """
        idx = torch.arange(H * W).view(H, W)

        visited = torch.zeros((H, W), dtype=torch.bool)
        total = H * W
        order = torch.empty(total, dtype=torch.long)

        # 右，下，左，上
        directions = [(0,1),(1,0),(0,-1),(-1,0)]

        row, col = 0, 0
        direction_idx = 0

        for i in range(total):
            order[i] = idx[row, col]
            visited[row, col] = True

            next_row = row + directions[direction_idx][0]
            next_col = col + directions[direction_idx][1]

            if not (0 <= next_row < H and 
                    0 <= next_col < W and 
                    not visited[next_row, next_col]):
                direction_idx = (direction_idx + 1) % 4

            row += directions[direction_idx][0]
            col += directions[direction_idx][1]

        return order

    def high_spectrum_scan_order(self, H, W):
        """
        High spectrum = Low spectrum 反向
        """
        low_order = self.low_spectrum_scan_order(H, W)
        return low_order.flip(0)

    def mid_spectrum_scan_order(self, H, W):
        """
        Mid spectrum: 交替取 Low 和 High
        low: [1,2,3,4,5], high: [5,4,3,2,1] -> mid: [1,5,2,4,3]
        """
        low = self.low_spectrum_scan_order(H, W)
        high = low.flip(0)
        mid = []
        for l, h in zip(low.tolist(), high.tolist()):
            mid.append(l)
            mid.append(h)
        # 如果长度是奇数，最后元素会重复，去掉重复
        mid = mid[:H*W]
        return torch.tensor(mid)
    
    def ssm(self, x):
        B, C, H, W = x.shape
        L = H * W

        xs = x.view(B, C, L)

        idx = self.scan_order.view(1, 1, -1).expand(B, C, -1)
        xs_scan = torch.gather(xs, 2, idx)

        x_dbl = torch.matmul(self.x_proj_weight.view(1, -1, C), xs_scan)

        dts, Bs, Cs = torch.split(
            x_dbl, 
            [self.dt_rank, self.d_state, self.d_state], 
            dim=1
        )

        dts = torch.matmul(self.dt_projs_weight.view(1, C, -1), dts)

        As = -torch.exp(self.A_logs)
        Ds = self.Ds

        h = self.selective_scan(
            xs_scan, dts, 
            As, Bs, None,
            z=None,
            delta_bias=self.dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )
        h = rearrange(h, "b d 1 hw -> b d (1 hw)")
        # 👉 用预计算 inv_idx
        inv_idx = self.inv_scan_order.view(1, 1, -1).expand(B, C, -1)
        h = torch.gather(h, 2, inv_idx)

        y = h * Cs
        y = y + xs * Ds.view(-1, 1)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) 

        x = rearrange(x, 'b h w d -> b d h w').contiguous()
        x = self.act(self.conv2d(x)) 
 
        y = self.ssm(x) 

        y = rearrange(y, 'b d (h w)-> b h w d', h=H, w=W)

        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y


class S3M(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        dt_init: str = "random",
        num_heads: int = 8,
        mlp_ratio = 4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        spectrum = 'low',
        hw = (200, 200),
        **kwargs,
    ):
        super().__init__()

        self.cpe1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, padding=0, groups=hidden_dim)
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SpectrumAwareSSM(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, dt_init=dt_init, spectrum = spectrum, hw = hw, **kwargs)
        self.drop_path = DropPath(drop_path)

        self.cpe2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, padding=0, groups=hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLP(in_features=hidden_dim, hidden_features=int(hidden_dim*mlp_ratio), act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=False)
        self.spectrum = spectrum
        self.init_weights()

    def init_weights(self):
        # 1. ===== CPE (Conditional Positional Encoding) =====
        # 直接对 Conv2d 对象进行初始化，不使用 hasattr 判断
        for cpe in [self.cpe1, self.cpe2]:
            if isinstance(cpe, nn.Conv2d):
                nn.init.constant_(cpe.weight, 0)
                if cpe.bias is not None:
                    nn.init.constant_(cpe.bias, 0)
        # 2. ===== LayerNorm (保持标准初始化) =====
        for ln in [self.ln_1, self.ln_2]:
            if isinstance(ln, nn.LayerNorm):
                nn.init.constant_(ln.weight, 1.0)
                nn.init.constant_(ln.bias, 0.0)
        # 3. ===== Self-Attention / SpectrumAwareSSM =====
        # 递归查找 self_attention 内部所有的线性投影层
        # 这种写法最稳，能抓到任何命名的输出投影层
        for name, m in self.self_attention.named_modules():
            if isinstance(m, nn.Linear):
                # 通常残差前的最后一层叫 proj 或 out_proj
                if 'out_proj' in name:
                    nn.init.constant_(m.weight, 0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    # 内部的输入投影（如 qkv 分支）保持默认或微小初始化
                    nn.init.normal_(m.weight, std=1e-3)
        # 4. ===== MLP (fc2 为输出层) =====
        for name, m in self.mlp.named_modules():
            if isinstance(m, nn.Linear):
                if 'fc2' in name:
                    nn.init.constant_(m.weight, 0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        self._is_init = True

    @property
    def is_init(self):
        return self._is_init

    @is_init.setter
    def is_init(self, value):
        self._is_init = value

    def forward(self, x: torch.Tensor):
        x = x + self.cpe1(x)
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.self_attention(self.ln_1(x)))
        x = x + self.cpe2(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        x = x.permute(0, 3, 1, 2)
        return x