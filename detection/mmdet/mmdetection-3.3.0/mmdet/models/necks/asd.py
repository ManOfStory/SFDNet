# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor
from .fpn import FPN
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
from mmengine.model.weight_init import constant_init, xavier_init
from .utils.S3M import S3M as S3M
from .utils.fft_utils import fft2_shift, ifft2_shift
from .utils.gaussian import GaussianBlur

class SequentialSpectrumMamba(BaseModule):
    """Sequential Spectrum Mamba Module for Frequency-Domain Feature Enhancement.

    This module decouples spatial features into amplitude and phase components
    via 2D FFT, processes them using specialized State Space Models (S3M),
    and reconstructs the enhanced spatial features via 2D IFFT.

    Args:
        num_outs (int): Number of output feature scales. Defaults to 5.
        channels (int): Number of input feature channels. Defaults to 768.
        feat_hw (List[Tuple[int, int]]): Spatial resolutions for each scale.
        spectrum (str): Spectrum mode indicator ('low', 'mid', 'high').
        init_cfg (dict, optional): Initialization config dict. Defaults to None.
    """

    def __init__(
        self,
        num_outs: int = 5,
        channels: int = 768,
        feat_hw: List[Tuple[int, int]] = [(304, 304), (152, 152), (76, 76), (38, 38), (19, 19)],
        spectrum: str = 'low',
        init_cfg: dict = None
    ):
        super().__init__(init_cfg=init_cfg)

        assert len(feat_hw) == num_outs, f"feat_hw length ({len(feat_hw)}) must match num_outs ({num_outs})"
        self.num_outs = num_outs
        self.channels = channels
        self.spectrum = spectrum

        self.pos_embed = nn.ParameterList([
            nn.Parameter(torch.zeros(1, channels, H, W))  # 显式指定 channels 维度，支持多通道广播
            for (H, W) in feat_hw
        ])

        self.Amp_S3M = nn.ModuleList([
            S3M(channels, d_state=1, spectrum=spectrum, hw=feat_hw[i]) 
            for i in range(num_outs)
        ])
        
        self.Pha_S3M = nn.ModuleList([
            S3M(channels, d_state=1, spectrum=spectrum, hw=feat_hw[i]) 
            for i in range(num_outs)
        ])

    def forward(self, x: torch.Tensor, level: int):
        pos = self.pos_embed[level]
        amp, pha = fft2_shift(x)

        amp_log = torch.log1p(amp)
        amp_input = amp_log + pos
        amp_s3m_log = self.Amp_S3M[level](amp_input)
        amp_s3m_log = torch.clamp(amp_s3m_log, min=0.0, max=8.0)
        amp_s3m = torch.expm1(amp_s3m_log)
         
        pha_input = pha + pos
        pha_s3m = self.Pha_S3M[level](pha_input)
        pha_s3m = pha_s3m.clamp(min=-torch.pi, max=torch.pi)

        F_star = ifft2_shift(amp_s3m, pha_s3m)
        return F_star
    
class RMSNorm(nn.Module):
    def __init__(self, dims, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dims))

    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm_x
    
class ASDDoG(nn.Module):
    def __init__(
        self,
        use_layer = [2,3,4,5],
        channels: int = 768,
        feat_hw: List[Tuple] = [(304, 304), (152, 152), (76, 76), (38, 38), (19, 19)],                 # list of (H, W), length = num_outs
        sigma = 1.0,
        k = 1.414,
    ):
        super().__init__()

        assert len(feat_hw) == len(use_layer)

        self.num_outs = len(use_layer)
        self.channels = channels

        self.SequentialSpectrumMamba_Low = SequentialSpectrumMamba(self.num_outs, channels, feat_hw, 'low')
        self.SequentialSpectrumMamba_Mid = SequentialSpectrumMamba(self.num_outs, channels, feat_hw, 'mid')
        self.SequentialSpectrumMamba_High = SequentialSpectrumMamba(self.num_outs, channels, feat_hw, 'high')
        # ======================================================
        # DoG parameters
        # ======================================================
        self.k = k
        self.sigma = sigma

        self.blur_1 = GaussianBlur(channels, k * sigma)
        self.blur_2 = GaussianBlur(channels, sigma)
        self.alpha_low = nn.Parameter(torch.ones(channels))
        self.alpha_mid = nn.Parameter(torch.ones(channels))
        self.alpha_high = nn.Parameter(torch.ones(channels))
        
        self.out_ffn = nn.ModuleList([
            nn.Linear(channels, channels) for _ in range(self.num_outs)
        ])
        self.rmsnorm = RMSNorm(channels, eps=1e-6)
        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.alpha_low, 1.0)
        nn.init.constant_(self.alpha_mid, 1.0)
        nn.init.constant_(self.alpha_high, 1.0)

        for m in self.out_ffn:
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0) 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self._is_init = True

    @property
    def is_init(self):
        return self._is_init

    @is_init.setter
    def is_init(self, value):
        self._is_init = value
        
    def forward(self, x, level):
        """
        x: feature maps, length = num_outs
               each shape: [B, C, H, W]
        """
        # ==================== DoG ====================
        P_low = self.blur_1(x)
        P_mid = self.blur_2(x) - P_low
        P_high = x - self.blur_2(x)
        
        # ======== Sequential Spectrum Mamba ==========
        F_low_star = self.SequentialSpectrumMamba_Low(P_low, level)
        F_mid_star = self.SequentialSpectrumMamba_Mid(P_mid, level)
        F_high_star = self.SequentialSpectrumMamba_High(P_high, level)
        
        # ================= Aggregate =================        
        alpha_low = self.alpha_low.view(1, -1, 1, 1)
        alpha_mid = self.alpha_mid.view(1, -1, 1, 1)
        alpha_high = self.alpha_high.view(1, -1, 1, 1)

        F_fused = alpha_low * F_low_star + \
             alpha_mid * F_mid_star + \
             alpha_high * F_high_star
        
        B, C, H, W = F_fused.shape

        F_fused = F_fused.permute(0, 2, 3, 1).contiguous().reshape(B, -1, C)

        F_fused = self.rmsnorm(F_fused)

        P_star = self.out_ffn[level](F_fused)
        
        P_star = P_star.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        return x + P_star

@MODELS.register_module()
class ASDFPN(BaseModule):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(mode='nearest'),
        init_cfg: MultiConfig = [
            dict(type='Xavier', layer='Conv2d', distribution='uniform'),
            dict(type='Xavier', layer='Linear', distribution='uniform')
        ],
        start_epoch = 8,
        feat_hw = [(200,200),(100,100),(50,50),(25,25),(13,13)],
        use_layer = [2,3,4,5],
        sigma = 1.0,
        k = 1.414
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        self.start_epoch = start_epoch
        self.use_layer = use_layer
        super().init_weights()
        self.ASD = ASDDoG(use_layer = use_layer,
                       channels = out_channels,
                       feat_hw = feat_hw,
                       sigma = sigma,
                       k = k)
    
    def forward(self, inputs: Tuple[Tensor], epoch: int=36) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        
        if epoch < self.start_epoch:
            return outs

        for idx, lv in enumerate(self.use_layer):
            outs[lv] = self.ASD(outs[lv], idx)

        return outs
