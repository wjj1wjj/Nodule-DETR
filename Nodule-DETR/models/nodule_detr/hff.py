"""Hierarchical Feature Fusion (HFF) and its SCDown operation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.misc import NestedTensor

def autopad(k, p=None, d=1):  
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
 
    default_act = nn.SiLU()  
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)
 
    def forward(self, x):
        return self.cv2(self.cv1(x))


class HFF(nn.Module):
    """Hierarchical Feature Fusion (paper section 3.3).

    Project stages 2–4 into P1–P3, derive P4, and produce F1–F4
    using spatial-channel decoupled bottom-up fusion. The final two
    transitions share a downsampler, as in the supplied implementation.
    """

    def __init__(self, backbone_channels, hidden_dim=256, num_feature_levels=4):
        super().__init__()
        if num_feature_levels != 4 or len(backbone_channels) != 3:
            raise ValueError("HFF expects three backbone stages and four feature levels.")
        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.scDown1 = SCDown(hidden_dim, hidden_dim, 3, 2)
        self.scDown2 = SCDown(hidden_dim, hidden_dim, 3, 2)
        self.scDown3 = SCDown(hidden_dim, hidden_dim, 3, 2)

    def forward(self, features, samples, position_encoding, pos):
        """Return fused tensors, their padding masks, and positional encodings."""
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = position_encoding(NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # F1=P1; F2=P2+D1(F1); F3=P3+D2(F2); F4=P4+D2(F3).
        # D2 is shared by the last two transitions in the source implementation.
        srcs[1] = srcs[1] + F.interpolate(self.scDown1(srcs[0]), size=(srcs[1].shape[2], srcs[1].shape[3]))  
        srcs[2] = srcs[2] + F.interpolate(self.scDown2(srcs[1]), size=(srcs[2].shape[2], srcs[2].shape[3]))  
        srcs[3] = srcs[3] + F.interpolate(self.scDown2(srcs[2]), size=(srcs[3].shape[2], srcs[3].shape[3]))  

        return srcs, masks, pos
