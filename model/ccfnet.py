import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable, Tuple
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def make_cbr(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.PReLU()
    )


def make_cbg(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.GELU()
    )


def rescale_to(x, scale_factor: float = 2, interpolation='nearest'):
    return F.interpolate(x, scale_factor=scale_factor, mode=interpolation)


def resize_as(x, y, interpolation='bilinear'):
    return F.interpolate(x, size=y.shape[-2:], mode=interpolation)


def image2patches(x):
    """b c (hg h) (wg w) -> (hg wg b) c h w"""
    x = rearrange(x, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)
    return x


def patches2image(x):
    """(hg wg b) c h w -> b c (hg h) (wg w)"""
    x = rearrange(x, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
    return x


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.register_buffer(
            "dim_t",
            torch.arange(0, self.num_pos_feats, dtype=torch.float32)
        )

    def forward(self, b, h, w):
        device = self.dim_t.device
        mask = torch.zeros([b, h, w], dtype=torch.bool, device=device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = self.temperature ** (2 * (self.dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)

        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class ResBlock2D(nn.Module):
    def __init__(self, dim, res_se_ratio):
        super().__init__()
        hidden_dim = int(res_se_ratio * dim)
        self.conv0 = nn.Conv2d(dim, hidden_dim, 3, 1, 1)
        self.conv1 = nn.Conv2d(hidden_dim, dim, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv0(x))
        rs1 = self.conv1(rs1)
        rs = torch.add(x, rs1)
        return rs


class PixelShuffle(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()
        self.upsamle = nn.Sequential(
            nn.Conv2d(dim, dim * (scale ** 2), 3, 1, 1, bias=False),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        return self.upsamle(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
            )
        else:
            self.up0 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, 2, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
                nn.LeakyReLU()
            )
            self.up1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 2, 2, 0),
                nn.LeakyReLU()
            )
        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        x1 = self.up1(x1)
        x = x1 + x2
        return self.relu(self.conv(x))


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.LeakyReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 2, 2, 0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp_2(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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


class CrossWindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_heads, kv_bias=True, q_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=kv_bias)
        self.q = nn.Linear(dim, dim, bias=q_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        B_, N, C = x.shape
        kv = self.kv(y).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


def window_partition(x, window_size: int):
    """Partition x into non-overlapping windows.

    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """Reverse windows back to feature map.

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class ConvPosEnc(nn.Module):
    """Depth-wise convolution to get the positional information."""

    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(
            dim,
            dim,
            to_2tuple(k),
            to_2tuple(1),
            to_2tuple(k // 2),
            groups=dim
        )

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        feat = feat.flatten(2).transpose(1, 2)
        x = x + feat
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        n_heads=16,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_pan = nn.Linear(dim, dim)
        self.proj_ms = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.cross_heads = n_heads
        self.cross_attn_0_to_1 = nn.MultiheadAttention(
            dim, self.cross_heads, dropout=0.0, batch_first=False
        )
        self.cross_attn_1_to_0 = nn.MultiheadAttention(
            dim, self.cross_heads, dropout=0.0, batch_first=False
        )

        self.relation_judger = nn.Sequential(
            Mlp_2(self.dim * 2, self.dim, dim),
            torch.nn.Softmax(dim=-1),
        )

        self.k_noise = nn.Embedding(2, dim)
        self.v_noise = nn.Embedding(2, dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, ms, pan):
        B, N, C = ms.shape

        # cross-attn per batch
        new_x0 = []
        new_x1 = []
        # direction: pan -> ms
        # q: [1, B*N, C]
        q0 = pan.reshape(1, B * N, C)

        # relation gate: [B, N, 1] -> [1, B*N, 1]
        rel0 = self.relation_judger(torch.cat([pan, ms], dim=-1)).reshape(1, B * N, C)

        noise_k0 = self.k_noise.weight[0].view(1, 1, C).expand(1, B * N, C) + q0
        noise_v0 = self.v_noise.weight[0].view(1, 1, C).expand(1, B * N, C) + q0

        k_other0 = q0 * rel0  # [1, B*N, C]
        v_other0 = ms.reshape(1, B * N, C)  # [1, B*N, C]

        k0 = torch.cat([noise_k0, k_other0], dim=0)  # [2, B*N, C]
        v0 = torch.cat([noise_v0, v_other0], dim=0)  # [2, B*N, C]

        out0, _ = self.cross_attn_0_to_1(q0, k0, v0)  # [1, B*N, C]
        new_x0 = pan + out0.reshape(B, N, C)

        # direction: ms -> pan
        q1 = ms.reshape(1, B * N, C)
        rel1 = self.relation_judger(torch.cat([ms, pan], dim=-1)).reshape(1, B * N, C)

        noise_k1 = self.k_noise.weight[1].view(1, 1, C).expand(1, B * N, C) + q1
        noise_v1 = self.v_noise.weight[1].view(1, 1, C).expand(1, B * N, C) + q1

        k_other1 = q1 * rel1
        v_other1 = pan.reshape(1, B * N, C)

        k1 = torch.cat([noise_k1, k_other1], dim=0)  # [2, B*N, C]
        v1 = torch.cat([noise_v1, v_other1], dim=0)  # [2, B*N, C]

        out1, _ = self.cross_attn_1_to_0(q1, k1, v1)  # [1, B*N, C]
        new_x1 = ms + out1.reshape(B, N, C)

        pan = self.proj_drop(self.proj_pan(new_x0))
        ms = self.proj_drop(self.proj_ms(new_x1))

        return ms, pan


class SARE(nn.Module):
    def __init__(self, dim, num_tokens=64, token_dim=128, reduction=4):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.token_dim = token_dim

        self.semantic_tokens = nn.Parameter(torch.randn(num_tokens, token_dim))
        nn.init.xavier_uniform_(self.semantic_tokens)

        self.query_proj = nn.Conv2d(dim, token_dim, 1)

        self.output_proj = nn.Conv2d(token_dim, dim, 1)

        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1),
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )
        self.res_proj = nn.Conv2d(dim, dim, 1)
        self.bn1 = nn.BatchNorm2d(dim)

        self.DWConv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.Dropout(0.05),
        )
        self.DWConv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim),
            nn.Dropout(0.05),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, stride=1),
        )

    def forward(self, x):
        """x: [B, C, H, W]"""
        B, C, H, W = x.shape
        residual = self.bn1(self.res_proj(x))

        x1 = self.DWConv1(x)
        x2 = self.DWConv2(x)

        q = self.query_proj(x).flatten(2)  # [B, token_dim, HW]
        q = q.permute(0, 2, 1)  # [B, HW, token_dim]

        # sim: [B, HW, num_tokens]
        sim = torch.matmul(q, self.semantic_tokens.T) / (self.token_dim ** 0.5)
        sim = F.softmax(sim, dim=-1)

        sem_out = torch.matmul(sim, self.semantic_tokens)
        sem_out = sem_out.permute(0, 2, 1).reshape(B, self.token_dim, H, W)

        gate = self.gate(x)
        enhanced = self.output_proj(sem_out) * gate + x

        enhanced = torch.cat((enhanced, x1, x2), dim=1)
        out = self.fusion(enhanced) + residual
        return out


class Block(nn.Module):
    def __init__(
        self,
        dim=32,
        num_heads=4,
        mlp_ratio=1.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.05,
        attn_drop=0.05,
        act_layer=nn.GELU,
        n_heads=8,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            n_heads=n_heads,
        )
        # NOTE: drop path for stochastic depth
        self.drop_out = (
            nn.Dropout(drop)
            if drop > 0.0
            else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.apply(self._init_weights)

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=qkv_bias),
        )

        self.kv = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1, bias=qkv_bias),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=qkv_bias),
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(0.)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.panenhance = SARE(dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, ms, pan):
        B, C, H, W = ms.shape

        q = self.q(ms)
        kv = self.kv(ms)
        k, v = kv.chunk(2, dim=1)
        q = rearrange(q, 'B (head C) H W -> B head C (H W)', head=self.num_heads)
        k = rearrange(k, 'B (head C) H W -> B head C (H W)', head=self.num_heads)
        v = rearrange(v, 'B (head C) H W -> B head C (H W)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        _, _, C, _ = q.shape
        mask4 = torch.zeros(B, self.num_heads, C, C, device=ms.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        index = torch.topk(attn, k=int(C * 8 / 10), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))
        attn4 = attn4.softmax(dim=-1)
        out4 = (attn4 @ v)
        out = out4 * self.attn4
        out = rearrange(out, 'B head C (H W) -> B (head C) H W', head=self.num_heads, H=H, W=W)
        ms = self.project_out(out) + ms

        pan = self.panenhance(pan)

        ms = rearrange(ms, 'B C H W -> B (H W) C', H=H, W=W)
        pan = rearrange(pan, 'B C H W -> B (H W) C', H=H, W=W)

        ms_e, pan_e = self.attn(
            self.norm1(ms),
            self.norm1(pan),
        )

        ms_e = self.drop_out(ms_e) + ms
        ms_f = self.drop_out(self.mlp(self.norm2(ms_e))) + ms_e

        pan_e = self.drop_out(pan_e) + pan
        pan_f = self.drop_out(self.mlp(self.norm2(pan_e))) + pan_e

        ms_f = rearrange(ms_f, 'B (H W) C -> B C H W', H=H, W=W)
        pan_f = rearrange(pan_f, 'B (H W) C -> B C H W', H=H, W=W)

        return ms_f, pan_f


class resblock(nn.Module):
    def __init__(self, channel):
        super(resblock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.act = nn.PReLU(num_parameters=channel, init=0.01)

    def forward(self, x):
        rs1 = self.act(self.conv1(x))
        rs2 = self.conv2(rs1) + x
        return rs2


class combine(nn.Module):
    def __init__(self, channel):
        super(combine, self).__init__()
        self.resblock = resblock(channel=channel)
        self.a = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x1, x2):
        rs1 = self.a * x1 + (1 - self.a) * x2
        rs2 = self.resblock(rs1)
        return rs2


class Net(nn.Module):
    def __init__(self, num_channels=8, dim=32, pan_dim=1, scale=4):
        super().__init__()
        self.act = nn.PReLU(num_parameters=num_channels, init=0.01)
        self.upsample = PixelShuffle(num_channels, scale)
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(pan_dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(num_channels, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.to_hrms = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, num_channels, 3, 1, 1)
        )

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        dim0 = dim
        dim1 = int(dim0 * 2)
        dim2 = int(dim1 * 2)
        dim3 = dim1
        dim4 = dim0

        self.combine = combine(dim)
        self.resblock = resblock(dim)

        # layer 0
        self.block0 = Block(dim0)
        self.down0 = Down(dim0, dim1)

        # layer 1
        self.block1 = Block(dim1)
        self.down1 = Down(dim1, dim2)

        # layer 2
        self.block2 = Block(dim2)
        self.up0 = Up(dim2, dim3)

        # layer 3
        self.block3 = Block(dim3)
        self.up1 = Up(dim3, dim4)

    def forward(self, ms, lms, pan):
        B, _, _, _ = pan.shape
        pan = self.raise_pan_dim(pan)
        lms_1 = self.act(self.upsample(ms) + lms)
        lms_2 = self.raise_ms_dim(lms_1)

        # layer 0
        x, y = self.block0(lms_2, pan)  # 32 64 64
        skip_c10 = x  # 32 64 64
        x = self.down0(x)  # 64 32 32
        skip_c11 = y  # 32 64 64
        y = self.down0(y)  # 64 32 32

        # layer 1
        x, y = self.block1(x, y)  # 64 32 32
        skip_c20 = x
        x = self.down1(x)  # 128 16 16
        skip_c21 = y  # 64 32 32
        y = self.down1(y)  # 128 16 16

        # layer 2
        x, y = self.block2(x, y)  # 128 16 16
        x = self.up0(x, skip_c20)  # 64 32 32
        y = self.up0(y, skip_c21)  # 64 32 32

        # layer 3
        x, y = self.block3(x, y)  # 64 32 32
        x = self.up1(x, skip_c10)  # 32 64 64
        y = self.up1(y, skip_c11)  # 32 64 64

        output = self.resblock(self.combine(x, y))

        output = self.to_hrms(output) + lms_1

        return output


def summaries(model, input_size, grad=False):
    if grad:
        from torchinfo import summary
        summary(model, input_size=input_size)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)


def prepare_input(resolution):
    device = torch.device('cuda:0')
    return {
        'ms': torch.randn(1, 8, 64, 64, device=device),
        'lms': torch.randn(1, 8, 256, 256, device=device),
        'pan': torch.randn(1, 1, 256, 256, device=device),
    }


def measure_model(model: torch.nn.Module, device: torch.device = None):
    if device is None:
        device = next(model.parameters()).device

    print(f"{'=' * 60}")
    print(f"Model Analysis on {device}")
    print(f"{'=' * 60}")

    flops_str, params_str = get_model_complexity_info(
        model,
        input_res=(1,),
        input_constructor=prepare_input,
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True
    )
    print(f"{'Number of parameters':<30} {params_str}")

    dummy_inputs = prepare_input(None)
    flops = FlopCountAnalysis(model, (dummy_inputs['ms'], dummy_inputs['lms'], dummy_inputs['pan']))
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    print(f"FLOPs: {flops.total() / 1e9:.2f}G")


if __name__ == '__main__':
    import time
    import torch
    from ptflops import get_model_complexity_info
    from fvcore.nn import FlopCountAnalysis

    model = Net().to('cuda:0')
    model.eval()

    measure_model(model)