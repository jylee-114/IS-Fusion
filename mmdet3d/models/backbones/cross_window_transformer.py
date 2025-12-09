import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.runner import BaseModule, ModuleList
from mmcv.utils import to_2tuple

from ..builder import BACKBONES
from ..utils.transformer import PatchEmbed, PatchMerging


class CrossWindowOp(BaseModule):
    """Shift-window structure with Conv-based Cross-window operation.

    * Input:  (B, L, C), hw_shape=(H, W)
    * Output: (B, L, C)
    * Same interface as Swin's ShiftWindowMSA, but internally composed of
      depthwise+pointwise convolutions instead of self-attention.
    """

    def __init__(self,
                 embed_dims,
                 window_size,
                 shift_size=0,
                 expansion=2,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        hidden_dims = embed_dims * expansion

        # Lightweight conv block operating only within windows
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dims, hidden_dims, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(
                hidden_dims,
                hidden_dims,
                kernel_size=3,
                padding=1,
                groups=hidden_dims,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dims, embed_dims, kernel_size=1, bias=True),
        )

        self.drop = build_dropout(dropout_layer)

    @staticmethod
    def window_partition(x, window_size):
        """x: (B, H, W, C) -> windows: (num_windows*B, ws, ws, C)"""
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size,
                   W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows

    @staticmethod
    def window_reverse(windows, window_size, H, W):
        """windows: (num_windows*B, ws, ws, C) -> x: (B, H, W, C)"""
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size,
                         window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, query, hw_shape):
        """
        Args:
            query: (B, L, C)
            hw_shape: (H, W)
        """
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        x = query.view(B, H, W, C)

        # Pad to be a multiple of window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = x.shape[1], x.shape[2]

        # Cyclic shift (same as Swin)
        if self.shift_size > 0:
            x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

        # Window partition
        windows = self.window_partition(x, self.window_size)   # (nW*B, ws, ws, C)
        windows = windows.permute(0, 3, 1, 2).contiguous()     # (nW*B, C, ws, ws)

        # Conv-based cross-window operation (local conv within windows)
        windows = self.conv(windows)

        windows = windows.permute(0, 2, 3, 1).contiguous()     # (nW*B, ws, ws, C)
        x = self.window_reverse(windows, self.window_size, H_pad, W_pad)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = self.drop(x)
        return x
    
class CrossWindowBlock(BaseModule):
    """SwinBlock with self-attention replaced by CrossWindowOp.

    Args:
        embed_dims (int): Number of channels.
        feedforward_channels (int): FFN hidden dimension.
        window_size (int): Window size.
        shift (bool): If True, shift by window_size//2.
        drop_rate (float): FFN dropout rate.
        drop_path_rate (float): Stochastic depth rate.
        act_cfg, norm_cfg, with_cp: Same as in Swin.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.cwin = CrossWindowOp(
            embed_dims=embed_dims,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            expansion=2,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x, hw_shape):
        """Forward pass with optional gradient checkpointing."""
        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.cwin(x, hw_shape)
            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)
            return x

        if self.with_cp and x.requires_grad:
            x = torch.utils.checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x
    
class CrossWindowBlockSequence(BaseModule):
    """Equivalent to Swin's SwinBlockSequence.

    Args:
        embed_dims: Number of channels.
        feedforward_channels: FFN hidden dimension.
        depth: Number of blocks in this stage.
        window_size: Window size.
        drop_rate: Dropout rate.
        drop_path_rate: Stochastic depth rate.
        downsample: Optional downsampling layer.
        act_cfg, norm_cfg, with_cp: Same as in Swin.
        init_cfg: Initialization config.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [drop_path_rate for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = CrossWindowBlock(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample is not None:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape
        
@BACKBONES.register_module()
class CrossWindowTransformer(BaseModule):
    """Shifted cross-window convolution-based backbone.

    Similar interface to SwinTransformer, but replaces WindowMSA
    with CrossWindowBlock for local convolution-based processing.

    Example config:
        img_backbone=dict(
            type='CrossWindowTransformer',
            pretrain_img_size=224,
            in_channels=3,
            embed_dims=96,
            patch_size=4,
            window_size=7,
            mlp_ratio=4,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),  # Unused but kept for config compatibility
            strides=(4, 2, 2, 2),
            out_indices=(1, 2, 3),
            patch_norm=True,
            drop_rate=0.,
            drop_path_rate=0.2,
            with_cp=False,
        )
    """

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),  # Dummy, for config compatibility
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 patch_norm=True,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)

        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple) and len(pretrain_img_size) == 1:
            pretrain_img_size = to_2tuple(pretrain_img_size[0])

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))
        else:
            self.absolute_pos_embed = None

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # Stochastic depth (drop path) schedule
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.stages = ModuleList()
        in_ch = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_ch,
                    out_channels=2 * in_ch,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = CrossWindowBlockSequence(
                embed_dims=in_ch,
                feedforward_channels=mlp_ratio * in_ch,
                depth=depths[i],
                window_size=window_size,
                drop_rate=drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.stages.append(stage)

            if downsample is not None:
                in_ch = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Norm layers for each stage output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            self.add_module(f'norm{i}', layer)

    def init_weights(self):
        # Initialization similar to Swin
        if self.use_abs_pos_embed and self.absolute_pos_embed is not None:
            trunc_normal_init(self.absolute_pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m.weight, std=.02)
                if m.bias is not None:
                    constant_init(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m.bias, 0)
                constant_init(m.weight, 1.0)

    def forward(self, x):
        # x: (B, 3, H, W)
        x, hw_shape = self.patch_embed(x)  # x: (B, L, C)

        if self.use_abs_pos_embed and self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                B, L, C = out.shape
                H, W = out_hw_shape
                out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs