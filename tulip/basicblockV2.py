import torch
import torch.nn as nn

from patchutills import PatchUnmerging,PatchExpanding

from swin_transformer.swintransformerblockv2 import SwinTransformerBlockV2
from swin_transformer.swintransforemerv2 import PatchMergingV2

class BasicBlockV2(nn.Module):
    def __init__(self, index: int, embed_dim: int = 96,input_resolution: tuple=(128, 128), window_size: int = 7, depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24), mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path: float = 0.1,
                 norm_layer=nn.LayerNorm, patch_merging: bool = True):
        super(BasicBlockV2, self).__init__()
        depth = depths[index]
        dim = embed_dim * 2 ** index
        num_head = num_heads[index]

        dpr = [rate.item() for rate in torch.linspace(0, drop_path, sum(depths))]
        drop_path_rate = dpr[sum(depths[:index]):sum(depths[:index + 1])]

        self.blocks = nn.ModuleList([
            SwinTransformerBlockV2(
                dim=dim,
                # input_resolution = (input_resolution[0] // (2 ** i),
                #                     input_resolution[1] // (2 ** i)), 
                input_resolution = input_resolution,
                num_heads=num_head,
                window_size=window_size,
                shift_size= 0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer)
            for i in range(depth)])
        
        if patch_merging:
            self.downsample = PatchMergingV2(input_resolution=input_resolution,
                                             dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
    
class BasicBlockUpV2(nn.Module):
    def __init__(self, index: int, embed_dim: int = 96, input_resolution: tuple=(128, 128),  window_size: int = 7, depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24), mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path: float = 0.1,
                 patch_expanding: bool = True, norm_layer=nn.LayerNorm, patch_unmerging: bool = False):
        super(BasicBlockUpV2, self).__init__()
        
        index = len(depths) - index - 2
        depth = depths[index]
        dim = embed_dim * 2 ** index
        num_head = num_heads[index]

        dpr = [rate.item() for rate in torch.linspace(0, drop_path, sum(depths))]
        drop_path_rate = dpr[sum(depths[:index]):sum(depths[:index + 1])]

        self.blocks = nn.ModuleList([
            SwinTransformerBlockV2(
                dim=dim,
                # input_resolution = (input_resolution[0] * (2 ** i),
                #                     input_resolution[1] * (2 ** i)), 
                input_resolution = input_resolution,
                num_heads=num_head,
                window_size=window_size,
                shift_size= 0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer)
            for i in range(depth)])
        if patch_expanding:
            if patch_unmerging:
                self.upsample = PatchUnmerging(dim = embed_dim * 2 ** index)
            else:
                self.upsample = PatchExpanding(dim=embed_dim * 2 ** index, norm_layer=norm_layer)
        else:
            self.upsample = nn.Identity()

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)

        x = self.upsample(x)
        return x
