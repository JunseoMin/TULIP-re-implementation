import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(4, 4), in_c: int = 3, embed_dim: int = 96, norm_layer: nn.Module = None, circular_padding: bool = False):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        self.circular_padding = circular_padding
        if circular_padding:
            self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(self.patch_size[0], 8), stride=patch_size)
        else:
            self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def padding(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        if H % self.patch_size[0] != 0 or W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[0] - W % self.patch_size[1],
                             0, self.patch_size[1] - H % self.patch_size[0],
                             0, 0))
        return x
    
    # Circular padding is only used on the width of range image 
    def circularpadding(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (2, 2, 0, 0), "circular")
        return x

    def forward(self, x):
        x = self.padding(x)

        if self.circular_padding:
            # Circular Padding
            x = self.circularpadding(x)

        x = self.proj(x)
        x = rearrange(x, 'B C H W -> B H W C')
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    @staticmethod
    def padding(x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape

        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        return x
    

    @staticmethod
    def merging(x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        return x

    def forward(self, x):
        x = self.padding(x)
        x = self.merging(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x

# Patch Unmerging layer
class PatchUnmerging(nn.Module):
    def __init__(self, dim: int):
        super(PatchUnmerging, self).__init__()
        self.dim = dim
        #ToDo: Use linear with norm layer?
        self.expand = nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=(1, 1))
        self.upsample = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x: torch.Tensor):
        x = rearrange(x, 'B H W C -> B C H W')
        x = self.expand(x.contiguous())
        x = self.upsample(x)
        # x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=1, P2=4)
        x = rearrange(x, 'B C H W -> B H W C')
        return x

# Original Patch Expanding layer used in Swin MAE
class PatchExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)
        # self.patch_size = patch_size

    def forward(self, x: torch.Tensor):
        
        x = self.expand(x)
        # x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=1, P2=4)
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=2, P2=2)
        x = self.norm(x)
        return x


# Original Final Patch Expanding layer used in Swin MAE
class FinalPatchExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm, upscale_factor = 4):
        super(FinalPatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, (upscale_factor**2) * dim, bias=False)
        self.norm = norm_layer(dim)
        self.upscale_factor = upscale_factor

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=self.upscale_factor,
                                                                P2=self.upscale_factor,
                                                                C = self.dim)
        x = self.norm(x)
        return x
   