import torch
import torch.nn as nn

class PixelShuffleHead(nn.Module):
    def __init__(self, dim: int, upscale_factor: int):
        super(PixelShuffleHead, self).__init__()
        self.dim = dim

        self.conv_expand = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim*(upscale_factor**2), kernel_size=(1, 1)),
                                         nn.LeakyReLU(inplace=True))


        # self.conv_expand = nn.Conv2d(in_channels=dim, out_channels=dim*(upscale_factor**2), kernel_size=(1, 1))
        self.upsample = nn.PixelShuffle(upscale_factor=upscale_factor)
        

    def forward(self, x: torch.Tensor):
        x = self.conv_expand(x)
        x = self.upsample(x)
     
        return x
