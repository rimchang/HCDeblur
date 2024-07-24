# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from torchvision.ops import DeformConv2d
from basicsr.models.archs.local_arch import replace_layers

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class KernelDeformableBlock(nn.Module):
    def __init__(self, channel_blur, channel_kernel, deform_groups):
        super().__init__()

        kernel_size = 3
        padding = kernel_size // 2
        out_channels = deform_groups * 3 * (kernel_size ** 2)

        self.naf_block_blur = NAFBlock(channel_blur)
        self.naf_block_kernel = NAFBlock(channel_kernel)
        self.attn_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channel_kernel + channel_blur, out_channels=channel_kernel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.offset_conv = nn.Conv2d(channel_kernel, out_channels, kernel_size, stride=1, padding=padding, bias=True)
        self.deform = DeformConv2d(channel_blur, channel_blur, kernel_size, padding=2, groups=deform_groups, dilation=2)
        self.fusion = nn.Conv2d(in_channels=channel_blur * 2, out_channels=channel_blur, kernel_size=3, stride=1, padding=1)

    def offset_gen(self, x):

        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return offset, mask

    def forward(self, feat_blur, feat_kernel):
        '''
        Input
            feat_blur: (B, 256, H/8, W/8)
            feat_kernel: (B, 256, H/8, W/8)

        Output
            feat_blur: (B, 256, H/8, W/8)
        '''

        feat_blur = self.naf_block_blur(feat_blur)
        feat_kernel = self.naf_block_kernel(feat_kernel)

        attn = self.attn_weight(torch.cat([feat_kernel, feat_blur], dim=1))
        feat_kernel = feat_kernel * attn

        offset, mask = self.offset_gen(self.offset_conv(feat_kernel))
        feat = self.deform(feat_blur, offset, mask)

        out = torch.cat((feat_blur, feat), dim=1)
        out = self.fusion(out)

        return out, feat_kernel


class HCDNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.kernel_intro = nn.Conv2d(in_channels=18, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)


        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.kernel_downs = nn.ModuleList()
        self.KDM_blocks = nn.ModuleList()
        chan = width
        deform_groups = [2, 8, 16, 32]
        for num, deform_group in zip(enc_blk_nums, deform_groups):
            self.KDM_blocks.append(KernelDeformableBlock(chan, chan, deform_group))
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            self.kernel_downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.KDM_blocks.append(KernelDeformableBlock(chan, chan, 64))
        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, kernel):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        kernel = self.check_image_size(kernel)

        kernel = kernel.float()

        # features of a blurred image and per-pixel kernels
        x = self.intro(inp)
        x_kernel = self.kernel_intro(kernel)

        encs = []
        for KDM, encoder, down, kernel_down in zip(self.KDM_blocks, self.encoders, self.downs, self.kernel_downs):
            x, x_kernel = KDM(x, x_kernel) # applying KDM
            x = encoder(x) # one NAFBlock
            encs.append(x)

            # downsampling features of a blurred image and kernels
            x = down(x)
            x_kernel = kernel_down(x_kernel)

        # last encoder
        x, x_kernel = self.KDM_blocks[-1](x, x_kernel)
        x = self.middle_blks(x)

        # decoder
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        return x

class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        kernels = torch.rand((1, 18, train_size[-2], train_size[-1]))

        with torch.no_grad():
            self.forward(imgs, kernels)

class HCDNet_Local(Local_Base, HCDNet):
    def __init__(self, *args, train_size=(1, 3, 384, 384), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        HCDNet.__init__(self, *args, **kwargs)

        print(train_size)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

