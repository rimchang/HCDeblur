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
from basicsr.models.archs.local_arch import replace_layers
from torchvision.ops import DeformConv2d
from basicsr.models.archs.core.raft import RAFT
from copy import deepcopy

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

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """

    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)


    return output

class NAFEncoderWarpAlignment(nn.Module):
    """ Encodes the input images using a residual network. Uses the alignment_net to estimate optical flow between
        reference (first) image and other images. Warps the embeddings of other images to reference frame co-ordinates
        using the estimated optical flow
    """
    def __init__(self, alignment_net, init_dim=64, train_alignment=True,
                 warp_type='bilinear', middle_seqs_bkl_num=1):
        super().__init__()
        input_channels = 3
        self.warp_type = warp_type
        self.init_dim = init_dim
        self.alignment_net = alignment_net
        self.train_alignment = train_alignment

        self.init_layer = nn.Conv2d(in_channels=input_channels, out_channels=init_dim, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.layers = nn.Sequential(
                    *[NAFBlock(init_dim) for _ in range(middle_seqs_bkl_num)]
                )

    def coords_grid(self, b, h, w, homogeneous=False, device=None):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

        stacks = [x, y]

        if homogeneous:
            ones = torch.ones_like(x)  # [H, W]
            stacks.append(ones)

        grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

        grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

        if device is not None:
            grid = grid.to(device)

        return grid

    def forward(self, x_deblur, x_seqs, flows):

        # x_seqs : b, t, 3, h, w
        # flows : b, t, 2, h, w
        # x_deblur : b, 3, h, w

        b, t, c, h, w = x_seqs.shape
        flows = flows.reshape(b, t, 2, h, w).reshape(b*t, 2, h, w)
        
        # resize deblurred image
        x_center = x_seqs[:,0,:,:,:]
        x_deblur_resize = F.interpolate(x_deblur, scale_factor=1/6, mode='bicubic', align_corners=True, antialias=True)

        # compute optical flows deblur => center frame of seqs
        if self.train_alignment:
            offsets_predictions = self.alignment_net(x_deblur_resize.view(-1, *x_deblur_resize.shape[-3:]) * 255.0, x_center.view(-1, *x_center.shape[-3:]) * 255.0, iters=12)
            offsets = offsets_predictions[-1]

        else:
            with torch.no_grad():
                self.alignment_net = self.alignment_net.eval()
                offsets_low, offsets = self.alignment_net(x_deblur_resize.view(-1, *x_deblur_resize.shape[-3:]) * 255.0, x_center.view(-1, *x_center.shape[-3:]) * 255.0, iters=12, test_mode=True)


        # Extract features of seqs
        x_seqs_feat = x_seqs.reshape(b*t,c,h,w)
        out = self.init_layer(x_seqs_feat)
        feat = self.layers(out)
        oth_feat = feat.view(b*t, self.init_dim, h, w)

        # Warp flows of seqs using the computed flow (delbur => center)
        grid = self.coords_grid(b*t, h, w, device=flows.device)
        grid = grid + flows
        offsets = offsets.unsqueeze(1).repeat(1, t, 1, 1, 1).reshape(b*t, 2, h, w)
        warped_grid = warp(grid, offsets) # warped coordinates of UW seqs

        # scale grid to [-1,1]
        warped_grid[:, 0, :, :] = 2.0 * warped_grid[:, 0, :, :].clone() / max(w - 1, 1) - 1.0
        warped_grid[:, 1, :, :] = 2.0 * warped_grid[:, 1, :, :].clone() / max(h - 1, 1) - 1.0

        # warp all frames to the deblurred frame
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        output = F.grid_sample(oth_feat, warped_grid, mode='bilinear', padding_mode='zeros', align_corners=True) # (b*t, init_dim ,h, w)
        output = output.view(b, t, self.init_dim, h, w)

        return output


class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, center_frame_idx=0):
        super(TSAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def masked_softmax(self, x, mask, dim=2):  # Assuming x has atleast 2 dimensions
        maxes = torch.max(x, dim, keepdim=True)[0]
        x_exp = torch.exp(x - maxes) * mask
        x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
        probs = x_exp / x_exp_sum
        return probs

    def forward(self, none, aligned_feat, mask):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        aligned_feat = aligned_feat.permute(0, 2, 1, 3, 4) # (b, t, c, h, w)

        b, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        embedding = torch.nn.functional.normalize(embedding, dim=2)
        embedding_ref = torch.nn.functional.normalize(embedding_ref, dim=1)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)

        corr_porb = torch.cat(corr_l, dim=1).unsqueeze(1) # (b, 1, t, h, w)
        mask = mask.mean([3, 4]).unsqueeze(3).unsqueeze(3)  # (B, 1, t, 1, 1)
        corr_prob = self.masked_softmax(corr_porb, mask, dim=2) # (B, 1, t, h, w)

        aligned_feat = aligned_feat * corr_prob.permute(0, 2, 1, 3, 4) # (b, t, c, h, w)
        # fusion
        feat = self.lrelu(aligned_feat.sum(dim=1)) # (b, c, h, w)

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(
            self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(
            self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(
            self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add

        return feat


class HCFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, seqs_chan=64, middle_blk_num=10, middle_seqs_bkl_num=10, train_alignment=True):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel*2, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # encoder for deblurred/blurred images
        deblur_chan_base = width
        deblur_chan = deblur_chan_base
        self.encoders.append(NAFBlock(deblur_chan))
        self.downs.append(
            nn.Sequential(
                nn.PixelUnshuffle(2),
                nn.Conv2d(deblur_chan * 4, 2 * deblur_chan, 1, 1)
            )
        )
        deblur_chan = deblur_chan * 2
        self.encoders.append(NAFBlock(deblur_chan))
        self.downs.append(
            nn.Sequential(
                nn.PixelUnshuffle(3),
                nn.Conv2d(deblur_chan * 9, 2 * deblur_chan, 1, 1)
            )
        )
        deblur_chan = deblur_chan * 2


        self.padder_size = 2 ** len(self.encoders)

        # align and extract burst features
        self.alignment_net = RAFT(small=True)
        self.encoder_seqs = NAFEncoderWarpAlignment(self.alignment_net, init_dim=seqs_chan, train_alignment=train_alignment, middle_seqs_bkl_num=1)
        self.load_RAFT(self.alignment_net, './pretrained_models/raft-small.pth')

        # frame-by-frame fusion
        self.fusion_deblur_seqs = nn.Conv2d(in_channels=deblur_chan + seqs_chan, out_channels=deblur_chan,
                                          kernel_size=3,
                                          stride=1, padding=1, groups=1, bias=True)

        # frame-by-frame processing
        self.middle_seqs_blks = nn.ModuleList()
        self.middle_seqs_blks = \
            nn.Sequential(
                *[NAFBlock(deblur_chan) for _ in range(middle_seqs_bkl_num)]
            )

        # merging burst to one frame
        self.fusion = TSAFusion()

        # merged frame processing
        chan = deblur_chan
        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        # decoder
        self.ups.append(
            nn.Sequential(
                nn.Conv2d(chan, (chan // 2) * 9, 1, bias=False),
                nn.PixelShuffle(3)
            )
        )
        chan = chan // 2
        self.decoders.append(
            NAFBlock(chan)
        )

        self.ups.append(
            nn.Sequential(
                nn.Conv2d(chan, (chan//2) * 4, 1, bias=False),
                nn.PixelShuffle(2)
            )
        )
        chan = chan // 2
        self.decoders.append(
            NAFBlock(chan)
        )


    def load_RAFT(self, net, load_path, strict=True):
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)
        print(' load net keys', load_net.keys)
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        net.load_state_dict(load_net, strict=strict)


    def forward(self, inp, deblur, seqs, seqs_mask, seqs_flow, random_crop_params=None):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        # x_seqs : b, t, c, h, w
        # flows : b, t, 2, h, w
        B_seqs, C_seqs, T_seqs, H_seqs, W_seqs = seqs.shape
        seqs = seqs.permute(0, 2, 1, 3, 4) # B, T, C, H, W
        seqs_flow = seqs_flow.permute(0, 2, 1, 3, 4)

        # align and extract burst features
        x_seqs = self.encoder_seqs(deblur, seqs, seqs_flow)

        # Only for the training process
        # For reducing warping artifacts in boundary, we extract features of larger resolution and crop features after warping.
        if random_crop_params is not None:
            lq_top = random_crop_params[0]
            lq_left = random_crop_params[1]
            lq_patch_size = random_crop_params[2]

            gt_top = lq_top * 6
            gt_left = lq_left * 6
            gt_patch_size = lq_patch_size * 6

            x_seqs = x_seqs[:,:,:,lq_top:lq_top+lq_patch_size, lq_left:lq_left+lq_patch_size]
            seqs_mask = seqs_mask[:,:,:,lq_top:lq_top+lq_patch_size, lq_left:lq_left+lq_patch_size]

            inp = inp[:,:,gt_top:gt_top+gt_patch_size,gt_left:gt_left+gt_patch_size]
            deblur = deblur[:,:,gt_top:gt_top+gt_patch_size,gt_left:gt_left+gt_patch_size]


        # extract deblurred/blurred features
        inp_concat = torch.cat([inp, deblur], dim=1)
        x = self.intro(inp_concat)
        x = self.encoders[0](x)
        x = self.downs[0](x)
        x = self.encoders[1](x)
        x = self.downs[1](x)

        # Concat, deblur, blur, and fusion
        B_seqs, T_seqs, C_seqs, H_seqs, W_seqs = x_seqs.shape
        B_ref, C_ref, H_ref, W_ref = x.shape
        x_repeat = x.view(B_ref, 1, C_ref, H_ref, W_ref).repeat(1, T_seqs, 1, 1, 1)

        # concat burst features and deblurred/blurred features
        x_seqs_ref = torch.cat([x_repeat, x_seqs], dim=2)
        x_seqs_ref = x_seqs_ref.reshape(B_ref*T_seqs, C_ref+C_seqs, H_ref, W_ref)

        # fusion concatenating features
        x_seqs_ref = self.fusion_deblur_seqs(x_seqs_ref)
        # processing burst features frame-by-frame
        x_seqs_ref = self.middle_seqs_blks(x_seqs_ref)

        # Fusion all seqs to the center frame
        x_seqs_ref = x_seqs_ref.reshape(B_ref, T_seqs, C_ref, H_ref, W_ref) # (B, T, C, H, W)
        x_seqs = x_seqs_ref
        x_seqs_center = x_seqs[:,0,:,:,:]
        x_seqs = x_seqs.permute(0, 2, 1, 3, 4) # (B, C, T, H, W)
        x_seqs = self.fusion(x_seqs_center, x_seqs, seqs_mask)

        # processing the fused frames
        x = self.middle_blks(x_seqs)

        # decoder
        for decoder, up in zip(self.decoders, self.ups):
            x = up(x)
            # we found removing skip-connection is better
            #x = x + enc_skip
            x = decoder(x)

        # final cnn
        x = self.ending(x)
        x = x + deblur

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

        low_res_size = [1, 3, train_size[-2]//6, train_size[-1]//6]

        deblur_imgs = torch.rand(train_size)
        short_seqs = torch.rand((1, 3, 8, train_size[-2]//6, train_size[-1]//6))
        short_seqs_mask = torch.rand((1, 1, 8, train_size[-2]//6, train_size[-1]//6))
        short_seqs_mask[:,:,-1,:,:] = 0
        short_seqs_mask[:,:,-2,:,:] = 0
        seqs_flow = torch.rand((1, 2, 8, train_size[-2] // 6, train_size[-1] // 6))

        with torch.no_grad():
            self.forward(imgs, deblur_imgs, short_seqs, short_seqs_mask, seqs_flow)

class HCFNet_Local(Local_Base, HCFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        HCFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

