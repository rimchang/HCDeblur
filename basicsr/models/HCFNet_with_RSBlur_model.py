# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import torch
import torch.nn.functional as F
import os
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import scipy.io

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.data.RSBlur_utils import *

from debayer import Debayer5x5, Layout

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

class HCFNet_with_RSBlur_model(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(HCFNet_with_RSBlur_model, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])

        ###################### RSBlur parameters ##################################
        # load paramerters for isp
        cam2xyz_W_np = scipy.io.loadmat('./mat_collections/W_CCM_0925_from_0913.mat')['colorCorrectionMatrix'].astype(
            'float32').transpose()
        cam2xyz_UW_np = scipy.io.loadmat('./mat_collections/UW_CCM_0925_from_0913.mat')['colorCorrectionMatrix'].astype(
            'float32').transpose()

        self.cam2xyz_W = torch.tensor(cam2xyz_W_np).cuda().float()
        self.cam2xyz_UW = torch.tensor(cam2xyz_UW_np).cuda().float()

        self.xyz2cam_W = torch.inverse(self.cam2xyz_W).cuda().float()
        self.xyz2cam_UW = torch.inverse(self.cam2xyz_UW).cuda().float()

        # lin2xyz
        self.M_lin2xyz = torch.tensor(scipy.io.loadmat('./mat_collections/M_lin2xyz.mat')['M'],
                                 dtype=torch.float32).cuda().float()

        # xyz2lin
        self.M_xyz2lin = torch.tensor(scipy.io.loadmat('./mat_collections/M_xyz2lin.mat')['M'],
                                 dtype=torch.float32).cuda().float()

        self.demosaic = [Debayer5x5(layout=Layout.RGGB).cuda(),\
                         Debayer5x5(layout=Layout.BGGR).cuda(),\
                         Debayer5x5(layout=Layout.GRBG).cuda(),\
                         Debayer5x5(layout=Layout.GBRG).cuda()]

        # self.demosaic = Demosaic()
        # self.demosaic.cuda()


    def RSBlurPipeline_for_W(self, blurred_pt, sat_mask_pt, red_gain, blue_gain, beta1, beta2, alpha_saturation,
                             vinetting_g_W):

        blurred_pt = blurred_pt.permute(0, 2, 3, 1)
        sat_mask_pt = sat_mask_pt.permute(0, 2, 3, 1)

        batch_size, _, _, _ = blurred_pt.shape

        # inverse tone mapping
        blurred_L = rgb2lin_pt(blurred_pt)

        # saturation synthesis
        blurred_L = blurred_L + (alpha_saturation.view(batch_size, 1, 1, 1) * sat_mask_pt)
        blurred_L = torch.clamp(blurred_L, 0, 1)

        blurred_sat = blurred_L.clone()

        # from linear RGB to XYZ
        img_XYZ = lin2xyz(blurred_L, self.M_lin2xyz)  # XYZ

        # from XYZ to Cam
        img_Cam = apply_cmatrix(img_XYZ, self.xyz2cam_W)

        # Mosaic
        img_Cam_split = torch.split(img_Cam, batch_size//4, dim=0)
        fr_now_split = torch.split(red_gain, batch_size//4, dim=0)
        fb_now_split = torch.split(blue_gain, batch_size//4, dim=0)
        vinetting_g_W_split = torch.split(vinetting_g_W, batch_size // 4, dim=0)


        img_mosaic_rggb = mosaic_bayer(img_Cam_split[0], 'RGGB')
        img_mosaic_rggb = WB_img(img_mosaic_rggb, 'RGGB', 1/fr_now_split[0], 1/fb_now_split[0])
        vignetting_rggb = mosaic_bayer(vinetting_g_W_split[0].unsqueeze(3).repeat(1, 1, 1, 3), 'RGGB')
        img_mosaic_rggb = img_mosaic_rggb * (1/vignetting_rggb)

        img_mosaic_bggr = mosaic_bayer(img_Cam_split[1], 'BGGR')
        img_mosaic_bggr = WB_img(img_mosaic_bggr, 'BGGR', 1/fr_now_split[1], 1/fb_now_split[1])
        vignetting_bggr = mosaic_bayer(vinetting_g_W_split[1].unsqueeze(3).repeat(1, 1, 1, 3), 'BGGR')
        img_mosaic_bggr = img_mosaic_bggr * (1 / vignetting_bggr)

        img_mosaic_grbg = mosaic_bayer(img_Cam_split[2], 'GRBG')
        img_mosaic_grbg = WB_img(img_mosaic_grbg, 'GRBG', 1/fr_now_split[2], 1/fb_now_split[2])
        vignetting_grbg = mosaic_bayer(vinetting_g_W_split[2].unsqueeze(3).repeat(1, 1, 1, 3), 'GRBG')
        img_mosaic_grbg = img_mosaic_grbg * (1 / vignetting_grbg)

        img_mosaic_gbrg = mosaic_bayer(img_Cam_split[3], 'GBRG')
        img_mosaic_gbrg = WB_img(img_mosaic_gbrg, 'GBRG', 1/fr_now_split[3], 1/fb_now_split[3])
        vignetting_gbrg = mosaic_bayer(vinetting_g_W_split[3].unsqueeze(3).repeat(1, 1, 1, 3), 'GBRG')
        img_mosaic_gbrg = img_mosaic_gbrg * (1 / vignetting_gbrg)

        img_mosaic = torch.cat([img_mosaic_rggb, img_mosaic_bggr, img_mosaic_grbg, img_mosaic_gbrg], dim=0) # (b, h/2, w/2, 4)

        # -------- ADDING POISSON-GAUSSIAN NOISE ON RAW -
        img_mosaic_noise = add_Poisson_noise_random(img_mosaic, beta1, beta2)

        # -------- ISP PROCESS --------------------------
        img_mosaic_split = torch.split(img_mosaic_noise, batch_size//4, dim=0)

        img_demosaic_rggb = img_mosaic_split[0] * vignetting_rggb
        img_demosaic_rggb = WB_img(img_demosaic_rggb, 'RGGB', fr_now_split[0], fb_now_split[0]) # (b, h//2, w//2, 4)
        img_demosaic_rggb = torch.nn.functional.pixel_shuffle(img_demosaic_rggb.permute(0, 3, 1, 2), 2) # (b, 1, h, w)
        img_demosaic_rggb = self.demosaic[0](img_demosaic_rggb) # (b, 3, h, w)
        #img_demosaic_rggb = self.demosaic.forward(img_demosaic_rggb, 'RGGB')  # (b, 3, h, w)

        img_demosaic_bggr = img_mosaic_split[1] * vignetting_bggr
        img_demosaic_bggr = WB_img(img_demosaic_bggr, 'BGGR', fr_now_split[1], fb_now_split[1])
        img_demosaic_bggr = torch.nn.functional.pixel_shuffle(img_demosaic_bggr.permute(0, 3, 1, 2), 2)
        img_demosaic_bggr = self.demosaic[1](img_demosaic_bggr)
        #img_demosaic_bggr = self.demosaic.forward(img_demosaic_bggr, 'BGGR')  # (b, 3, h, w)

        img_demosaic_grbg = img_mosaic_split[2] * vignetting_grbg
        img_demosaic_grbg = WB_img(img_demosaic_grbg, 'GRBG', fr_now_split[2], fb_now_split[2])
        img_demosaic_grbg = torch.nn.functional.pixel_shuffle(img_demosaic_grbg.permute(0, 3, 1, 2), 2)
        img_demosaic_grbg = self.demosaic[2](img_demosaic_grbg)
        #img_demosaic_grbg = self.demosaic.forward(img_demosaic_grbg, 'GRBG')  # (b, 3, h, w)

        img_demosaic_gbrg = img_mosaic_split[3] * vignetting_gbrg
        img_demosaic_gbrg = WB_img(img_demosaic_gbrg, 'GBRG', fr_now_split[3], fb_now_split[3])
        img_demosaic_gbrg = torch.nn.functional.pixel_shuffle(img_demosaic_gbrg.permute(0, 3, 1, 2), 2)
        img_demosaic_gbrg = self.demosaic[3](img_demosaic_gbrg)
        #img_demosaic_gbrg = self.demosaic.forward(img_demosaic_gbrg, 'GBRG')  # (b, 3, h, w)

        img_demosaic = torch.cat([img_demosaic_rggb, img_demosaic_bggr, img_demosaic_grbg, img_demosaic_gbrg], dim=0)
        img_demosaic = torch.clamp(img_demosaic, 0, 1) # (b, c, h, w)
        img_demosaic = img_demosaic.permute(0, 2, 3, 1) # (b, h, w, c)

        # from Cam to XYZ
        img_IXYZ = apply_cmatrix(img_demosaic, self.cam2xyz_W)

        # frome XYZ to linear RGB
        img_IL = xyz2lin(img_IXYZ, self.M_xyz2lin)

        # tone mapping
        img_Irgb = lin2rgb_pt(img_IL)
        img_Irgb = torch.clamp(img_Irgb, 0, 1)  # (h, w, c)

        blurred = img_Irgb

        # don't add noise on saturated region
        sat_region = torch.ge(blurred_sat, 1.0)
        non_sat_region = torch.logical_not(sat_region)
        blurred = (blurred_sat * sat_region) + (blurred * non_sat_region)

        blurred = blurred.permute(0, 3, 1, 2)

        return blurred


    def RSBlurPipeline_for_W_randomBayer(self, blurred_pt, sat_mask_pt, red_gain, blue_gain, beta1, beta2, alpha_saturation,
                             vinetting_g_W):

        blurred_pt = blurred_pt.permute(0, 2, 3, 1)
        sat_mask_pt = sat_mask_pt.permute(0, 2, 3, 1)

        batch_size, _, _, _ = blurred_pt.shape

        # inverse tone mapping
        blurred_L = rgb2lin_pt(blurred_pt)

        # saturation synthesis
        blurred_L = blurred_L + (alpha_saturation.view(batch_size, 1, 1, 1) * sat_mask_pt)
        blurred_L = torch.clamp(blurred_L, 0, 1)

        blurred_sat = blurred_L.clone()

        # from linear RGB to XYZ
        img_XYZ = lin2xyz(blurred_L, self.M_lin2xyz)  # XYZ

        # from XYZ to Cam
        img_Cam = apply_cmatrix(img_XYZ, self.xyz2cam_W)

        # Mosaic
        bayer_pattern = random.choice(['RGGB', 'BGGR', 'GRBG', 'GBRG'])
        img_mosaic = mosaic_bayer(img_Cam, bayer_pattern)

        # inverse white balance
        img_mosaic = WB_img(img_mosaic, bayer_pattern, 1 / red_gain, 1 / blue_gain)

        # inverse vignetting correction
        vignetting_mosaic = mosaic_bayer(vinetting_g_W.unsqueeze(3).repeat(1, 1, 1, 3), bayer_pattern)
        img_mosaic = img_mosaic * (1 / vignetting_mosaic)

        # -------- ADDING POISSON-GAUSSIAN NOISE ON RAW -
        img_mosaic_noise = add_Poisson_noise_random(img_mosaic, beta1, beta2)
        # vignetting correction
        img_mosaic_noise = img_mosaic_noise * vignetting_mosaic

        # -------- ISP PROCESS --------------------------
        # White balance
        img_demosaic = WB_img(img_mosaic_noise, bayer_pattern, red_gain, blue_gain)

        # demosaic
        img_demosaic = torch.nn.functional.pixel_shuffle(img_demosaic.permute(0, 3, 1, 2), 2)
        if bayer_pattern == 'RGGB':
            img_demosaic = self.demosaic[0](img_demosaic).permute(0, 2, 3, 1)
        elif bayer_pattern == 'BGGR':
            img_demosaic = self.demosaic[1](img_demosaic).permute(0, 2, 3, 1)
        elif bayer_pattern == 'GRBG':
            img_demosaic = self.demosaic[2](img_demosaic).permute(0, 2, 3, 1)
        elif bayer_pattern == 'GBRG':
            img_demosaic = self.demosaic[3](img_demosaic).permute(0, 2, 3, 1)

        # from Cam to XYZ
        img_IXYZ = apply_cmatrix(img_demosaic, self.cam2xyz_W)

        # frome XYZ to linear RGB
        img_IL = xyz2lin(img_IXYZ, self.M_xyz2lin)

        # tone mapping
        img_Irgb = lin2rgb_pt(img_IL)
        img_Irgb = torch.clamp(img_Irgb, 0, 1)  # (h, w, c)

        blurred = img_Irgb

        # don't add noise on saturated region
        sat_region = torch.ge(blurred_sat, 1.0)
        non_sat_region = torch.logical_not(sat_region)
        blurred = (blurred_sat * sat_region) + (blurred * non_sat_region)

        blurred = blurred.permute(0, 3, 1, 2)

        return blurred

    def RSBlurPipeline_for_UW(self, blurred_pt, red_gain, blue_gain, beta1, beta2, vinetting_g_UW):

        blurred_pt = blurred_pt.permute(0, 2, 3, 1)

        batch_size, _, _, _ = blurred_pt.shape

        # inverse tone mapping
        blurred_L = rgb2lin_pt(blurred_pt)
        blurred_sat = blurred_L.clone()

        # from linear RGB to XYZ
        img_XYZ = lin2xyz(blurred_L, self.M_lin2xyz)  # XYZ

        # from XYZ to Cam
        img_Cam = apply_cmatrix(img_XYZ, self.xyz2cam_UW)

        # we skipped mosaic for UW, as low-resolution of UW, there are too many artifacts on UW
        #img_mosaic = mosaic_bayer(img_Cam, bayer_pattern)

        # inverse white balance
        # img_mosaic = WB_img(img_mosaic, bayer_pattern, 1 / red_gain, 1 / blue_gain)
        img_mosaic = img_Cam
        img_mosaic[:, :, :, 0:1] = img_mosaic[:, :, :, 0:1] * (1 / red_gain.view(batch_size, 1, 1, 1))
        img_mosaic[:, :, :, 2:3] = img_mosaic[:, :, :, 2:3] * (1 / blue_gain.view(batch_size, 1, 1, 1))

        # inverse vignetting correction
        vignetting_mosaic = vinetting_g_UW.unsqueeze(3).repeat(1, 1, 1, 3)
        img_mosaic = img_mosaic * (1 / vignetting_mosaic)

        # -------- ADDING POISSON-GAUSSIAN NOISE ON RAW -
        img_mosaic_noise = add_Poisson_noise_random(img_mosaic, beta1, beta2)

        # -------- ISP PROCESS --------------------------
        img_demosaic = img_mosaic_noise * vignetting_mosaic

        img_demosaic = img_demosaic
        img_demosaic[:, :, :, 0:1] = img_demosaic[:, :, :, 0:1] * (red_gain.view(batch_size, 1, 1, 1))
        img_demosaic[:, :, :, 2:3] = img_demosaic[:, :, :, 2:3] * (blue_gain.view(batch_size, 1, 1, 1))

        # img_demosaic = torch.nn.functional.pixel_shuffle(img_demosaic.permute(0, 3, 1, 2), 2) # (b, 1, h, w)
        # if bayer_pattern == 'RGGB':
        #     img_demosaic = self.demosaic[0](img_demosaic)
        # elif bayer_pattern == 'BGGR':
        #     img_demosaic = self.demosaic[1](img_demosaic)
        # elif bayer_pattern == 'GRBG':
        #     img_demosaic = self.demosaic[2](img_demosaic)
        # elif bayer_pattern == 'GBRG':
        #     img_demosaic = self.demosaic[3](img_demosaic)
        img_demosaic = torch.clamp(img_demosaic, 0, 1) # (b, c, h, w)


        # from Cam to XYZ
        img_IXYZ = apply_cmatrix(img_demosaic, self.cam2xyz_UW)

        # frome XYZ to linear RGB
        img_IL = xyz2lin(img_IXYZ, self.M_xyz2lin)

        # tone mapping
        img_Irgb = lin2rgb_pt(img_IL)
        img_Irgb = torch.clamp(img_Irgb, 0, 1)  # (h, w, c)

        blurred = img_Irgb

        # don't add noise on saturated region
        sat_region = torch.ge(blurred_sat, 1.0)
        non_sat_region = torch.logical_not(sat_region)
        blurred = (blurred_sat * sat_region) + (blurred * non_sat_region)

        blurred = blurred.permute(0, 3, 1, 2)

        return blurred

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None


        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        optim_params_warmup = []


        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if k.startswith('module.alignment_net'):
                    if not self.opt['network_g']['train_alignment']:
                        v.requires_grad = False
                    else:
                        print(k)
                    optim_params_warmup.append(v)
                else:
                    optim_params.append(v)

        if self.opt['network_g']['train_alignment']:
            # for raft params, set smaller lr
            params = [{'params': optim_params,
                        "lr": self.opt['train']['optim_g']['lr'],
                        "weight_decay":self.opt['train']['optim_g']['weight_decay'],
                        "betas":self.opt['train']['optim_g']['betas']},\
                      {'params': optim_params_warmup,
                        "lr": self.opt['train']['RAFT']['lr'],
                        "weight_decay":self.opt['train']['RAFT']['weight_decay']}]
        else:
            params = [{'params': optim_params,
                        "lr": self.opt['train']['optim_g']['lr'],
                        "weight_decay":self.opt['train']['optim_g']['weight_decay'],
                        "betas":self.opt['train']['optim_g']['betas']}]

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(params,
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(params,
                                                **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(params,
                                                **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)


    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()

        # set up warm-up learning rate of RAFT
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append(
                    [v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate

            # we only warmup RAFT optimizer
            warm_up_lr_l[0][0] = init_lr_g_l[0][0]

            self._set_lr(warm_up_lr_l)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)

        self.deblur_imgs = data['lq_deblur'].to(self.device)

        self.short_seqs = data['short_seqs'].to(self.device)

        self.short_seqs_mask = data['short_seqs_mask'].to(self.device)

        self.seqs_flow = data['seqs_flow'].to(self.device)


        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if 'img_sat_mask' in data:
            self.img_sat_mask = data['img_sat_mask'].to(self.device)

        if 'alpha_saturation' in data:
            self.alpha_saturation = data['alpha_saturation'].to(self.device).float()

        if 'red_gain_W' in data:
            self.red_gain_W = data['red_gain_W'].to(self.device)

        if 'blue_gain_W' in data:
            self.blue_gain_W = data['blue_gain_W'].to(self.device)

        if 'beta1_W' in data:
            self.beta1_W = data['beta1_W'].to(self.device)

        if 'beta2_W' in data:
            self.beta2_W = data['beta2_W'].to(self.device)

        if 'vignetting_g_W' in data:
            self.vignetting_g_W = data['vignetting_g_W'].to(self.device)

        # for UW
        if 'vignetting_g_UW' in data:
            self.vignetting_g_UW = data['vignetting_g_UW'].to(self.device)

        if 'red_gain_UW' in data:
            self.red_gain_UW = data['red_gain_UW'].to(self.device)

        if 'blue_gain_UW' in data:
            self.blue_gain_UW = data['blue_gain_UW'].to(self.device)

        if 'beta1_UW' in data:
            self.beta1_UW = data['beta1_UW'].to(self.device)

        if 'beta2_UW' in data:
            self.beta2_UW = data['beta2_UW'].to(self.device)


        if not is_val:
            # Only for the training process
            # For reducing warping artifacts in boundary, we extract features of larger resolution and crop features after warping.

            h_lq, w_lq = self.opt['datasets']['train']['inp_size']//6, self.opt['datasets']['train']['inp_size']//6
            lq_patch_size = self.opt['datasets']['train']['gt_size']//6

            gt_patch_size = self.opt['datasets']['train']['gt_size']

            lq_top = torch.randint(0, h_lq - lq_patch_size + 1, size=(1,)).item()
            lq_left = torch.randint(0, w_lq - lq_patch_size + 1, size=(1,)).item()

            self.crop = [lq_top, lq_left, lq_patch_size]

            scale = 6
            top_gt, left_gt = int(lq_top * scale), int(lq_left * scale)
            self.gt = self.gt[:,:,top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size]

        if not is_val:
            b, c, h, w = self.lq.shape

            # on-the-fly RSBlur pipeline on wide images
            with torch.no_grad():
                if b >= 4:
                    rsblur_lq = self.RSBlurPipeline_for_W(self.lq, self.img_sat_mask, self.red_gain_W, self.blue_gain_W, self.beta1_W, self.beta2_W,
                                                       self.alpha_saturation, self.vignetting_g_W)
                    self.lq = rsblur_lq
                else:
                    rsblur_lq = self.RSBlurPipeline_for_W_randomBayer(self.lq, self.img_sat_mask, self.red_gain_W, self.blue_gain_W, self.beta1_W, self.beta2_W,
                                                       self.alpha_saturation, self.vignetting_g_W)
                    self.lq = rsblur_lq

        if not is_val:
            # on-the-fly RSBlur pipeline on ultra-wide images
            with torch.no_grad():
                beta1_UW = torch.max(self.beta1_UW, torch.ones_like(self.beta1_UW) * 0.00000000001)
                beta2_UW = torch.max(self.beta2_UW, torch.ones_like(self.beta2_UW) * 0.00000000001)

                batch_size, c, seqs_size, h, w = self.short_seqs.shape

                short_seqs = self.short_seqs.permute(0, 2, 1, 3, 4)  # b, t, c, h, w
                short_seqs = short_seqs.reshape(batch_size * seqs_size, c, h, w)

                red_gain_UW = self.red_gain_UW.view(batch_size, 1).repeat(1, seqs_size).view(-1, 1)
                blue_gain_UW = self.blue_gain_UW.view(batch_size, 1).repeat(1, seqs_size).view(-1, 1)
                beta1_UW = beta1_UW.view(batch_size, 1).repeat(1, seqs_size).view(-1)
                beta2_UW = beta2_UW.view(batch_size, 1).repeat(1, seqs_size).view(-1)
                vignetting_g_UW = self.vignetting_g_UW.view(batch_size, 1, h, w).repeat(1, seqs_size, 1, 1).view(
                    batch_size * seqs_size, h, w)

                rsblur_seqs = self.RSBlurPipeline_for_UW(short_seqs, red_gain_UW, blue_gain_UW, beta1_UW, beta2_UW, vignetting_g_UW)
                noisy_short_seqs = rsblur_seqs.reshape(batch_size, seqs_size, c, h, w).permute(0, 2, 1, 3, 4)

                # we only synthesis noise on ISO_uw <= 400
                flag = (self.beta1_UW > 0)
                self.short_seqs = (noisy_short_seqs * flag.view(-1, 1, 1, 1, 1)) + (self.short_seqs * (~flag).view(-1, 1, 1, 1, 1))



    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)


        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        #adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i//scale*scale
        step_j = step_j//scale*scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(self.lq[:, :, i // scale :(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()


        preds = self.net_g(self.lq, self.deblur_imgs, self.short_seqs, self.short_seqs_mask, self.seqs_flow, self.crop)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)

            # print('l pix ... ', l_pix)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        l_total = l_total #+ 0. * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                pred = self.net_g(self.lq[i:j], self.deblur_imgs[i:j], self.short_seqs[i:j], self.short_seqs_mask[i:j], self.seqs_flow[i:j])
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data, is_val=True)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                #del self.gt



            torch.cuda.empty_cache()

            if save_img:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    # visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
                    visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)

                    imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                    imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                else:
                    if self.opt['is_train']:

                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}.png')

                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')

                    imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics
        
        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        torch.distributed.reduce(metrics, dst=0)
        if self.opt['rank'] == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                               tb_logger, metrics_dict)
        return 0.

    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(*args, **kwargs)


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

