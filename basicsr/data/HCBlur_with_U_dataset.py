from torch.utils import data as data
from torchvision.transforms.functional import normalize
import random

from basicsr.data.data_util import paired_paths_from_list
from basicsr.data.transforms import Augment_with_flows, Input_random_crop_with_grid, mod_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
import cv2
import glob
import numpy as np
import torch
import scipy.io
import os
import torch.nn.functional as F
from basicsr.data.RSBlur_utils import random_noise_UW, random_noise_W
from basicsr.data.HCBlur_utils import ptsWtoUW, ptsUWtoW, coords_grid

class HCBlur_with_U_dataset(data.Dataset):
    def __init__(self, opt):
        super(HCBlur_with_U_dataset, self).__init__()
        self.opt = opt

        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.flows_folder, self.uw_folder = opt['dataroot_flows'], opt['dataroot_uws']
        self.paths = paired_paths_from_list([opt['dataroot_lq'], opt['dataroot_gt']], ['lq', 'gt'], opt['datalist'])

        # load stereoParams
        K_W = scipy.io.loadmat('mat_collections/K.mat')['K'].astype('float32').transpose(1, 0)
        K_UW = scipy.io.loadmat('mat_collections/K2.mat')['K2'].astype('float32').transpose(1, 0)
        E = scipy.io.loadmat('mat_collections/E2.mat')['E2'].astype('float32').transpose(1, 0)

        self.K_W_pt = torch.from_numpy(K_W).unsqueeze(0)
        self.K_UW_pt = torch.from_numpy(K_UW).unsqueeze(0)
        self.E_pt = torch.from_numpy(E).unsqueeze(0)

        self.depth_dict, unique_depth_list = self.getAllEstimatedDepth(self.lq_folder)

        self.UW_Height, self.UW_Width = 1280, 720
        self.W_Height, self.W_Width = 3840, 2160

        unique_depth_np = np.array([float(depth) for depth in unique_depth_list]).astype('float32')
        unique_depth_pt = torch.from_numpy(unique_depth_np.copy())
        unique_depth_pt = unique_depth_pt.view(1, unique_depth_pt.shape[0], 1, 1).repeat(1, 1, self.W_Height, self.W_Width)  # [B, D, H, W]

        self.depth2grid, self.depth2UWdepth_dict = self.getGridFromDepth(self.K_W_pt, self.K_UW_pt, self.E_pt,
                                                                         unique_depth_pt, unique_depth_list)

        self.W_ori_grid = coords_grid(1, self.W_Height, self.W_Width, homogeneous=False,
                           device=unique_depth_pt.device)  # [B, 2, H, W]

        self.UW_ori_grid = coords_grid(1, self.UW_Height, self.UW_Width, homogeneous=False,
                           device=unique_depth_pt.device)  # [B, 2, H, W]

        # rsblur paramerters
        vinetting_g_W_np = scipy.io.loadmat('./mat_collections/g_W_RSBlur.mat')['g'].astype('float32')
        vinetting_g_UW_np = scipy.io.loadmat('./mat_collections/g_UW_RSBlur.mat')['g'].astype('float32')

        self.vinetting_g_W = vinetting_g_W_np.copy()
        self.vinetting_g_UW = vinetting_g_UW_np.copy()

        # load iso, wb params for isp simulation
        metadata_root = 'datalist/allMetadata.txt'
        with open(metadata_root, 'rt') as f:
            medadata_list = f.readlines()

        self.video_name2meta = {}
        for metadata_line in medadata_list:
            video_name, redGain_W, blueGain_W, ISO_W, redGain_UW, blueGain_UW, ISO_UW = metadata_line.strip().split(' ')
            self.video_name2meta[video_name] = [float(value) for value in
                                                [redGain_W, blueGain_W, ISO_W, redGain_UW, blueGain_UW, ISO_UW]]

    def getGridFromDepth(self, K_W_pt, K_UW_pt, E_pt, unique_depth_pt, unique_depth):
        depth2grid_dict = {}
        depth2UWdepth_dict = {}
        with torch.no_grad():
            grid = coords_grid(unique_depth_pt.shape[0], self.W_Height, self.W_Width, homogeneous=True,
                               device=None)  # [B, 3, H, W]
            # calibration matrix based on matlab, so we convert matlab's coordinate (0~h-1) => (1~h)
            pixel_coords, depth_uw = ptsWtoUW(grid, unique_depth_pt, K_W_pt, K_UW_pt, E_pt)

            # normalize to [-1, 1]
            x_grid = 2 * pixel_coords[:, 0] / (self.UW_Width - 1) - 1
            y_grid = 2 * pixel_coords[:, 1] / (self.UW_Height - 1) - 1

            out_grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

        for i in range(out_grid.shape[1]):
            depth2grid_dict[unique_depth[i]] = out_grid[:, i, :, :].view(1, self.W_Height, self.W_Width, 2).clone() # [B, H, W, 2]
            depth2UWdepth_dict[unique_depth[i]] = depth_uw[:,i]

        return depth2grid_dict, depth2UWdepth_dict

    def getAllEstimatedDepth(self, lq_folder):

        unique_depth_list = []
        depth_list = glob.glob(os.path.join(lq_folder, 'shortUW_depth/**/*.txt'))
        depth_dict = {}
        for txt_path in depth_list:
            video_name = txt_path.split('/')[-1][:-10]

            with open(txt_path, 'rt') as f:
                depth_txt = f.readlines()

            for txt in depth_txt:
                split_txt = txt.strip().split(' ')
                sample_name = split_txt[0]
                seqs_name = split_txt[1]
                depth = float(split_txt[2])

                new_key = '%s_%s_%s' % (video_name, sample_name, seqs_name)
                depth_dict[new_key] = '%.2f' % depth

                if '%.2f' % depth not in unique_depth_list:
                    unique_depth_list.append('%.2f' % depth)

        return depth_dict, unique_depth_list

    def getShortUWCenterTime(self, path):
        name_split = path.split('/')[-1][:-4].split('_')
        center_time = int(name_split[0]) + (int(name_split[1]) - int(name_split[0])) // 2

        return center_time

    def getLongWCenterTime(self, path):
        name_split = path.split('/')[-1][:-4].split('_')
        center_time = int(name_split[0]) + (int(name_split[2]) - int(name_split[0])) // 2

        return center_time

    def getCenterFramePath(self, lq_path):

        short_dir = '/'.join(lq_path.split('/')[:-3]).replace(self.lq_folder, self.uw_folder).replace('/longW/', '/') + '/UWseqs/*'
        short_seqs_all = glob.glob(short_dir + '/') # three samples

        if len(short_seqs_all) == 0:
            print(short_dir, short_seqs_all)

        if self.opt['phase'] == 'train':
            short_seq_dir = random.choice(short_seqs_all)
        else:
            short_seq_dir = short_seqs_all[0]

        short_seq_imgs = glob.glob(short_seq_dir + '/*.jpg')
        short_seq_imgs = sorted(short_seq_imgs)
        short_seq_centered_exp = [self.getShortUWCenterTime(path) for path in short_seq_imgs]

        long_center_time = self.getLongWCenterTime(lq_path)

        short_center_index = np.argmin(np.abs(np.array(short_seq_centered_exp) - long_center_time))
        short_center_path = short_seq_imgs[short_center_index]

        return short_center_path, short_seq_dir, short_center_index

    def getSequencesPath(self, short_seq_dir):

        seqs_path = glob.glob(short_seq_dir + '/*.jpg')
        seqs_path = sorted(seqs_path)

        return seqs_path

    def getFlowPath(self, short_seq_dir):

        flow_dir = short_seq_dir.replace(self.uw_folder, self.flows_folder).replace('UWseqs', 'UWflows')
        flow_path = os.path.join(flow_dir, 'centerized_accum_flows.npy')

        return flow_path


    def __getitem__(self, index):

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_gt = cv2.imread(gt_path).astype('float32') / 255.0
        img_gt = img_gt[:, :, ::-1]  # BGR2RGB

        lq_path = self.paths[index]['lq_path']
        img_lq = cv2.imread(lq_path).astype('float32') / 255.0
        img_lq = img_lq[:, :, ::-1]  # BGR2RGB

        seqs_flow = None

        target_grid = self.W_ori_grid
        W_ori_grid = self.W_ori_grid


        # load deblurred images of HCDNet
        if self.opt['phase'] == 'train':
            lq_deblur_path = self.paths[index]['lq_path'].replace(self.opt['dataroot_lq'], self.opt['dataroot_deblur']) # (h/6, w/6, c)
            img_lq_deblur = cv2.imread(lq_deblur_path).astype('float32') / 255.0
            img_lq_deblur = img_lq_deblur[:, :, ::-1]  # BGR2RGB
        else:
            lq_deblur_path = os.path.join(self.opt['dataroot_deblur'], self.paths[index]['lq_path'].split('/')[-1]) # (h/6, w/6, c)
            img_lq_deblur = cv2.imread(lq_deblur_path).astype('float32') / 255.0
            img_lq_deblur = img_lq_deblur[:, :, ::-1]  # BGR2RGB


        short_center_path, short_seq_dir, short_center_index = self.getCenterFramePath(lq_path)
        if self.opt['use_seqs_flow']:
            flow_path = self.getFlowPath(short_seq_dir)
            short_flow = np.load(flow_path)  # (burst_len, 2, H, W)

            short_flow_pt = torch.from_numpy(short_flow) # (burst_len, 2, H, W)
            seqs_flow = short_flow_pt.clone()

            ones = torch.ones_like(seqs_flow[:, 0:1, :, :])
            homogenous_seqs_flow = torch.cat([seqs_flow, ones], dim=1)

            flow_dim, flow_c, flow_h, flow_w = homogenous_seqs_flow.shape

            flow_path_split = flow_path.split('/')
            key = '%s_%s_%s' % (flow_path_split[-5], flow_path_split[-4], flow_path_split[-2])
            W_depth = self.depth_dict[key]
            UW_depth = self.depth2UWdepth_dict[W_depth]
            depth_estimated_grid = ones * UW_depth

            seqs_flow = ptsUWtoW(homogenous_seqs_flow.float(), depth_estimated_grid, self.K_W_pt.repeat(flow_dim, 1, 1),
                                self.K_UW_pt.repeat(flow_dim, 1, 1), self.E_pt.repeat(flow_dim, 1, 1))
            seqs_flow = seqs_flow.view(flow_dim, 2, flow_h, flow_w)

            target_grid = self.depth2grid[W_depth] # (1, self.W_Height, self.W_Width, 2)

            crop_H = (1280 - 840) // 2
            crop_W = (720 - 560) // 2
            padding = [crop_W, crop_W, crop_H, crop_H]
            seqs_flow = F.pad(seqs_flow, padding)


        if self.opt['use_short_sequences']:
            short_paths = self.getSequencesPath(short_seq_dir)
            short_seqs = [cv2.imread(path).astype('float32') / 255.0 for path in short_paths] # (n_seqs, H, W, C)
            short_seqs = [short_img[:,:,::-1] for short_img in short_seqs]
            short_seqs_pt = torch.from_numpy(np.array(short_seqs)).permute(0, 3, 1, 2)

            short_center_path_split = short_center_path.split('/')

            key = '%s_%s_%s' % (short_center_path_split[-5], short_center_path_split[-4], short_center_path_split[-2])
            W_depth = self.depth_dict[key]
            target_grid = self.depth2grid[W_depth]  # (1, self.W_Height, self.W_Width, 2)


        if self.opt['RSBlur']:
            lq_path = self.paths[index]['lq_path']
            sat_mask_path = lq_path.replace('_blur.png', '_satmask.png')
            img_sat_mask = cv2.imread(sat_mask_path).astype('float32') / 255.0
            img_sat_mask = img_sat_mask[:, :, ::-1]  # BGR2RGB
        else:
            img_sat_mask = np.zeros_like(img_lq)

        vinetting_g_W = self.vinetting_g_W
        crop_H = (1280 - 640) // 2
        crop_W = (720 - 360) // 2
        vinetting_g_UW = self.vinetting_g_UW[crop_H:-crop_H, crop_W:-crop_W]

        # augmentation for training
        # for reducing the burden of dataloder, we crop the warping map (taget_grid) and warp optical flows and ultra-wide frames later.
        if self.opt['phase'] == 'train':
            gt_size = self.opt['inp_size']
            # random crop
            # W_ori_grid [B, 2, H, W] => [B, H, W, 2]


            lq_and_gt_satmask_deblur = [img_lq, img_gt, img_sat_mask, vinetting_g_W, img_lq_deblur]
            lq_and_gt_satmask_deblur, vinetting_g_UW, grids = Input_random_crop_with_grid(lq_and_gt_satmask_deblur, vinetting_g_UW[:,:,None], [target_grid, W_ori_grid.permute(0, 2, 3, 1)],
                                                                            gt_size, 6,
                                                                            gt_path)

            img_lq = lq_and_gt_satmask_deblur[0]
            img_gt = lq_and_gt_satmask_deblur[1]
            img_sat_mask = lq_and_gt_satmask_deblur[2]
            vinetting_g_W = lq_and_gt_satmask_deblur[3]
            img_lq_deblur = lq_and_gt_satmask_deblur[4]

            vinetting_g_UW = vinetting_g_UW[:,:,0]

            target_grid = grids[0]
            # [B, H, W, 2] => [B, 2, H, W]
            W_ori_grid = grids[1].permute(0, 3, 1, 2)



        if self.opt['RSBlur']:
            video_name = lq_path.split('/')[-5]
            # load metadata and sample noise params
            redGain_W, blueGain_W, ISO_W, redGain_UW, blueGain_UW, ISO_UW = self.video_name2meta[video_name]
            if ISO_UW < 400:
                iso_W = random.uniform(100, 800)
                beta1_W, beta2_W = random_noise_W(iso_W)

                iso_UW = random.uniform(1, 4) * iso_W
                beta1_UW, beta2_UW = random_noise_UW(iso_UW)
            else:
                iso_W = ISO_UW * random.uniform(1, 1 / 4)
                beta1_W, beta2_W = random_noise_W(iso_W)

                iso_UW = 0
                beta1_UW, beta2_UW = 0, 0

            alpha_saturation = random.uniform(0.25, 2.75)

            alpha_saturation = torch.tensor(alpha_saturation)
            red_gain_W = torch.tensor([redGain_W]).float()
            blue_gain_W = torch.tensor([blueGain_W]).float()

            beta1_W = torch.tensor(beta1_W).float()
            beta2_W = torch.tensor(beta2_W).float()

            red_gain_UW = torch.tensor([redGain_UW]).float()
            blue_gain_UW = torch.tensor([blueGain_UW]).float()

            beta1_UW = torch.tensor(beta1_UW).float()
            beta2_UW = torch.tensor(beta2_UW).float()
        else:
            alpha_saturation = torch.tensor(0)
            red_gain_W = torch.tensor([0]).float()
            blue_gain_W = torch.tensor([0]).float()

            beta1_W = torch.tensor(0).float()
            beta2_W = torch.tensor(0).float()

            red_gain_UW = torch.tensor([0]).float()
            blue_gain_UW = torch.tensor([0]).float()

            beta1_UW = torch.tensor(0).float()
            beta2_UW = torch.tensor(0).float()

        target_grid_ori = target_grid.clone()
        target_grid = F.interpolate(target_grid.permute(0, 3, 1, 2), scale_factor=1 / 6,
                                    mode='bilinear',
                                    align_corners=True, antialias=False).permute(0, 2, 3, 1)
        W_ori_grid = F.interpolate(W_ori_grid, scale_factor=1 / 6, mode='bilinear',
                                   align_corners=True, antialias=False)

        if self.opt['use_seqs_flow']:
            _, h, w, c = target_grid.shape
            flow_dim, _, _, _ = seqs_flow.shape

            # warp optical flows according to the cropped warping map
            warped_seqs_flow= F.grid_sample(seqs_flow, target_grid.repeat(flow_dim, 1, 1, 1).view(flow_dim, h, w, 2),
                                            mode='bilinear',
                                            padding_mode='zeros',
                                            align_corners=True).view(flow_dim, 2, h, w)  # [B, C, H, W]

            seqs_flow = warped_seqs_flow.clone()
            seqs_flow = (seqs_flow - W_ori_grid) * 1/6
            seqs_flow = seqs_flow.numpy()


        if self.opt['use_short_sequences']:
            _, h, w, c = target_grid.shape

            seqs, _, _, _ = short_seqs_pt.shape

            # warp ultra-wide images according to the cropped warping map
            warped_short_seqs_pt = F.grid_sample(short_seqs_pt, target_grid.view(1, h, w, 2).repeat(seqs, 1, 1, 1),
                                            mode='bicubic',
                                            padding_mode='zeros',
                                            align_corners=True)  # [B, C, D, H, W]

            warped_short_seqs = warped_short_seqs_pt.permute(0, 2, 3, 1).numpy()
            warped_short_seqs_list = np.split(warped_short_seqs, seqs)
            warped_short_seqs_list = [arr[0,:,:,:] for arr in warped_short_seqs_list]

        # augmentation for training
        if self.opt['phase'] == 'train':
            # flip, rotation
            flow_list = [seqs_flow]

            all_list = [img_gt, img_lq, img_lq_deblur, img_sat_mask, vinetting_g_W[:,:,None], vinetting_g_UW[:,:,None]] + warped_short_seqs_list
            all_list, flow_list = Augment_with_flows(all_list, self.opt['use_flip'],
                                                                  self.opt['use_rot'], flow_list)

            img_gt = all_list[0]
            img_lq = all_list[1]
            img_lq_deblur = all_list[2]
            img_sat_mask = all_list[3]
            vinetting_g_W = all_list[4][:,:,0]
            vinetting_g_UW = all_list[5][:,:,0]

            warped_short_seqs_list = all_list[6:]

            seqs_flow = flow_list[0]

        img_sat_mask = torch.from_numpy(img_sat_mask.copy().transpose(2, 0, 1))
        vinetting_g_W = torch.from_numpy(vinetting_g_W.copy())
        vinetting_g_UW = torch.from_numpy(vinetting_g_UW.copy())

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt.copy(), img_lq.copy()],
                                    bgr2rgb=False,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)


        if self.opt['use_seqs_flow']:
            seqs_flow = torch.from_numpy(seqs_flow.copy())

            if self.opt['first_center']:

                seqs_flow_list = list(torch.split(seqs_flow, 1, dim=0))
                seqs_flow_list = [temp.squeeze(0) for temp in seqs_flow_list]

                reference_flow = seqs_flow_list.pop(short_center_index)
                seqs_flow_list = [reference_flow] + seqs_flow_list

                seqs_flow = torch.stack(seqs_flow_list, dim=1).half() # (2, T, H, W)
            else:
                seqs_flow = seqs_flow.permute(1, 0, 2, 3)

        if self.opt['use_deblur']:
            img_lq_deblur = img2tensor(img_lq_deblur.copy(),
                                     bgr2rgb=False,
                                     float32=True)

            if self.mean is not None or self.std is not None:
                normalize(img_lq_deblur, self.mean, self.std, inplace=True)

        if self.opt['use_short_sequences']:
            new_seqs_list = [temp.copy() for temp in warped_short_seqs_list]
            warped_short_seqs_list = img2tensor(new_seqs_list,
                                     bgr2rgb=False,
                                     float32=True)

            if self.mean is not None or self.std is not None:
                for i, short_seq in enumerate(warped_short_seqs_list):
                    warped_short_seqs_list[i] = normalize(short_seq, self.mean, self.std, inplace=True) # (c, h, w)
            
            if self.opt['first_center']:
                reference_short = warped_short_seqs_list.pop(short_center_index)
                warped_short_seqs_list = [reference_short] + warped_short_seqs_list

            short_seqs = torch.stack(warped_short_seqs_list, dim=1)  # (C, T, H, W)

        if seqs_flow is None:
            c, h, w = img_gt.shape
            seqs_flow = torch.zeros([2, 1, h, w]).half()

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'seqs_flow': seqs_flow,
            'lq_deblur': img_lq_deblur,
            'short_seqs': short_seqs,
            'short_center_index': short_center_index,

            'img_sat_mask': img_sat_mask,
            'red_gain_W': red_gain_W,
            'blue_gain_W': blue_gain_W,
            'beta1_W': beta1_W,
            'beta2_W': beta2_W,
            'alpha_saturation': alpha_saturation,
            'vignetting_g_W': vinetting_g_W,

            'red_gain_UW': red_gain_UW,
            'blue_gain_UW': blue_gain_UW,
            'beta1_UW': beta1_UW,
            'beta2_UW': beta2_UW,
            'vignetting_g_UW': vinetting_g_UW
        }

    def __len__(self):
        return len(self.paths)

def collect_fn_HCBlur_with_U(batch):

    # due to variable size of U, we padded U to the biggest size of U in batch
    # later, the padded frames are skipped in merging frames.

    blur_list = []
    target_list = []
    lq_path_list = []
    gt_path_list = []
    short_seqs_list = []
    lq_deblur_list = []
    index_list = []

    img_sat_mask_list = []
    red_gain_W_list = []
    blue_gain_W_list = []
    beta1_W_list = []
    beta2_W_list = []
    alpha_saturation_list = []
    vinetting_g_W_list = []

    red_gain_UW_list = []
    blue_gain_UW_list = []
    beta1_UW_list = []
    beta2_UW_list = []
    vignetting_g_UW_list = []
    seqs_flow_list = []

    for item in batch:
        blur_list.append(item['lq'])
        target_list.append(item['gt'])
        lq_path_list.append(item['lq_path'])
        gt_path_list.append(item['gt_path'])
        short_seqs_list.append(item['short_seqs'])
        lq_deblur_list.append(item['lq_deblur'])
        index_list.append(item['short_center_index'])

        img_sat_mask_list.append(item['img_sat_mask'])
        red_gain_W_list.append(item['red_gain_W'])
        blue_gain_W_list.append(item['blue_gain_W'])
        beta1_W_list.append(item['beta1_W'])
        beta2_W_list.append(item['beta2_W'])
        alpha_saturation_list.append(item['alpha_saturation'])
        vinetting_g_W_list.append(item['vignetting_g_W'])

        red_gain_UW_list.append(item['red_gain_UW'])
        blue_gain_UW_list.append(item['blue_gain_UW'])
        beta1_UW_list.append(item['beta1_UW'])
        beta2_UW_list.append(item['beta2_UW'])
        vignetting_g_UW_list.append(item['vignetting_g_UW'])
        seqs_flow_list.append(item['seqs_flow'])

    max_t = 1
    for short_img in short_seqs_list:
        c, t, h, w = short_img.size()
        max_t = max(max_t, t)

    new_mask = []
    new_patch = []
    for short_img in short_seqs_list:
        c, t, h, w = short_img.size()
        padding = (0, 0, 0, 0, 0, max_t - t)
        padded_patch = torch.nn.functional.pad(short_img, padding, "replicate")
        padded_mask = torch.nn.functional.pad(torch.ones([1, t, h, w]).float(), padding, "constant", 0)
        new_patch.append(padded_patch)
        new_mask.append(padded_mask)

    new_flow = []
    for short_flow in seqs_flow_list:
        c, t, h, w = short_flow.size()
        padding = (0, 0, 0, 0, 0, max_t - t)
        padded_patch = torch.nn.functional.pad(short_flow.float(), padding, "replicate")
        new_flow.append(padded_patch.half())

    return {'lq':torch.stack(blur_list, 0),\
            'gt':torch.stack(target_list, 0),\
            'lq_path': lq_path_list, \
            'gt_path': gt_path_list, \
            'short_seqs':torch.stack(new_patch, 0),\
            'short_seqs_mask': torch.stack(new_mask, 0),\
            'seqs_flow': torch.stack(new_flow, 0), \
            'lq_deblur': torch.stack(lq_deblur_list, 0),
            'short_center_index': index_list,

            'img_sat_mask': torch.stack(img_sat_mask_list, 0),
            'red_gain_W': torch.stack(red_gain_W_list),
            'blue_gain_W': torch.stack(blue_gain_W_list),
            'beta1_W': torch.stack(beta1_W_list),
            'beta2_W': torch.stack(beta2_W_list),
            'alpha_saturation': torch.stack(alpha_saturation_list),
            'vignetting_g_W': torch.stack(vinetting_g_W_list, 0),

            'red_gain_UW': torch.stack(red_gain_UW_list),
            'blue_gain_UW': torch.stack(blue_gain_UW_list),
            'beta1_UW': torch.stack(beta1_UW_list),
            'beta2_UW': torch.stack(beta2_UW_list),
            'vignetting_g_UW': torch.stack(vignetting_g_UW_list, 0)
            }