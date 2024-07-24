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
from basicsr.data.HCBlur_utils import ptsWtoUW, ptsUWtoW, interpolate_grid, coords_grid


class HCBlur_with_K_dataset(data.Dataset):
    def __init__(self, opt):
        super(HCBlur_with_K_dataset, self).__init__()
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


        self.depth2grid, self.depth2UWdepth_dict = self.getGridFromDepth(self.K_W_pt, self.K_UW_pt, self.E_pt, unique_depth_pt, unique_depth_list)

        self.W_ori_grid = coords_grid(unique_depth_pt.shape[0], self.W_Height, self.W_Width, homogeneous=False)  # [B, 2, H, W]

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
        short_seqs_all = glob.glob(short_dir + '/') # two samples


        if len(short_seqs_all) == 0:
            print(short_dir, short_seqs_all)

        if self.opt['phase'] == 'train':
            short_seq_dir = random.choice(short_seqs_all)
            #short_seq_dir = short_seqs_all[0]
        else:
            short_seq_dir = short_seqs_all[0]

        short_seq_imgs = glob.glob(short_seq_dir + '/*.jpg')
        short_seq_centered_exp = [self.getShortUWCenterTime(path) for path in short_seq_imgs]

        long_center_time = self.getLongWCenterTime(lq_path)

        short_center_index = np.argmin(np.abs(np.array(short_seq_centered_exp) - long_center_time))
        short_center_path = short_seq_imgs[short_center_index]

        return short_center_path, short_seq_dir

    def getFlowPath(self, short_seq_dir):

        flow_dir = short_seq_dir.replace(self.uw_folder, self.flows_folder).replace('UWseqs', 'UWflows')
        flow_path = os.path.join(flow_dir, 'centerized_accum_flows.npy')

        return flow_path


    def __getitem__(self, index):


        scale = self.opt['scale']

        # Load gt and lq images.
        gt_path = self.paths[index]['gt_path']
        img_gt = cv2.imread(gt_path).astype('float32') / 255.0
        img_gt = img_gt[:,:,::-1] # BGR2RGB

        lq_path = self.paths[index]['lq_path']
        img_lq = cv2.imread(lq_path).astype('float32') / 255.0
        img_lq = img_lq[:, :, ::-1]  # BGR2RGB

        short_flow = None
        target_grid = self.W_ori_grid
        W_ori_grid = self.W_ori_grid


        short_center_path, short_seq_dir = self.getCenterFramePath(lq_path)
        if self.opt['use_K']:

            # load optical flows
            flow_path = self.getFlowPath(short_seq_dir)
            short_flow = np.load(flow_path)  # (burst_len, 2, H, W)

            # extract center timestamps of uw frames
            short_center_exp = []
            short_seqs_path = glob.glob(short_seq_dir + '/*.jpg')
            short_seqs_path = sorted(short_seqs_path)
            for path in short_seqs_path:
                name_split = path.split('/')[-1][:-4].split('_')
                center_time = int(name_split[0]) + (int(name_split[1]) - int(name_split[0])) // 2
                short_center_exp.append(center_time)

            # extract timestamps from name of images
            W_name = lq_path.split('/')[-1]
            W_name_split = W_name.split('_')
            start_timestamp_W = float(W_name_split[0])
            end_timestamp_W = float(W_name_split[2])

            short_flow_pt = torch.from_numpy(short_flow)
            with torch.no_grad():
                # interpolated to fixed temporal sizes
                cropped_interpolated_flow_pt = interpolate_grid(short_flow_pt, start_timestamp_W, end_timestamp_W, short_center_exp)

                ones = torch.ones_like(cropped_interpolated_flow_pt[:, 0:1, :, :])
                homogenous_accum_flows = torch.cat([cropped_interpolated_flow_pt, ones], dim=1)

                flow_dim, flow_c, flow_h, flow_w = homogenous_accum_flows.shape

                flow_path_split = flow_path.split('/')
                key = '%s_%s_%s' % (flow_path_split[-5], flow_path_split[-4], flow_path_split[-2])
                W_depth = self.depth_dict[key]
                UW_depth = self.depth2UWdepth_dict[W_depth]
                depth_estimated_grid = ones * UW_depth

                # convert coordinate system
                flowsOnW = ptsUWtoW(homogenous_accum_flows, depth_estimated_grid, self.K_W_pt.repeat(flow_dim, 1, 1),
                                    self.K_UW_pt.repeat(flow_dim, 1, 1), self.E_pt.repeat(flow_dim, 1, 1))
                flowsOnW = flowsOnW.view(flow_dim, 2, flow_h, flow_w)

                target_grid = self.depth2grid[W_depth] # (1, self.W_Height, self.W_Width, 2)

                crop_H = (1280 - 840) // 2
                crop_W = (720 - 560) // 2
                padding = [crop_W, crop_W, crop_H, crop_H]
                flowsOnW = F.pad(flowsOnW, padding)

        # for faster training, we can load pre-saved kernels from a disk.
        if self.opt['use_interpolated_K']:
            flow_path = self.getFlowPath(short_seq_dir)
            flow_path = flow_path.replace('centerized_accum_flows.npy', 'interpolated_centerized_accum_flow.npy')
            flowsOnW = torch.from_numpy(np.load(flow_path))  # (burst_len, 2, H, W)

            flow_path_split = flow_path.split('/')
            key = '%s_%s_%s' % (flow_path_split[-5], flow_path_split[-4], flow_path_split[-2])
            W_depth = self.depth_dict[key]
            target_grid = self.depth2grid[W_depth]  # (1, self.W_Height, self.W_Width, 2)

            crop_H = (1280 - 840) // 2
            crop_W = (720 - 560) // 2
            padding = [crop_W, crop_W, crop_H, crop_H]
            flowsOnW = F.pad(flowsOnW, padding)

            flow_dim, flow_c, flow_h, flow_w = flowsOnW.shape


        if self.opt['RSBlur']:
            # load saturation mask
            lq_path = self.paths[index]['lq_path']
            sat_mask_path = lq_path.replace('_blur.png', '_satmask.png')
            img_sat_mask = cv2.imread(sat_mask_path).astype('float32') / 255.0
            img_sat_mask = img_sat_mask[:, :, ::-1]  # BGR2RGB
        else:
            img_sat_mask = np.zeros_like(img_lq)

        vinetting_g_W = self.vinetting_g_W
        vinetting_g_UW = self.vinetting_g_UW

        # augmentation for training
        # for reducing the burden of dataloder, we crop the warping map (taget_grid) and warp blur kernels later.
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            lq_sat_mat_vinetting = [img_lq, img_sat_mask, vinetting_g_W]
            img_gt, lq_sat_mat_vinetting, grids = Input_random_crop_with_grid(img_gt, lq_sat_mat_vinetting, [target_grid, W_ori_grid.permute(0, 2, 3, 1)],
                                                                            gt_size, scale,
                                                                            gt_path)

            target_grid = grids[0]
            # [B, H, W, 2] => [B, 2, H, W]
            W_ori_grid = grids[1].permute(0, 3, 1, 2)

            img_lq = lq_sat_mat_vinetting[0]
            img_sat_mask = lq_sat_mat_vinetting[1]
            vinetting_g_W = lq_sat_mat_vinetting[2]

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

            # saturration synthesis parameter
            alpha_saturation = random.uniform(0.25, 2.75)

            alpha_saturation = torch.tensor(alpha_saturation)
            red_gain_W = torch.tensor([redGain_W]).float()
            blue_gain_W = torch.tensor([blueGain_W]).float()

            beta1_W = torch.tensor(beta1_W).float()
            beta2_W = torch.tensor(beta2_W).float()
        else:
            alpha_saturation = torch.tensor(0)
            red_gain_W = torch.tensor([0]).float()
            blue_gain_W = torch.tensor([0]).float()

            beta1_W = torch.tensor(0).float()
            beta2_W = torch.tensor(0).float()

        if self.opt['use_K'] or self.opt['use_interpolated_K']:

            _, h, w, c = target_grid.shape

            # warp optical flows according to the cropped warping map
            warped_flowsOnW = F.grid_sample(flowsOnW.float(), target_grid.repeat(flow_dim, 1, 1, 1).view(flow_dim, h, w, 2),
                                            mode='bilinear',
                                            padding_mode='zeros',
                                            align_corners=True).view(flow_dim, 2, h, w)  # [B, C, D, H, W]

            short_flow_pt = warped_flowsOnW.clone()
            short_flow_pt = short_flow_pt - W_ori_grid
            short_flow = short_flow_pt.numpy()

        # augmentation for training
        if self.opt['phase'] == 'train':
            # flip, rotation for images and flows
            [img_gt, img_lq, img_sat_mask, vinetting_g_W], short_flow = Augment_with_flows([img_gt, img_lq, img_sat_mask, vinetting_g_W[:,:,None]], self.opt['use_flip'],
                                                                  self.opt['use_rot'], short_flow)

            vinetting_g_W = vinetting_g_W[:,:,0]

        img_sat_mask = torch.from_numpy(img_sat_mask.copy().transpose(2, 0, 1))
        vinetting_g_W = torch.from_numpy(vinetting_g_W.copy())


        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt.copy(), img_lq.copy()],
                                    bgr2rgb=False,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        if self.opt['use_K'] or self.opt['use_interpolated_K']:
            short_flow = torch.from_numpy(short_flow.copy())

            # relative coordinates
            if self.opt['normalize_K']:
                short_flow = short_flow - short_flow[4:5,:,:,:]

            exp_len, c, h, w = short_flow.shape
            short_flow = short_flow.reshape(exp_len * c, h, w).half()

        if short_flow == None:
            c, h, w = img_gt.shape
            short_flow = torch.zeros([18, h, w]).half()

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'kernel': short_flow,

            'img_sat_mask': img_sat_mask,
            'red_gain_W': red_gain_W,
            'blue_gain_W': blue_gain_W,
            'beta1_W': beta1_W,
            'beta2_W': beta2_W,
            'alpha_saturation': alpha_saturation,
            'vignetting_g_W' : vinetting_g_W
        }

    def __len__(self):
        return len(self.paths)

