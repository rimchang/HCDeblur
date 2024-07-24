import sys
sys.path.append('core')

import glob
import os
import tqdm
import numpy as np
import torch
import cv2
import argparse
from copy import deepcopy
import torch.nn.functional as F
import scipy.io

from basicsr.models.archs.core.raft import RAFT


parser = argparse.ArgumentParser()
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
parser.add_argument('--raft_small', action='store_true')
parser.add_argument('--save_viz', action='store_true')
parser.add_argument('--root_path')
parser.add_argument('--out_path')
args = parser.parse_args()

longW_root_path = os.path.join(args.root_path, 'longW')
shortUW_root_path = os.path.join(args.root_path, 'shortUW')

flow_out_path = args.out_path

K_W = scipy.io.loadmat('./mat_collections/K.mat')['K'].astype('float32').transpose(1, 0)
K_UW = scipy.io.loadmat('./mat_collections/K2.mat')['K2'].astype('float32').transpose(1, 0)
E = scipy.io.loadmat('./mat_collections/E2.mat')['E2'].astype('float32').transpose(1, 0)

K_W_pt = torch.from_numpy(K_W).unsqueeze(0).cuda()
K_UW_pt = torch.from_numpy(K_UW).unsqueeze(0).cuda()
E_pt = torch.from_numpy(E).unsqueeze(0).cuda()

UW_Height, UW_Width = 1280, 720
W_Height, W_Width = 3840, 2160

######################## load RAFT ########################
DEVICE = 'cuda'

def load_network(net, load_path, strict=True, param_key='params'):
    """Load network.

    Args:
        load_path (str): The path of networks to be loaded.
        net (nn.Module): Network.
        strict (bool): Whether strictly loaded.
        param_key (str): The parameter key of loaded network. If set to
            None, use the root 'path'.
            Default: 'params'.
    """
    load_net = torch.load(
        load_path, map_location=lambda storage, loc: storage)
    if param_key is not None:
        load_net = load_net[param_key]
    print(' load net keys', load_net.keys)
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    net.load_state_dict(load_net, strict=strict)


if args.raft_small:
    model = RAFT(small=True)
    load_network(model, './pretrained_models/raft-small.pth', param_key=None)
else:
    model = RAFT(small=False)
    load_network(model, './pretrained_models/raft-sintel.pth', param_key=None)

model.to(DEVICE)
model.eval()



class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def save_flows(uw_frame_paths, out_dir, model, longW_path):

    timestamp_UW = np.array([float(uw_frame.split('/')[-1].split('_')[1]) for uw_frame in uw_frame_paths])
    start_timestamp_UW = np.array([float(uw_frame.split('/')[-1].split('_')[0]) for uw_frame in uw_frame_paths])
    exposureTime_UW = timestamp_UW[0] - start_timestamp_UW[0]

    centerized_accum_flow = compute_centerized_flow(model, longW_path,
                                                            uw_frame_paths,
                                                            exposureTime_UW)


    centerized_accum_flow_np = centerized_accum_flow.cpu().numpy().astype('float16').copy()
    crop_H = (1280 - 840)//2
    crop_W = (720 - 560)//2
    centerized_accum_flow_np = centerized_accum_flow_np[:,:,crop_H:-crop_H, crop_W:-crop_W]
    # we save flows as float16 for disk usage
    np.save(out_dir + '/centerized_accum_flows.npy', centerized_accum_flow_np)


def image2torch(img):
    img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[np.newaxis].to(DEVICE)


def getShortUWCenterTime(path):
    name_split = path.split('/')[-1][:-4].split('_')
    center_time = int(name_split[0]) + (int(name_split[1]) - int(name_split[0])) // 2

    return center_time


def getLongWCenterTime(path):
    name_split = path.split('/')[-1][:-4].split('_')
    center_time = int(name_split[0]) + (int(name_split[2]) - int(name_split[0])) // 2

    return center_time


def compute_centerized_flow(model, long_path, short_seq_path, exposureTime_UW):
    # compute centerized flow
    short_center_exp = []
    for path in short_seq_path:
        center_time = int(path.split('/')[-1].split('_')[0]) - (exposureTime_UW / 2)
        short_center_exp.append(center_time)

    long_center_time = getLongWCenterTime(long_path)
    short_center_index = np.argmin(np.abs(np.array(short_center_exp) - long_center_time))

    centerized_forward_flow = []
    forward_short_seq_path = short_seq_path[short_center_index:]
    for img1_path, img2_path in zip(forward_short_seq_path[:-1], forward_short_seq_path[1:]):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        with torch.no_grad():
            img1_torch = image2torch(img1)
            img2_torch = image2torch(img2)

            padder = InputPadder(img1_torch.shape)
            img1_torch, img2_torch = padder.pad(img1_torch, img2_torch)

            flow_low, flow_up = model(img1_torch.contiguous(), img2_torch.contiguous(), iters=20, test_mode=True)
            centerized_forward_flow.append(flow_up)

    # meshgrid
    B, C, H, W = centerized_forward_flow[0].size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(img1_torch.device)  # [B, 2, H, W]

    centerized_accum_forward_list = []
    centerized_accum_forward_list.append(grid.clone())
    temp_grid = grid.clone()
    for flow in centerized_forward_flow:
        vgrid = temp_grid.clone()
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        temp_vgrid = vgrid.permute(0, 2, 3, 1)
        vgrid_flow = F.grid_sample(flow, temp_vgrid)

        temp_grid += vgrid_flow
        centerized_accum_forward_list.append(temp_grid.clone())

    centerized_inverse_flow = []
    inverse_short_seq_path = short_seq_path[:short_center_index + 1]
    for img1_path, img2_path in zip(inverse_short_seq_path[:-1], inverse_short_seq_path[1:]):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        with torch.no_grad():
            img1_torch = image2torch(img1)
            img2_torch = image2torch(img2)

            padder = InputPadder(img1_torch.shape)
            img1_torch, img2_torch = padder.pad(img1_torch, img2_torch)

            flow_low, flow_up = model(img2_torch.contiguous(), img1_torch.contiguous(), iters=20, test_mode=True)

            centerized_inverse_flow.append(flow_up)

    # compute accumulated inverse flow
    centerized_accum_inverse_flow_list = []
    # centerized_accum_inverse_flow_list.append(grid.clone())
    temp_grid = grid.clone()
    for flow in centerized_inverse_flow[::-1]:
        vgrid = temp_grid.clone()
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        temp_vgrid = vgrid.permute(0, 2, 3, 1)
        vgrid_flow = F.grid_sample(flow, temp_vgrid)

        temp_grid += vgrid_flow
        centerized_accum_inverse_flow_list.append(temp_grid.clone())
    centerized_accum_inverse_flow_list = centerized_accum_inverse_flow_list[::-1]

    # merge forward and inverse flow
    centerized_accum_flow_list = centerized_accum_inverse_flow_list + centerized_accum_forward_list
    centerized_accum_flow = torch.cat(centerized_accum_flow_list, dim=0)

    return centerized_accum_flow

if __name__ == '__main__':
    if not os.path.exists(flow_out_path):
        os.mkdir(flow_out_path)

    longW_list = glob.glob(os.path.join(longW_root_path, '**/**/**/longW/blur/*_blur.png'))

    for longW_path in tqdm.tqdm(longW_list):
        longW_path_split = longW_path.split('/')

        day_name = longW_path_split[-6]
        video_name = longW_path_split[-5]
        sample_name = longW_path_split[-4]

        out_dir_video = os.path.join(flow_out_path, video_name)

        seqs_list = os.listdir(os.path.join(shortUW_root_path, day_name, video_name, sample_name, 'UWseqs'))
        for seq_name in seqs_list:
            uw_path_list = glob.glob(os.path.join(shortUW_root_path, day_name, video_name, sample_name, 'UWseqs', seq_name, '**.jpg'))
            uw_path_list = sorted(uw_path_list)
            uw_frames_path = np.array(uw_path_list)

            new_flow_path = os.path.join(flow_out_path, day_name, video_name, sample_name, 'UWflows', seq_name)
            os.makedirs(new_flow_path, exist_ok=True)

            save_flows(uw_frames_path, new_flow_path, model, longW_path)
