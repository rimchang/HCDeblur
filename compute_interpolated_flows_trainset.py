
import lmdb
from tqdm import tqdm
from os import path as osp
import os
import numpy as np
import glob
import torch
import scipy.io
from basicsr.data.HCBlur_utils import coords_grid, ptsWtoUW, ptsUWtoW, interpolate_grid

dataset_root = 'datasets/HCBlur_Syn_train'
out_path = 'datasets/HCBlur_Syn_train/shortUW_flows'

def coords_grid(b, h, w, homogeneous=False, device=None):
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

def getAllEstimatedDepth(lq_folder):

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

def getGridFromDepth(K_W_pt, K_UW_pt, E_pt, unique_depth_pt, unique_depth):
    depth2grid_dict = {}
    depth2UWdepth_dict = {}
    with torch.no_grad():
        grid = coords_grid(unique_depth_pt.shape[0], W_Height, W_Width, homogeneous=True,
                           device=None).to(K_W_pt.device)  # [B, 3, H, W]
        # calibration matrix based on matlab, so we convert matlab's coordinate (0~h-1) => (1~h)
        pixel_coords, depth_uw = ptsWtoUW(grid, unique_depth_pt, K_W_pt, K_UW_pt, E_pt)

        # normalize to [-1, 1]
        x_grid = 2 * pixel_coords[:, 0] / (UW_Width - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (UW_Height - 1) - 1

        out_grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    for i in range(out_grid.shape[1]):
        depth2grid_dict[unique_depth[i]] = out_grid[:, i, :, :].view(1, W_Height, W_Width,
                                                                     2).clone()  # [B, H, W, 2]
        depth2UWdepth_dict[unique_depth[i]] = depth_uw[:, i]

    return depth2grid_dict, depth2UWdepth_dict

# load stereoParams
K_W = scipy.io.loadmat('mat_collections/K.mat')['K'].astype('float32').transpose(1, 0)
K_UW = scipy.io.loadmat('mat_collections/K2.mat')['K2'].astype('float32').transpose(1, 0)
E = scipy.io.loadmat('mat_collections/E2.mat')['E2'].astype('float32').transpose(1, 0)

K_W_pt = torch.from_numpy(K_W).unsqueeze(0)
K_UW_pt = torch.from_numpy(K_UW).unsqueeze(0)
E_pt = torch.from_numpy(E).unsqueeze(0)

depth_dict, unique_depth_list = getAllEstimatedDepth(dataset_root)

UW_Height, UW_Width = 1280, 720
W_Height, W_Width = 3840, 2160

unique_depth_np = np.array([float(depth) for depth in unique_depth_list]).astype('float32')
unique_depth_pt = torch.from_numpy(unique_depth_np.copy())
unique_depth_pt = unique_depth_pt.view(1, unique_depth_pt.shape[0], 1, 1).repeat(1, 1, W_Height, W_Width)  # [B, D, H, W]

depth2grid, depth2UWdepth_dict = getGridFromDepth(K_W_pt, K_UW_pt, E_pt, unique_depth_pt, unique_depth_list)

def compute_interpolated_flows(short_flow_path, lq_path):
    short_flow = np.load(short_flow_path)
    short_flow_pt = torch.from_numpy(short_flow)

    short_center_exp = []
    short_seqs_path = glob.glob(short_flow_path.replace('shortUW_flows', 'shortUW').replace('UWflows', 'UWseqs').replace(
        'centerized_accum_flows.npy', '*.jpg'))
    short_seqs_path = sorted(short_seqs_path)

    for path in short_seqs_path:
        name_split = path.split('/')[-1][:-4].split('_')
        center_time = int(name_split[0]) + (int(name_split[1]) - int(name_split[0])) // 2
        short_center_exp.append(center_time)

    with torch.no_grad():
        long_min_exp = int(lq_path.split('/')[-1].split('_')[0])
        long_max_exp = int(lq_path.split('/')[-1].split('_')[2])

        cropped_interpolated_flow_pt = interpolate_grid(short_flow_pt, long_min_exp, long_max_exp, short_center_exp)

        ones = torch.ones_like(cropped_interpolated_flow_pt[:, 0:1, :, :])
        homogenous_accum_flows = torch.cat([cropped_interpolated_flow_pt, ones], dim=1)

        flow_dim, flow_c, flow_h, flow_w = homogenous_accum_flows.shape

        flow_path_split = short_flow_path.split('/')
        key = '%s_%s_%s' % (flow_path_split[-5], flow_path_split[-4], flow_path_split[-2])

        W_depth = depth_dict[key]
        UW_depth = depth2UWdepth_dict[W_depth]
        depth_estimated_grid = ones * UW_depth

        flowsOnW = ptsUWtoW(homogenous_accum_flows, depth_estimated_grid, K_W_pt.repeat(flow_dim, 1, 1),
                            K_UW_pt.repeat(flow_dim, 1, 1), E_pt.repeat(flow_dim, 1, 1))
        flowsOnW = flowsOnW.view(flow_dim, 2, flow_h, flow_w)
        flowsOnW = flowsOnW.cpu().numpy().astype('float16')

    return flowsOnW

if __name__ == '__main__':


    datalist_path = 'datalist/HCBlur_Syn_train.txt'
    with open(datalist_path, 'rt') as f:
        datalist = f.readlines()

    all_npy_list = []
    for datalist_txt in datalist:
        gt_path, lq_path = datalist_txt.strip().split(' ')

        flow_dir = '/'.join(lq_path.split('/')[:-3]).replace('longW/', 'shortUW_flows/') + '/UWflows/'
        seqs_list = os.listdir(os.path.join(dataset_root, flow_dir))
        for seqs_name in seqs_list:
            npy_path = os.path.join(dataset_root, flow_dir, seqs_name, 'centerized_accum_flows.npy')
            all_npy_list.append([lq_path, npy_path])

    for idx, (lq_path, flow_path) in tqdm(enumerate(all_npy_list)):

        flowsOnW = compute_interpolated_flows(flow_path, lq_path)
        new_flow_path = flow_path.replace(dataset_root+'/shortUW_flows', out_path).replace('centerized_accum_flows.npy', 'interpolated_centerized_accum_flow.npy')
        os.makedirs(os.path.dirname(new_flow_path), exist_ok=True)

        np.save(new_flow_path, flowsOnW)
