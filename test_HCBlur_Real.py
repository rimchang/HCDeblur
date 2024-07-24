import torch
import glob
import cv2
import os
import tqdm
import numpy as np
import scipy.io
import torch.nn.functional as F
from copy import deepcopy
import argparse

from basicsr.models.archs.core.raft import RAFT
from basicsr.data.HCBlur_utils import ptsWtoUW, ptsUWtoW, interpolate_grid, coords_grid, rgb2lin_np, lin2rgb_np
from basicsr.models.archs.HCDNet_arch import HCDNet_Local
from basicsr.models.archs.HCFNet_arch import HCFNet_Local



parser = argparse.ArgumentParser()
parser.add_argument('--DNet_weight_path', default='pretrained_models/HC-DNet.pth')
parser.add_argument('--FNet_weight_path', default='pretrained_models/HC-FNet.pth')
parser.add_argument('--dataset_root', default='/Jsrim_mango/release_dataset/HCBlur_Real')
parser.add_argument('--out_dir', default='results_real/HCDeblur_0705')
parser.add_argument('--viz_dir', default='results_real/HCDeblur_0705_viz')
parser.add_argument('--raft_small', action='store_true', help='use small version of RAFT')
parser.add_argument('--viz', action='store_true', help='store visualization results')
args = parser.parse_args()

DNet_weight_path = args.DNet_weight_path
FNet_weight_path = args.FNet_weight_path
root_dir = args.dataset_root
out_dir = args.out_dir
viz_dir = args.viz_dir
viz = args.viz
RAFT_small = args.raft_small

normalize_flows = True
reference_first = True

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

######################## load RAFT ########################

if RAFT_small:
    model = RAFT(small=True)
    load_network(model, './pretrained_models/raft-small.pth', param_key=None)
else:
    model = RAFT(small=False)
    load_network(model, './pretrained_models/raft-sintel.pth', param_key=None)

model.to(DEVICE)
model.eval()


######################## load HCDNet ########################

D_net = HCDNet_Local(3, 16, 1, [1,1,1,1], [1,1,1,1])
load_network(D_net, DNet_weight_path)
D_net.to(DEVICE)
D_net.eval()


######################## load HCFNet ########################

F_net = HCFNet_Local(3, 16, 64, middle_blk_num=10, middle_seqs_bkl_num=10, train_size=(1, 3, 384, 384))
load_network(F_net, FNet_weight_path)
F_net.to(DEVICE)
F_net.eval()


# utils for RAFT
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


def image2torch(img):
    img = torch.from_numpy(img*255).permute(2, 0, 1).float()
    return img[np.newaxis].to(DEVICE)


def compute_centerized_flow(model, uw_frames, short_center_index):

    crop_H = (1280 - 840)//2
    crop_W = (720 - 560)//2

    # compute forward flows from center to end of uw frames
    centerized_forward_flow = []
    forward_short_seq_path = uw_frames[short_center_index:]
    for img1, img2 in zip(forward_short_seq_path[:-1], forward_short_seq_path[1:]):

        with torch.no_grad():
            img1_torch = image2torch(img1)
            img2_torch = image2torch(img2)

            # we only compute flows on cropped images for reduing computational costs
            img1_torch = img1_torch[:,:,crop_H:-crop_H, crop_W:-crop_W]
            img2_torch = img2_torch[:,:,crop_H:-crop_H, crop_W:-crop_W]

            padder = InputPadder(img1_torch.shape)
            img1_torch, img2_torch = padder.pad(img1_torch, img2_torch)

            flow_low, flow_up = model(img1_torch.contiguous(), img2_torch.contiguous(), iters=20, test_mode=True)

            padding = [crop_W, crop_W, crop_H, crop_H]
            flow_up = F.pad(flow_up, padding)

            centerized_forward_flow.append(flow_up)

    # meshgrid
    B, C, H, W = centerized_forward_flow[0].size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(img1_torch.device)  # [B, 2, H, W]

    # accumulate the forward flows
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

    # compute inverse flows from end to center
    centerized_inverse_flow = []
    inverse_short_seq_path = uw_frames[:short_center_index + 1]
    for img1, img2 in zip(inverse_short_seq_path[:-1], inverse_short_seq_path[1:]):

        with torch.no_grad():
            img1_torch = image2torch(img1)
            img2_torch = image2torch(img2)

            # we only compute flows on cropped images for reduing computational costs
            img1_torch = img1_torch[:,:,crop_H:-crop_H, crop_W:-crop_W]
            img2_torch = img2_torch[:,:,crop_H:-crop_H, crop_W:-crop_W]

            padder = InputPadder(img1_torch.shape)
            img1_torch, img2_torch = padder.pad(img1_torch, img2_torch)

            flow_low, flow_up = model(img2_torch.contiguous(), img1_torch.contiguous(), iters=20, test_mode=True)

            padding = [crop_W, crop_W, crop_H, crop_H]
            flow_up = F.pad(flow_up, padding)

            centerized_inverse_flow.append(flow_up)

    # accumulate the inverse flows
    centerized_accum_inverse_flow_list = []
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

def viz_blur_kernels(img, kernels, output_path):
    image = img.copy()

    B, C, H, W = kernels.size()

    xx = torch.arange(0, W, 100).view(1, -1)
    yy = torch.arange(0, H, 100).view(-1, 1)
    xx = xx.repeat(yy.shape[0], 1)
    yy = yy.repeat(1, xx.shape[1])

    for index in range(kernels.shape[0] - 1):
        index_grid = kernels[index, :, yy, xx]

        index_xx = index_grid[0, :, :].numpy().astype('int32')
        index_yy = index_grid[1, :, :].numpy().astype('int32')

        next_grid = kernels[index + 1, :, yy, xx]

        next_xx = next_grid[0, :, :].numpy().astype('int32')
        next_yy = next_grid[1, :, :].numpy().astype('int32')

        # import pdb; pdb.set_trace()

        for (x, y, x_next, y_next) in zip(index_xx.flatten(), index_yy.flatten(), next_xx.flatten(), next_yy.flatten()):
            cv2.line(image, (x, y), (x_next, y_next), (0, 255, 0), thickness=3)

    cv2.imwrite(output_path, image)


# load stereoParams
K_W = scipy.io.loadmat('CameraCalibration/K.mat')['K'].astype('float32').transpose(1, 0)
K_UW = scipy.io.loadmat('CameraCalibration/K2.mat')['K2'].astype('float32').transpose(1, 0)
E = scipy.io.loadmat('CameraCalibration/E2.mat')['E2'].astype('float32').transpose(1, 0)

K_W_pt = torch.from_numpy(K_W).unsqueeze(0).cuda()
K_UW_pt = torch.from_numpy(K_UW).unsqueeze(0).cuda()
E_pt = torch.from_numpy(E).unsqueeze(0).cuda()

# compute depth_candidates
B = 14.9425
f = 2743.6
disparity_candidates = np.arange(0, 70, 5) + 0.00001
depth_candidates = np.zeros_like(disparity_candidates)

UW_Height, UW_Width = 1280, 720
W_Height, W_Width = 3840, 2160

for i in range(disparity_candidates.shape[0]):
    depth_candidates[i] = B * f / disparity_candidates[i]

depth = torch.from_numpy(depth_candidates.astype('float32'))
depth = depth.view(1, depth_candidates.shape[0], 1, 1).repeat(1, 1, W_Height, W_Width).cuda()  # [B, D, H, W]

# pre-computed the warping grids according to the depth values
with torch.no_grad():
    grid = coords_grid(1, W_Height, W_Width, homogeneous=True, device=depth.device).cuda()  # [B, 3, H, W]
    pixel_coords, depth_uw = ptsWtoUW(grid, depth, K_W_pt, K_UW_pt, E_pt)

    # normalize to [-1, 1]
    x_grid = 2 * pixel_coords[:, 0] / (UW_Width - 1) - 1
    y_grid = 2 * pixel_coords[:, 1] / (UW_Height - 1) - 1

    out_grid = torch.stack([x_grid, y_grid], dim=-1).cuda()  # [B, D, H*W, 2]

W_ori_grid = coords_grid(1, W_Height, W_Width, homogeneous=False).cuda()  # [B, 2, H, W]
W_ori_grid_resize = F.interpolate(W_ori_grid, scale_factor=1 / 6, mode='bilinear',
                                  align_corners=True, antialias=False)

if __name__ == '__main__':

    W_path_list = glob.glob(os.path.join(root_dir, 'longW/*.png'))
    if len(W_path_list) != 471:
        assert "wrong total number of longW frames"

    for W_path in tqdm.tqdm(W_path_list):

        # read longW frame
        W_img = cv2.imread(W_path)
        W_img = cv2.cvtColor(W_img, cv2.COLOR_BGR2RGB).astype('float32') / 255

        W_name = W_path.split('/')[-1][:-4]
        W_name_split = W_name.split('_')

        # read corresponding UW frames
        UW_frames_paths = glob.glob(os.path.join(root_dir, 'shortUW', W_name,'*.jpg'))
        UW_frames_paths = sorted(UW_frames_paths)
        UW_frames_list = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype('float32') / 255.0 for img_path in
                     UW_frames_paths]

        # extract timestamps from name of images
        start_timestamp_W = float(W_name_split[0])
        end_timestamp_W = float(W_name_split[1])

        start_timestamp_UW = []
        end_timestamp_UW = []

        for uw_frame_path in UW_frames_paths:
            UW_name = uw_frame_path.split('/')[-1][:-4]
            UW_name_split = UW_name.split('_')

            start_timestamp_UW.append(float(UW_name_split[0]))
            end_timestamp_UW.append(float(UW_name_split[1]))

        start_timestamp_UW = np.array(start_timestamp_UW)
        end_timestamp_UW = np.array(end_timestamp_UW)

        # averaging uw frames for FOV-alignment
        lin_img_list = [rgb2lin_np(img) for img in UW_frames_list[1:-1]]
        avgUW_img = lin2rgb_np(np.mean(lin_img_list, axis=0)).astype('float32')

        # numpy2torch
        avgUW_img_pt = torch.from_numpy(avgUW_img).permute(2, 0, 1).unsqueeze(0).cuda()
        longW_img_pt = torch.from_numpy(W_img).permute(2, 0, 1).unsqueeze(0).cuda()

        ################# FOV alignment #################
        with torch.no_grad():

            b, d, h, w = depth.size()
            # warping avgUW according to the pre-defined depths
            warped_feature = F.grid_sample(avgUW_img_pt, out_grid.view(b, d * h, w, 2), mode='bilinear',
                                           padding_mode='zeros',
                                           align_corners=True).view(b, 3, d, h, w)  # [B, C, D, H, W]

            sqerr = (warped_feature - longW_img_pt.unsqueeze(2)) ** 2
            err_plane = sqerr.permute(0, 1, 3, 4, 2).reshape(b, -1, d).mean(1)

            # find best depth index
            depth_index = err_plane.argmin()

            if viz:
                # warp center frames for visualization
                long_center_time = start_timestamp_W + (end_timestamp_W - start_timestamp_W) / 2
                short_center_time = start_timestamp_UW + (end_timestamp_UW - start_timestamp_UW) / 2
                short_center_index = np.argmin(np.abs(np.array(short_center_time) - long_center_time))
                centerUW_img_pt = torch.from_numpy(UW_frames_list[short_center_index]).permute(2, 0, 1).unsqueeze(0).cuda()

                target_grid = out_grid[:, depth_index, :, :]
                warped_centerUW = F.grid_sample(centerUW_img_pt, target_grid.view(1, h, w, 2),
                                                 mode='bicubic',
                                                 padding_mode='zeros',
                                                 align_corners=True)  # [B, C, D, H, W]

                new_dir = os.path.join(viz_dir)
                os.makedirs(new_dir, exist_ok=True)
                cv2.imwrite('%s/%s_warpedUW.png' % (new_dir, W_name),
                            warped_centerUW[0].cpu().detach().permute(1, 2, 0).numpy()[:, :, [2, 1, 0]] * 255)

        ################# compute optical flows #################
        with torch.no_grad():
            long_center_time = start_timestamp_W + (end_timestamp_W - start_timestamp_W) / 2
            short_center_time = start_timestamp_UW + (end_timestamp_UW - start_timestamp_UW) / 2
            short_center_index = np.argmin(np.abs(np.array(short_center_time) - long_center_time))
            centerized_accum_flow = compute_centerized_flow(model, UW_frames_list, short_center_index)
            interpolated_accum_flows = interpolate_grid(centerized_accum_flow, start_timestamp_W, end_timestamp_W,
                                                        short_center_time).float()

        ################# compute blur kernels #################
        with torch.no_grad():
            ones = torch.ones_like(interpolated_accum_flows[:, 0:1, :, :])
            homogenous_accum_flows = torch.cat([interpolated_accum_flows, ones], dim=1)

            flow_dim, flow_c, flow_h, flow_w = homogenous_accum_flows.shape

            depth_estimated = depth_uw[0, depth_index]
            depth_estimated_repeat = ones * depth_estimated

            # convert coordinates of uw frames to w frame
            flowsOnW = ptsUWtoW(homogenous_accum_flows, depth_estimated_repeat, K_W_pt.repeat(flow_dim, 1, 1),
                                K_UW_pt.repeat(flow_dim, 1, 1), E_pt.repeat(flow_dim, 1, 1))
            flowsOnW = flowsOnW.view(flow_dim, 2, flow_h, flow_w)

            # warp according to the estimated depth
            target_grid = out_grid[:, depth_index, :, :]
            warped_flowsOnW = F.grid_sample(flowsOnW, target_grid.repeat(flow_dim, 1, 1).view(flow_dim, h, w, 2),
                                            mode='bilinear',
                                            padding_mode='zeros',
                                            align_corners=True).view(flow_dim, 2, h, w)  # [B, C, D, H, W]

            blur_kernels = warped_flowsOnW - W_ori_grid

            if normalize_flows:
                blur_kernels = blur_kernels - blur_kernels[4:5, :, :, :]

        if viz:
            new_dir = os.path.join(viz_dir)
            os.makedirs(new_dir, exist_ok=True)
            viz_blur_kernels((W_img[:,:,[2,1,0]] * 255 + 0.5).astype('uint8'), warped_flowsOnW.cpu(),
                            '%s/%s_warpedFlows.png' % (new_dir, W_name))

        ################# HC-DNet #################
        with torch.no_grad():
            output_dnet = D_net(longW_img_pt.cuda(), blur_kernels.reshape(1, -1, h, w).cuda())
            new_dir = os.path.join(out_dir, 'HCDNet')
            os.makedirs(new_dir, exist_ok=True)
            cv2.imwrite('%s/%s_HCDNet.png' % (new_dir, W_name),
                        output_dnet[0].cpu().detach().permute(1, 2, 0).numpy()[:, :, [2, 1, 0]] * 255)

        ################# FOV-align optical flows to lower resolutions (x1/6) #################
        with torch.no_grad():
            ones = torch.ones_like(centerized_accum_flow[:, 0:1, :, :])
            homogenous_accum_flows = torch.cat([centerized_accum_flow, ones], dim=1)

            flow_dim, flow_c, flow_h, flow_w = homogenous_accum_flows.shape

            # convert coordinates of uw frames to w frame
            depth_estimated_repeat = ones * depth_estimated
            seqs_flow = ptsUWtoW(homogenous_accum_flows, depth_estimated_repeat, K_W_pt.repeat(flow_dim, 1, 1),
                                 K_UW_pt.repeat(flow_dim, 1, 1), E_pt.repeat(flow_dim, 1, 1))
            seqs_flow = seqs_flow.view(flow_dim, 2, flow_h, flow_w)

            # resize the warping grid for HC-FNet
            target_grid = out_grid[:, depth_index, :, :].view(1, h, w, 2)
            target_grid = F.interpolate(target_grid.permute(0, 3, 1, 2), scale_factor=1 / 6,
                                        mode='bilinear',
                                        align_corners=True, antialias=False).permute(0, 2, 3, 1)

            _, h, w, c = target_grid.shape
            flow_dim, _, _, _ = seqs_flow.shape

            # warp flows according to the resized warping grid
            warped_seqs_flow = F.grid_sample(seqs_flow, target_grid.repeat(flow_dim, 1, 1, 1).view(flow_dim, h, w, 2),
                                             mode='bilinear',
                                             padding_mode='zeros',
                                             align_corners=True).view(flow_dim, 2, h, w)  # [B, C, D, H, W]

            seqs_flow = warped_seqs_flow.clone()
            seqs_flow = (seqs_flow - W_ori_grid_resize) * 1 / 6

            if reference_first:
                seqs_flow_list = list(torch.split(seqs_flow, 1, dim=0))
                seqs_flow_list = [temp.squeeze(0) for temp in seqs_flow_list]

                reference_flow = seqs_flow_list.pop(short_center_index)
                seqs_flow_list = [reference_flow] + seqs_flow_list

                seqs_flow = torch.stack(seqs_flow_list, dim=1)  # (2, T, H, W)

        ################# FOV-align UW frames to lower resolutions (x1/6) #################
        with torch.no_grad():
            uw_seqs = np.array(UW_frames_list)
            uw_seqs_pt = torch.from_numpy(uw_seqs).permute(0, 3, 1, 2).cuda()
            seqs, _, _, _ = uw_seqs_pt.shape

            # warp uw frames according to the resized warping grid
            warped_uw_seqs_pt = F.grid_sample(uw_seqs_pt, target_grid.view(1, h, w, 2).repeat(seqs, 1, 1, 1),
                                                 mode='bicubic',
                                                 padding_mode='zeros',
                                                 align_corners=True)  # [B, C, D, H, W]

            warped_uw_seqs = warped_uw_seqs_pt
            if reference_first:
                warped_uw_seqs_list = np.split(warped_uw_seqs, seqs)
                reference_short = warped_uw_seqs_list.pop(short_center_index)
                warped_uw_seqs_list = [reference_short] + warped_uw_seqs_list

                uw_seqs = torch.stack(warped_uw_seqs_list, dim=1)  # (C, T, H, W)

        if viz:
            for i in range(uw_seqs.shape[1]):
                short_img = uw_seqs[:,i,:,:]

                new_dir = os.path.join(viz_dir, W_name)
                os.makedirs(new_dir, exist_ok=True)
                cv2.imwrite('%s/UW_%03d.png' % (new_dir, i),
                            short_img[0].cpu().detach().permute(1, 2, 0).numpy()[:, :, [2, 1, 0]] * 255)

        ################# HC-FNet #################
        with torch.no_grad():
            inp = longW_img_pt
            deblur = output_dnet
            seqs = uw_seqs.permute(0, 2, 1, 3, 4)
            seqs_mask = torch.ones_like(seqs)[:, 0:1, :, :, :]
            seqs_flow = seqs_flow.unsqueeze(0)


            output_fnet = F_net(inp, deblur, seqs, seqs_mask, seqs_flow)


            new_dir = os.path.join(out_dir, 'HCFNet')
            os.makedirs(new_dir, exist_ok=True)
            cv2.imwrite('%s/%s_HCFNet.png' % (new_dir, W_name),
                        output_fnet[0].cpu().detach().permute(1, 2, 0).numpy()[:, :, [2, 1, 0]] * 255)


