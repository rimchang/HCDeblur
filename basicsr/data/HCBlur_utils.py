import torch
import numpy as np


def rgb2lin_np(x):
    # based matlab rgb2lin
    gamma = 2.4
    a = 1 / 1.055
    b = 0.055 / 1.055
    c = 1 / 12.92
    d = 0.04045

    in_sign = -2 * (x < 0) + 1
    abs_x = np.abs(x)

    lin_range = (abs_x < d)
    gamma_range = np.logical_not(lin_range)

    out = x.copy()
    out[lin_range] = c * abs_x[lin_range]
    out[gamma_range] = np.exp(gamma * np.log(a * abs_x[gamma_range] + b))

    out = out * in_sign

    return out


def lin2rgb_np(x):
    # based matlab lin2rgb
    gamma = 1 / 2.4
    a = 1.055
    b = -0.055
    c = 12.92
    d = 0.0031308

    in_sign = -2 * (x < 0) + 1
    abs_x = np.abs(x)

    lin_range = (abs_x < d)
    gamma_range = np.logical_not(lin_range)

    out = x.copy()
    out[lin_range] = c * abs_x[lin_range]
    out[gamma_range] = a * np.exp(gamma * np.log(abs_x[gamma_range])) + b

    out = out * in_sign

    return out

def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.
frames_list
    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[:, 1:] - fp[:, :-1]) / (xp[:, 1:] - xp[:, :-1])  # slope
    b = fp[:, :-1] - (m.mul(xp[:, :-1]))

    indicies = torch.sum(torch.ge(x[:, :, None], xp[:, None, :]), -1) - 1  # torch.ge:  x[i] >= xp[i] ? true: false
    indicies = torch.clamp(indicies, 0, m.shape[-1] - 1)

    line_idx = torch.arange(len(indicies), device=indicies.device).view(-1, 1)
    line_idx = line_idx.expand(indicies.shape)

    return m[line_idx, indicies].mul(x) + b[line_idx, indicies]


def interpolate_grid(accum_grid, long_min_exp, long_max_exp, short_center_times, bins=9):
    # compute interpolated grid
    seq_relative_exp = [(short_center_time - long_min_exp) / (long_max_exp - long_min_exp) for short_center_time in
                        short_center_times]

    exp_len, c, h, w = accum_grid.shape

    # (exp_len, h*w) => (h*w, exp_len)
    accum_grid_x = accum_grid[:, 0, :, :].view(exp_len, -1).permute(1, 0)
    accum_grid_y = accum_grid[:, 1, :, :].view(exp_len, -1).permute(1, 0)

    sc = torch.linspace(0, 1, bins).expand(h * w, -1).to(accum_grid_y.device)  # (h*w, exp_len)
    x = torch.tensor(seq_relative_exp).expand(h * w, -1).to(accum_grid_y.device)  # (h*w, exp_len)

    #print(sc.shape, x.shape, accum_grid_x.shape, accum_grid_y.shape)

    y = accum_grid_x
    # (h*w, exp_len) => (exp_len, h*w) => (exp_len, 1, h, w)
    interpolated_x = interpolate(sc, x, y).permute(1, 0).view(-1, 1, h, w)

    y = accum_grid_y
    # (h*w, exp_len) => (exp_len, h*w) => (exp_len, 1, h, w)
    interpolated_y = interpolate(sc, x, y).permute(1, 0).view(-1, 1, h, w)
    interpolated_accum_grid = torch.cat([interpolated_x, interpolated_y], dim=1)

    return interpolated_accum_grid

def ptsWtoUW(grid, depth, K_W, K_UW, pose):
    b, c, h2, w2 = grid.size()
    _, d, _, _ = depth.size()

    grid = grid.clone()
    # convert to the matlab coordinates as our calibration matrix computed from the matlab
    grid[:,:2,:,:] = grid[:,:2,:,:] + 1

    points = torch.inverse(K_W).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
    points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
        1, 1, d, 1) * depth.view(b, 1, d, h2 * w2)  # [B, 3, D, H*W]
    points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
    points = torch.bmm(K_UW, points.view(b, 3, -1)).view(b, 3, d, h2 * w2)  # [B, 3, D, H*W]

    pixel_coords = points[:, :2] / (points[:, -1:] + 0.00001)

    # we save the mean depth values for warping uw frames
    depth_uw = points[:,-1,:,:].mean(-1)

    # convert to the python coordinates
    pixel_coords[:,:2] = pixel_coords[:,:2] -1

    return pixel_coords, depth_uw


def ptsUWtoW(grid, depth, K_W, K_UW, pose):
    b, c, h2, w2 = grid.size()
    _, d, _, _ = depth.size()

    grid = grid.clone()
    # convert to the matlab coordinates as our calibration matrix computed from the matlab
    grid[:,:2,:,:] = grid[:,:2,:,:] + 1

    points = torch.inverse(K_UW).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
    points = torch.bmm(torch.inverse(pose[:, :3, :3]), points).unsqueeze(2).repeat(
        1, 1, d, 1) * depth.view(b, 1, d, h2 * w2)  # [B, 3, D, H*W]
    points = points + (-1 * pose[:, :3, -1:]).unsqueeze(-1)  # [B, 3, D, H*W]
    # reproject to 2D image plane
    points = torch.bmm(K_W, points.view(b, 3, -1)).view(b, 3, d, h2 * w2)  # [B, 3, D, H*W]
    pixel_coords = points[:, :2] / (points[:, -1:] + 0.00001)

    # convert to the python coordinates
    pixel_coords[:,:2] = pixel_coords[:,:2] -1

    return pixel_coords


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


