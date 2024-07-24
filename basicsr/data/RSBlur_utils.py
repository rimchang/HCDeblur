import torch
import random
import math

def rgb2lin_pt(x):
    gamma = 2.4
    a = 1 / 1.055
    b = 0.055 / 1.055
    c = 1 / 12.92
    d = 0.04045

    in_sign = -2 * torch.lt(x, 0) + 1
    abs_x = torch.abs(x)

    lin_range = torch.lt(abs_x, d)
    gamma_range = torch.logical_not(lin_range)

    lin_value = (c * abs_x)
    gamma_value = torch.exp(gamma * torch.log(a * abs_x + b))

    new_x = (lin_value * lin_range) + (gamma_value * gamma_range)
    new_x = new_x * in_sign

    return new_x


def lin2rgb_pt(x):
    gamma = 1 / 2.4
    a = 1.055
    b = -0.055
    c = 12.92
    d = 0.0031308

    in_sign = -2 * torch.lt(x, 0) + 1
    abs_x = torch.abs(x)

    lin_range = torch.lt(abs_x, d)
    gamma_range = torch.logical_not(lin_range)

    lin_range = lin_range
    gamma_range = gamma_range

    lin_value = (c * abs_x)
    gamma_value = a * torch.exp(gamma * torch.log(abs_x)) + b
    new_x = (lin_value * lin_range) + (gamma_value * gamma_range)
    new_x = new_x * in_sign

    return new_x


def lin2xyz(x, M):

    b, h, w, c = x.shape
    v = x.reshape(b, h * w, c)
    xyz = torch.matmul(v, M)
    xyz = xyz.reshape(b, h, w, c)

    return xyz


def xyz2lin(x, M):
    b, h, w, c = x.shape
    xyz = x.reshape(b, h * w, c)
    lin_rgb = torch.matmul(xyz, M)
    lin_rgb = lin_rgb.reshape(b, h, w, c)

    return lin_rgb


def apply_cmatrix(img, matrix):
    # img : (b, h, w, c)
    # matrix : (3, 3)

    """
    same results below code
    img_reshape = img.reshape(1, h*w, 3)
    out2 = torch.matmul(img_reshape, matrix.permute(0, 2, 1))
    out2 = out2.reshape(1, h, w, 3)
    """

    images = img[:, :, :, None, :]  # (b, h, w, 1, c)
    ccms = matrix[None, None, None, :, :]  # (1, 1, 1, 3, 3)
    out = torch.sum(images * ccms, -1)  # (h, w, 3)

    return out


def mosaic_bayer(image, pattern):
    """Extracts RGGB Bayer planes from an RGB image."""
    shape = image.shape

    if pattern == 'RGGB':
        red = image[:, 0::2, 0::2, 0]  # (b, h/2, w/2)
        green_red = image[:, 0::2, 1::2, 1]
        green_blue = image[:, 1::2, 0::2, 1]
        blue = image[:, 1::2, 1::2, 2]
    elif pattern == 'BGGR':
        red = image[:, 0::2, 0::2, 2]  # (b, h/2, w/2)
        green_red = image[:, 0::2, 1::2, 1]
        green_blue = image[:, 1::2, 0::2, 1]
        blue = image[:, 1::2, 1::2, 0]
    elif pattern == 'GRBG':
        red = image[:, 0::2, 0::2, 1]  # (b, h/2, w/2)
        green_red = image[:, 0::2, 1::2, 0]
        green_blue = image[:, 1::2, 0::2, 2]
        blue = image[:, 1::2, 1::2, 1]
    elif pattern == 'GBRG':
        red = image[:, 0::2, 0::2, 1]  # (b, h/2, w/2)
        green_red = image[:, 0::2, 1::2, 2]
        green_blue = image[:, 1::2, 0::2, 0]
        blue = image[:, 1::2, 1::2, 1]

    image = torch.stack((red, green_red, green_blue, blue), dim=3)  # (b, h/2, w/2, 4)
    image = image.view(-1, shape[1] // 2, shape[2] // 2, 4)

    return image


def add_Poisson_noise_random(img, beta1, beta2):

    random_K_v = beta1.view(-1, 1, 1, 1).to(img.device)

    noisy_img = torch.poisson(img / random_K_v)
    noisy_img = noisy_img * random_K_v

    random_other = beta2.view(-1, 1, 1, 1).to(img.device)
    noisy_img = noisy_img + (torch.normal(torch.zeros_like(noisy_img), std=1) * torch.sqrt(random_other))

    return noisy_img


def WB_img(img, pattern, fr_now, fb_now):
    red_gains = fr_now
    blue_gains = fb_now
    green_gains = torch.ones_like(red_gains)

    if pattern == 'RGGB':
        gains = torch.cat([red_gains, green_gains, green_gains, blue_gains], dim=1)
    elif pattern == 'BGGR':
        gains = torch.cat([blue_gains, green_gains, green_gains, red_gains], dim=1)
    elif pattern == 'GRBG':
        gains = torch.cat([green_gains, red_gains, blue_gains, green_gains], dim=1)
    elif pattern == 'GBRG':
        gains = torch.cat([green_gains, blue_gains, red_gains, green_gains], dim=1)

    gains = gains[:, None, None, :]
    img = img * gains

    return img


def random_noise_W(iso):
    """Generates random noise levels from a log-log linear distribution."""

    iso2shot = lambda x: 1.8768e-06 * x + 9.1875e-05
    shot_noise = iso2shot(iso) + random.gauss(mu=0.0, sigma=4.63727e-04 / 2.0)  # roughly amount of iso 200

    while shot_noise <= 0:
        shot_noise = iso2shot(iso) + random.gauss(mu=0.0, sigma=4.63727e-04 / 2.0)  # roughly amount of iso 200

    log_shot_noise = math.log(shot_noise)
    logshot2logread = lambda x: 1.1137 * x + -5.4715
    log_read_noise = logshot2logread(log_shot_noise)
    read_noise = math.exp(log_read_noise) + random.gauss(mu=0.0,
                                                         sigma=8.29245e-07 / 2.0)  # roughly amount of iso 200

    while read_noise <= 0:
        log_read_noise = logshot2logread(log_shot_noise)
        read_noise = math.exp(log_read_noise) + random.gauss(mu=0.0,
                                                             sigma=8.29245e-07 / 2.0)  # roughly amount of iso 200


    # consider resolution between raw <=> srgb images
    shot_noise = shot_noise * (1 / 1.1439)
    read_noise = read_noise * (1 / 1.1439)

    return shot_noise, read_noise


def random_noise_UW(iso):
    """Generates random noise levels from a log-log linear distribution."""

    iso2shot = lambda x: 3.6760e-06 * x + 5.8747e-04
    shot_noise = iso2shot(iso) + random.gauss(mu=0.0, sigma=1.27653e-03 / 2.0)  # roughly amount of iso 200

    while shot_noise <= 0:
        shot_noise = iso2shot(iso) + random.gauss(mu=0.0, sigma=1.27653e-03 / 2.0)  # roughly amount of iso 200

    log_shot_noise = math.log(shot_noise)
    logshot2logread = lambda x: 1.0753 * x + -5.0671
    log_read_noise = logshot2logread(log_shot_noise)
    read_noise = math.exp(log_read_noise) + random.gauss(mu=0.0,
                                                         sigma=4.7136e-06 / 2.0)  # roughly amount of iso 200

    while read_noise <= 0:
        log_read_noise = logshot2logread(log_shot_noise)
        read_noise = math.exp(log_read_noise) + random.gauss(mu=0.0,
                                                             sigma=4.7136e-06 / 2.0)  # roughly amount of iso 200

    # consider resolution between raw <=> srgb images
    shot_noise = shot_noise * (1 / 3.5373)
    read_noise = read_noise * (1 / 3.5373)

    return shot_noise, read_noise