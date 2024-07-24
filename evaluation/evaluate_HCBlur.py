import os
import skimage
import numpy as np
from glob import glob
from natsort import natsorted
from skimage import io
import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm
import concurrent.futures
import argparse

parser = argparse.ArgumentParser(description='eval arg')
parser.add_argument('--input_dir', type=str, default='')
parser.add_argument('--out_txt', type=str, default='')
parser.add_argument('--gt_root', type=str, default='./datasets/HCBlur_Syn_test')
parser.add_argument('--core', type=int, default=8)
args = parser.parse_args()



def compute_psnr(image_true, image_test):
    return peak_signal_noise_ratio(image_true, image_test, data_range=1.0)


def compute_ssim(tar_img, prd_img):
    return structural_similarity(tar_img, prd_img, multichannel=True, data_range=1.0)

def proc(filename):
    tar, prd = filename
    tar_img = io.imread(tar)
    prd_img = io.imread(prd)

    if prd_img.shape[2] == 4:
        prd_img = prd_img[:,:,:3]

    tar_img = tar_img.astype(np.float32) / 255.0
    prd_img = prd_img.astype(np.float32) / 255.0

    PSNR = compute_psnr(tar_img, prd_img)
    SSIM = compute_ssim(tar_img, prd_img)
    return (PSNR, SSIM)


if __name__ == '__main__':

    if skimage.__version__ != '0.17.2':
        print("please use skimage==0.17.2 and python3")
        exit()

    input_dir = args.input_dir
    if args.out_txt == '':
        out_txt = input_dir.split('/')[-1] + '.txt'
    else:
        out_txt = args.out_txt
    print(out_txt)

    # find mapping imgname <=> gtpath
    with open('./datalist/HCBlur_Syn_test.txt', 'rt') as f:
        datalist = f.readlines()

    name2gtPath = {}
    for txt_line in datalist:
        txt_split = txt_line.strip().split(' ')
        blur_name = txt_split[1].split('/')[-1]
        gt_path = txt_split[0]
        gt_path = os.path.join(args.gt_root, gt_path)
        name2gtPath[blur_name] = gt_path

    file_path = os.path.join(input_dir)
    path_list = natsorted(glob(os.path.join(file_path, '*blur.png')))
    assert len(path_list) == len(datalist)

    gt_list = []
    for inp_path in path_list:
        inp_name = inp_path.split('/')[-1]
        gt_path = name2gtPath[inp_name]
        gt_list.append(gt_path)

    assert len(path_list) != 0, "Predicted files not found"
    assert len(gt_list) != 0, "Target files not found"

    psnr, ssim, files = [], [], []
    img_files = [(i, j) for i, j in zip(gt_list, path_list)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.core) as executor:
        for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
            psnr.append(PSNR_SSIM[0])
            ssim.append(PSNR_SSIM[1])
            files.append(filename[0])

    sat_psnrs = []
    sat_ssims = []
    non_sat_psnrs = []
    non_sat_ssims = []

    txt_list = []
    for i, values in enumerate(files):
        tar_path = values
        tar_path = '/'.join(tar_path.split('/')[-5:])

        txt = '{:s} {:f} {:f}\n'.format(tar_path, psnr[i], ssim[i])
        txt_list.append(txt)

    avg_psnr = sum(psnr) / len(psnr)
    avg_ssim = sum(ssim) / len(ssim)

    txt = 'For {:s} dataset PSNR: {:f} SSIM: {:f}\n'.format(input_dir, avg_psnr, avg_ssim)
    print(txt)
    txt_list.append(txt)

    with open(out_txt, 'wt') as f:
        f.writelines(txt_list)