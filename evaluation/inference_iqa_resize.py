import argparse
import glob
import os
import pyiqa
from pyiqa import create_metric
from tqdm import tqdm
import csv
import torchvision.transforms.functional as TF
from PIL import Image

def main():
    """Inference demo for pyiqa.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=None, help='input image/folder path.')
    parser.add_argument('-r', '--ref', type=str, default=None, help='reference image/folder path if needed.')
    parser.add_argument(
        '--metric_mode',
        type=str,
        default='FR',
        help='metric mode Full Reference or No Reference. options: FR|NR.')
    parser.add_argument('-m', '--metric_name', type=str, default='PSNR', help='IQA metric name, case sensitive.')
    parser.add_argument('--save_file', type=str, default=None, help='path to save results.')

    args = parser.parse_args()

    metric_name = args.metric_name.lower()

    # set up IQA model
    iqa_model = create_metric(metric_name, metric_mode=args.metric_mode)
    metric_mode = iqa_model.metric_mode

    input_paths = sorted(glob.glob(os.path.join(args.input)))
    if args.ref is not None:
        ref_paths = sorted(glob.glob(os.path.join(args.ref)))

    assert len(input_paths) == 471, print(len(input_paths))

    if args.save_file:
        sf = open(args.save_file, 'w')
        sfwriter = csv.writer(sf)

    new_width = 360
    new_height = 640

    avg_score = 0
    test_img_num = len(input_paths)
    if metric_name != 'fid':
        pbar = tqdm(total=test_img_num, unit='image')
        for idx, img_path in enumerate(input_paths):
            img_name = os.path.basename(img_path)
            if metric_mode == 'FR':
                ref_img_path = ref_paths[idx]
            else:
                ref_img_path = None

            img = Image.open(img_path)
            img = img.resize((new_width, new_height), resample=Image.BICUBIC)
            img_tensor = TF.to_tensor(img).unsqueeze(0)

            score = iqa_model(img_tensor, ref_img_path).cpu().item()
            avg_score += score
            pbar.update(1)
            pbar.set_description(f'{metric_name} of {img_name}: {score}')
            pbar.write(f'{metric_name} of {img_name}: {score}')
            if args.save_file:
                sfwriter.writerow([img_name, score])
            
        pbar.close()
        avg_score /= test_img_num
    else:
        assert os.path.isdir(args.input), 'input path must be a folder for FID.'
        avg_score = iqa_model(args.input, args.ref)

    msg = f'Average {metric_name} score of {args.input} with {test_img_num} images is: {avg_score}'
    print(msg)

    if args.save_file:
        sfwriter.writerow(['AVG', avg_score])


    if args.save_file:
        sf.close()

    if args.save_file:
        print(f'Done! Results are in {args.save_file}.')
    else:
        print(f'Done!')


if __name__ == '__main__':
    if pyiqa.__version__ != '0.1.8':
        print("please use pyiqa==0.1.8")
        exit()

    main()
