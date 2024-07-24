## Deep Hybrid Camera Deblurring for Smartphone Cameras
##### [Project](https://cg.postech.ac.kr/research/hcdeblur/) | [Paper](https://cg.postech.ac.kr/research/hcdeblur/assets/pdf/HCDeblur.pdf) | [Supple](https://cg.postech.ac.kr/research/hcdeblur/assets/HCDeblur_supple.zip)

### Official Implementation of SIGGRAPH Paper 

> Deep Hybrid Camera Deblurring for Smartphone Cameras<br>
> Jaesung Rim<sup>1</sup>, [Junyong Lee<sup>2</sup>](https://junyonglee.me/), Heemin Yang<sup>1</sup>, [Sunghyun Cho<sup>1</sup>](https://www.scho.pe.kr/). <br>
> <sup>1</sup>POSTECH, <sup>2</sup>Samsung AI Center Toronto<br>
> *ACM SIGGRAPH 2024 Conference Papers*<br>

## Installation 

```bash
pip install -r requirements.txt;
python setup.py develop --no_cuda_ext;
pip install git+https://github.com/cheind/pytorch-debayer;
```
## Download

### Dataset [[Google Drive]](https://drive.google.com/drive/folders/1Lk3Gh8_mnpbVuxRz6g0wsVsJ26sP8lXx?usp=sharing) 

<details>
<summary><strong>Descriptions</strong> (click) </summary>

- HCBlur-Syn
  - HCBlur_Syn_train : 5,795 samples for training.
    - We synthesize noise and saturation pixels during training process. 
    - Please refer to [RSBlurPipeline_for_W](https://github.com/rimchang/HCDeblur/blob/6a7bd2fb093a97a5a2f9e2b2c816bc4c28508ea5/basicsr/models/HCFNet_with_RSBlur_model.py#L79) and [RSBlurPipeline_for_UW](https://github.com/rimchang/HCDeblur/blob/6a7bd2fb093a97a5a2f9e2b2c816bc4c28508ea5/basicsr/models/HCFNet_with_RSBlur_model.py#L262)
  - HCBlur_Syn_val : 880 samples for validation.
  - HCBlur_Syn_test : 1,731 samples for evaluation.
- HCBlur-Real
  - 471 pairs of real-world blurred W and U

#### The HCBlur-Syn dataset

```bash
# HCBlur_Syn_train.zip
HCBlur_Syn_test
├── longW # long-exposure wide images
│   ├── 0908/20230908_10_32_05/000001
│   │   ├── longW/blur # folder of blurred image
│   │   ├── longW/gt # folder of gt sharp image
│   ...
├── shortUW # short-exposure ultra-wide images
│   ├── 0908/20230908_10_32_05/000001
│   │   ├── UWseqs/000001 # ultra-wide sequnece corresponding to longW/0908/20230908_10_32_05/000001
│   ...
├── shortUW_depth # estimated depth from FOV alignment step.
│   ├── 0908/20230908_10_32_05_depth.txt # estimated depth values
│   ...
├── shortUW_flows # estimated optical flows from ultra-wide images.
│   ├── 0908/20230908_10_32_05/000001
│   │   ├── UWflows/000001 # estimate optical flows
│   ...
...
```

#### The HCBlur-Real dataset
```bash
# HCBlur_Real.zip
HCBlur_Real
├── longW # long-exposure wide images
│   ├── 1780013444228916_1780013544228916.png 
│   ...
├── shortUW 
│   ├── 1780013444228916_1780013544228916 # ultra-wide sequnece corresponding to 1780013444228916_1780013544228916.png
│   │   ├── 1780013434097457_1780013442430791.jpg 
│   ...
...
```
</details>

### Dataset splits [[link]](./datalist/)

### Pre-trained models [[Google Drive]](https://drive.google.com/drive/folders/1G8ND0oPQ1sA2XQ1sXTR_Esp2ehJy9HGJ?usp=sharing)
<details>
<summary><strong>Descriptions</strong> (click) </summary>

- HC-DNet.pth: Weight of HC-DNet trained on HCBlur.
- HC-FNet.pth: Weight of HC-FNet trained on HCBlur.
- raft-sintel: Weight of RAFT.
- raft-small: Weight of RAFT_small.
</details>

## Demo
```bash
# ./HCDeblur

# demo of two samples of HCBlur-Real
python test_HCBlur_Real.py --dataset_root=demo --out_dir=results_real/demo
```

## Testing

```bash
# ./HCDeblur
# datasets should be located in datasets
# pre-trained weights should be located in pretrained_models

## test on HCBlur-Syn
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4114 basicsr/test.py -opt options/test/HCDNet-test.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4115 basicsr/test.py -opt options/test/HCFNet-test.yml --launcher pytorch

## test on HCBlur-Real
python test_HCBlur_Real.py --dataset_root=datasets/HCBlur-Real --out_dir=results/HCBlur-Real
```

## Evaluation

```bash
# ./HCDeblur

# compute PSNR and SSIM on HCBlur-Syn
python evaluation/evaluate_HCBlur.py --input_dir=results/HCDNet --out_txt=HCDNet.txt
python evaluation/evaluate_HCBlur.py --input_dir=results/HCFNet --out_txt=HCFNet.txt

# compute non-reference metrics on HCBlur-Real
bash evaluation/evaluation_NR_metrics.sh "results/HCBlur-Real/HCDNet/*_HCDNet.png" HCDNet;
bash evaluation/evaluation_NR_metrics.sh "results/HCBlur-Real/HCFNet/*_HCFNet.png" HCFNet;
```

## Training (Soon)

## License

The HCBlur dataset is released under CC BY 4.0 license.

## Acknowledment

The code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR), [NAFNet](https://github.com/megvii-research/NAFNet), [RAFT](https://github.com/princeton-vl/RAFT), [EDVR](https://github.com/xinntao/EDVR) and [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch).

## Citation

If you use our dataset for your research, please cite our paper.

```bibtex
@inproceedings{HCDeblur_rim,
 title={Deep Hybrid Camera Deblurring for Smartphone Cameras},
 author={Rim, Jaesung and Lee, Junyong and Yang, Heemin and Cho, Sunghyun},
 booktitle={ACM SIGGRAPH 2024 Conference Papers},
 year={2024}
}