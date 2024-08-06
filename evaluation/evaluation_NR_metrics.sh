#!/bin/bash

RESULTS_PATH=$1
PREFIX_NAME=$2

python evaluation/inference_iqa_resize.py -i="${RESULTS_PATH}" -m=niqe --save_file=${PREFIX_NAME}_niqe.txt;
python evaluation/inference_iqa_resize.py -i="${RESULTS_PATH}" -m=brisque --save_file=${PREFIX_NAME}_brisque.txt;
python evaluation/inference_iqa_resize.py -i="${RESULTS_PATH}" -m=topiq_nr-spaq --save_file=${PREFIX_NAME}_topiq_spaq.txt;
