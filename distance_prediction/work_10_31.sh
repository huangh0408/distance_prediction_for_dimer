#!/bin/sh
#CUDA_VISIBLE_DEVICES="3" python train.protein_8_26.py
#CUDA_VISIBLE_DEVICES="0" python test.protein_8_26.py
CUDA_VISIBLE_DEVICES="3" python train.protein_10_31.py
