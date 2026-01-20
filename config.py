import os

# Dataset root path
'''
    define your root directory
'''
# config.py
ROOT = "/kaggle/input/parkbohee/AnimalCLEF2025-main/dataset"  # 실제 dataset 폴더 위치에 맞춰 수정
OUT_DIR = "/kaggle/working/out"
DEVICE = "cuda"

# Model settings
MEGAD_NAME = 'hf-hub:BVRA/MegaDescriptor-L-384'
EVA_NAME = 'EVA02-L-14-336'
EVA_WEIGHT_NAME = 'merged2b_s6b_b61k'
DEVICE = 'cuda'

# Threshold
THRESHOLD = 0.35
