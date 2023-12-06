# OLFF: Key Object Localization and Multi-Scale Features Fusion for FineGrained Visual Classification
# requirements
torch == 1.10.0+cu111

# Datasets
Download datasets CUB-200-2011， Stanford Dogs， Oxford-IIIT Pets and Nabirds FGVC datasets in './opt/minist/tiger' folder.

# pretrained weights
We used pretrained weights from the original ViT_B-16

# Training
step1: python3 train_fuse_cub_step1_v10.py
step2: python3 train_fuse_cub_step2_v10.py
