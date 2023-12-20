#!/bin/bash
### make sure that you have modified the EXP_NAME, CKPT, DATASETS_TEST
#eval "$(conda shell.bash hook)"
#conda activate dire

EXP_NAME="model_adm_imagenet"
#CKPT="/kaggle/input/classifier-ckpt/lsun_iddpm.pth"
CKPT="/kaggle/working/DetectArtForensics/dataset/exp/lsun_adm/ckpt/model_epoch_latest.pth"
DATASETS_TEST="dire"
python test.py --gpus 0 --ckpt $CKPT --exp_name $EXP_NAME datasets_test $DATASETS_TEST