#!/bin/bash
### make sure that you have modified the EXP_NAME, DATASETS, DATASETS_TEST
eval "$(conda shell.bash hook)"
#conda activate dire
EXP_NAME="gray_101"
DATASETS="dire"
DATASETS_TEST="dire"
python train.py --gpus 0 --exp_name $EXP_NAME datasets $DATASETS datasets_test $DATASETS_TEST
