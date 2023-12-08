## set MODEL_PATH, num_samples, has_subfolder, images_dir, recons_dir, dire_dir
export CUDA_VISIBLE_DEVICES=3,4,6,7
export NCCL_P2P_DISABLE=1
MODEL_PATH="/kaggle/input/classifier-ckpt/256x256_diffusion_uncond.pt" # "models/lsun_bedroom.pt, models/256x256_diffusion_uncond.pt"

SAMPLE_FLAGS="--batch_size 8 --num_samples 1000  --timestep_respacing ddim20 --use_ddim True"
#SAVE_FLAGS="--images_dir /kaggle/working/dataset/images/test/real --recons_dir /kaggle/working/dataset/recons/test/real --dire_dir /kaggle/working/dataset/dire/test/real"

SAVE_FLAGS="--images_dir /kaggle/working/dataset/images/test/fake --recons_dir /kaggle/working/dataset/recons/test/fake --dire_dir /kaggle/working/dataset/dire/test/fake"


MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"

export CUDA_VISIBLE_DEVICES=0

python /kaggle/working/DetectArtForensics/guided-diffusion/compute_dire.py --model_path $MODEL_PATH $MODEL_FLAGS  $SAVE_FLAGS $SAMPLE_FLAGS --has_subfolder False