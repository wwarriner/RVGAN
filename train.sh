#! /bin/bash
#SBATCH --partition=pascalnodes
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=rvgan-train
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

module load cuda10.1/toolkit/10.1.243
module load Anaconda3
conda activate retina-gan
CUDA_VISIBLE_DEVICES=0

npz="./train_data/stare_mini_crop/image_data.npz"
save="./save/stare_mini"

python -u train.py \
    --npz_file "${npz}" \
    --savedir "${save}" \
    --input_dim 128 \
    --batch_size 4 \
    --epochs 5 \
    --inner_weight 0.5
