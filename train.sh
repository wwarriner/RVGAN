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

name="stare_mini"

config="./config.yaml"
npz="./train_data/${name}_chunks/image_data.npz"
save="./out/${name}"

python -u train.py \
    --config_file "${config}" \
    --npz_file "${npz}" \
    --save_folder "${save}"
