#! /bin/bash
#SBATCH --partition=pascalnodes
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=rvgan-eval
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

module load cuda10.1/toolkit/10.1.243
module load Anaconda3
conda activate retina-gan

name="stare"

config="./config.yaml"
model="./out/${name}"
input="./train_data/${name}"
output="./eval/${name}"

python -u eval.py \
    --config_file="${config}" \
    --model_folder="${model}" \
    --input_folder="${input}" \
    --output_folder="${output}" \
    --fov_scale_factor=0.452073938 \
    --image_extension=".TIF" \
    --mask_extension=".TIF"
