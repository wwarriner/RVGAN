#! /bin/bash
#SBATCH --partition=express
#SBATCH --time=5:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=rvgan-preprocess
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

module load Anaconda3
conda activate retina-gan

name="stare_mini"

config="./config.yaml"
in="./train_data/${name}"
out="./train_data/${name}_chunks"
ext=".png"

python -u preprocess.py \
    --config_file "${config}" \
    --input_folder "${in}" \
    --output_folder "${out}" \
    --image_extension "${ext}"