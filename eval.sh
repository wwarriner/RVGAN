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

config="./config.yaml"
save="./save/stare"
input="./train_data/stare"
output="./eval/stare"

python -u eval.py \
    --config_file="${config}" \
    --model_folder="${save}" \
    --input_folder="${input}" \
    --output_folder="${output}"
