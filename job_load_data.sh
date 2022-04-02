#!/bin/bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --job-name=nlp
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

module load anaconda
source activate kietcdx
python src/load_data.py --data_dir="data" --train_prefix="train" --valid_prefix="valid" --train_frac=0.85 --model_type="gpt2"
