#!/bin/bash
#SBATCH --partition=SCSEGPU_MSAI
#SBATCH --qos=q_msai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --job-name=nlp
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

module load anaconda
source activate kietcdx
python src/evaluate.py \
    --seed=0 \
    --data_dir="data" \
    --model_type="gpt2" \
    --bos_token="<bos>" \
    --sp1_token="<sp1>" \
    --sp2_token="<sp2>" \
    --gpu="0" \
    --max_len=1024 \
    --top_p=0.8 \
    --ckpt_dir="saved_models" \
    --ckpt_name="saved_models/gpt2/best_ckpt_epoch=19_valid_loss=2.3336_question_loss.ckpt" \