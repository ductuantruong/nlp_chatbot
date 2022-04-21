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
cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888