#!/bin/bash
#SBATCH -J EDD_MultiStreamFocus
#SBATCH --account=gts-mhale30
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH --partition=cpu-medium
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o edd_multistream_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chawkins64@gatech.edu

# Load required modules
module load anaconda3

# Navigate to project directory
cd /storage/home/hcoda1/7/chawkins64/p-mhale30-0

# Run EDD experiments
python run_edd_experiments.py \
    --algorithms "focus_decay,xumei" \
    --nus "0,1000,5000,10000" \
    --Ms "5,10,15,20" \
    --threshold-min 1000 \
    --threshold-max 10000 \
    --threshold-steps 10 \
    --T 1000000 \
    --mu1 1.0 \
    --sims 500 \
    --workers 32 \
    --save

echo "EDD experiments completed!"