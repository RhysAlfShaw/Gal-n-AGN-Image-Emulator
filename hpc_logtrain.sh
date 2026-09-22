#!/bin/bash
#SBATCH --job-name=Emulator-train-Diffusion
#SBATCH --output=dr1-emulator/Diffusion-train-%j.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50Gb 
#SBATCH --time=24:00:00         # Hours:Mins:Secs


# set CHAIN_COUNT to the first argument ($1). 
CHAIN_COUNT=${1:-2}

# this will queue a job immediately before this one starts, the job be wait until the current job finishes before starting. 
# Allows long training to be split into multiple jobs and avoids going to the back of the queue.
# for this to work you must regualarly save training checkpoints and be able to resume training from that checkpoint.

if [ $CHAIN_COUNT -gt 0 ]; then
    NEXT_COUNT=$((CHAIN_COUNT - 1))
    echo "Queuing next job... ($NEXT_COUNT subsequent jobs remaining). It will wait in the queue until Job $SLURM_JOB_ID finishes."
    sbatch --dependency=afterany:$SLURM_JOB_ID $0 $NEXT_COUNT
fi

source ~/miniforge3/bin/activate
conda activate euclid_emulator
python dr1-emulator/Emulator/train_diffusers_model.py