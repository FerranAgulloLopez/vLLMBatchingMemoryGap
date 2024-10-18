#!/bin/bash
#SBATCH --job-name=${EXP_NAME}
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --output=${EXP_RESULTS_PATH}/log_%j.out
#SBATCH --error=${EXP_RESULTS_PATH}/log_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --gres gpu:1
#SBATCH --time=${EXP_MAX_DURATION_SECONDS}
module load singularity
singularity exec --nv --env PYTHONPATH=. vllm-benchmark-default.sif python3 ${EXP_BENCHMARK_EXECUTABLE} ${EXP_ARGS}