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
singularity exec --nv --env PYTHONPATH=. --bind ${EXP_HOME_CODE_DIR}/vllm/core:${EXP_CONTAINER_CODE_DIR}/vllm/core --bind ${EXP_HOME_CODE_DIR}/vllm/engine:${EXP_CONTAINER_CODE_DIR}/vllm/engine --bind ${EXP_HOME_CODE_DIR}/vllm/entrypoints:${EXP_CONTAINER_CODE_DIR}/vllm/entrypoints --bind ${EXP_HOME_CODE_DIR}/vllm/executor:${EXP_CONTAINER_CODE_DIR}/vllm/executor --bind ${EXP_HOME_CODE_DIR}/vllm/lora:${EXP_CONTAINER_CODE_DIR}/vllm/lora --bind ${EXP_HOME_CODE_DIR}/vllm/model_executor:${EXP_CONTAINER_CODE_DIR}/vllm/model_executor --bind ${EXP_HOME_CODE_DIR}/vllm/worker:${EXP_CONTAINER_CODE_DIR}/vllm/worker --bind ${EXP_HOME_CODE_DIR}/vllm/config.py:${EXP_CONTAINER_CODE_DIR}/vllm/config.py --bind ${EXP_HOME_CODE_DIR}/vllm/sequence.py:${EXP_CONTAINER_CODE_DIR}/vllm/sequence.py --bind ${EXP_HOME_CODE_DIR}/vllm/outputs.py:${EXP_CONTAINER_CODE_DIR}/vllm/outputs.py ${EXP_CONTAINER_IMAGE} ${EXP_BENCHMARK_COMMAND}