#!/bin/bash
#SBATCH --job-name=_250_230_2
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --output=benchmarks_simple/definitive_results/initial/llama-2-7b/_250_230_2/log_%j.out
#SBATCH --error=benchmarks_simple/definitive_results/initial/llama-2-7b/_250_230_2/log_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --gres gpu:1
#SBATCH --time=00:10:00
module load singularity
singularity exec --nv --env PYTHONPATH=/usr/local/lib/python3.12/dist-packages /gpfs/scratch/bsc98/bsc098069/llm_benchmarking/images/vllm-benchmark-default-nsight.sif python3 benchmarks_simple/offline_profiler.py --result-dir='benchmarks_simple/definitive_results/initial/llama-2-7b/_250_230_2' --model=/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-2-7b --json=times --save-chrome-traces-folder=chrome_traces --max-num-batched-tokens=256000 --max-num-seqs=1024 --prompt-len='250' --output-len='230' --batch-size='2'
