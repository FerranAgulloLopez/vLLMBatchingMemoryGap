#!/bin/bash
#SBATCH --job-name=_0.8_
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --output=benchmarks/definitive_results/theory/S_default/llama-2-7b/p25_dataset/_0.8_/log_%j.out
#SBATCH --error=benchmarks/definitive_results/theory/S_default/llama-2-7b/p25_dataset/_0.8_/log_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --gres gpu:1
#SBATCH --time=01:30:00
module load singularity
singularity exec --nv --env PYTHONPATH=. /gpfs/scratch/bsc98/bsc098069/llm_benchmarking/images/vllm-benchmark-default.sif python3 benchmarks/benchmark_serving_with_metrics.py --port='50037' --result-dir='benchmarks/definitive_results/theory/S_default/llama-2-7b/p25_dataset/_0.8_' --backend='openai' --disable-tqdm --dataset-path='/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/data/dummy_dataset_p25.json' --endpoint='/v1/completions' --model='/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-2-7b' --num-prompts='3000' --save-result --launch-server --server-args='--port=50037 --disable-log-requests --model=/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-2-7b --max-num-seqs=3000 --gpu-memory-utilization='0.8''