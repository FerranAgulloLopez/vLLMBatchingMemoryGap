- Create base docker image: `docker build -f Dockerfile -t vllm .`
- Create singularity image with additional requirements to run benchmark:
  - Default: `sudo singularity build vllm-benchmark-default.sif benchmarks/deployment/slurm/SingularityBenchmark.def`
  - With Nsight: `sudo singularity build vllm-benchmark-default-nsight.sif benchmarks/deployment/slurm/SingularityBenchmark_nsight.def`
- Schedule different benchmarks e.g.:
  ```
  EXP_CONTAINER_IMAGE=/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/images/vllm-benchmark-default-nsight.sif PYTHONPATH=. python3 benchmarks/deployment/slurm/launcher.py --user bsc98 --queue acc_debug --max-duration 00:10:00 --results-path benchmarks/definitive_results/test --default-server-args "{'--disable-log-requests': '', '--model': '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-2-7b'}" --default-benchmark-args "{'--backend': 'openai', '--disable-tqdm': '', '--dataset-path': '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/data/dummy_dataset_mean.json', '--endpoint': '/v1/completions', '--model': '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-2-7b', '--save-result': ''}" --test-server-args "{}" --test-benchmark-args "{'--num-prompts': ['100']}"
  ```
