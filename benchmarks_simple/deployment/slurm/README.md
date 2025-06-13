- Create base docker image: `docker build -f Dockerfile -t vllm .`
- Create singularity image with additional requirements to run benchmark:
  - Default: `sudo singularity build vllm-benchmark-default.sif benchmarks_simple/deployment/slurm/SingularityBenchmark.def`
  - With Nsight: `sudo singularity build vllm-benchmark-default-nsight.sif benchmarks_simple/deployment/slurm/SingularityBenchmark_nsight.def`
- Schedule different benchmarks e.g.:
  ```
  EXP_CONTAINER_IMAGE=/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/images/vllm-benchmark-default-nsight.sif PYTHONPATH=. python3 benchmarks_simple/deployment/slurm/launcher.py --user bsc98 --queue acc_debug --max-duration 00:10:00 --results-path benchmarks_simple/definitive_results/test --default-args "{'--model': '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-2-7b', '--json': 'times',  '--save-chrome-traces-folder': 'chrome_traces', '--max-num-batched-tokens': '256000', '--max-num-seqs': '1024'}" --test-args "{'--prompt-len': ['250'], '--output-len': ['230'], '--batch-size': ['1']}"
  ```