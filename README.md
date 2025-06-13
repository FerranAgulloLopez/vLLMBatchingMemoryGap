# Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference
_This repository was created through a fork of the [vLLM repository](https://github.com/vllm-project/vllm) as a result of the research manuscript [Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference](https://arxiv.org/abs/2503.08311)_

### Introduction
This repository includes the code modifications made to the original vLLM repository to support the experiments presented in the mentioned manuscript. It also contains all the results used to generate the manuscript's tables and figures.

The sections below detail:
- The specific code changes
- The structure of the modified repository
- The required steps for setting up the environment
- The required steps to run the experiments

### Code changes
This repository includes modifications only to the benchmarking components of the serving system. The core server part remains unchanged. We work in two benchmark modes following the available code in the vLLM respository, each located in a separate directory:

- Default Benchmarking (Online Serving): Located in the _benchmarks_ folder, this is the standard online serving benchmark.
- Simple Benchmarking (Offline Serving): Located in the _benchmarks_simple_ folder, this mode performs offline serving through direct Python calls to the server.

Our main changes in these two modes are the following:
- Default Benchmarking: An updated benchmark script, _benchmarks/benchmark_serving_with_metrics.py_, has been added with the following new features:
  - Metrics logging: It collects the metrics from the vLLM Prometheus endpoint and nvidia-smi command throughout the benchmark duration. This is handled in _benchmarks/concurrent_metrics_checker.py_ and can be disabled using the `--disable-log-stats` flag.
  - Integrated server launching: It can launch the server automatically using the `--launch-server` flag. Server arguments can be passed with `--server-args`. This simplifies the process of running a benchmark from a single entry point.
  - Multi-Server support: It is able to launch multiple server instances that share GPU resources through the `--multiple-servers` flag. Benchmark requests will be evenly distributed across the configured servers.
  - MPS support: When using server replication, NVIDIA MPS (Multi-Process Service) can be enabled via `--use-mps` flag.
  - HPC Slurm deployment: We included the required steps and data to deploy and run this benchmark with Slurm in HPC environments (singularity images) in the directory _benchmarks/deployment/slurm_.
  - Batch running: It includes the script _benchmarks/deployment/slurm/launcher_ that acts as a launcher to run multiple experiments with different configurations transparent from Slurm.
  - Nsight Systems profiling: The previous launcher script allows enabling NVIDIA Nsight Systems profiling with `--with-nsight` flag. The corresponding arguments can be included via the flag `--nsight-args`.
- Simple Benchmarking: An updated benchmark script, _benchmarks_simple/decoding_offline_profiler.py_, has been added with the following new features:
  - Phase control: It controls the prefill and decode phases of requests. It makes all requests to run their prefill phase first, and, once all are finished, it runs their decode phase.
  - NVTX: It produces NVTX signals that mark the different parts of the execution, including the prefill and decode phases, for improving the subsequent analysis of outcome results. This option is enabled through the flag `--include-nvtx-regions`.
  - HPC Slurm deployment: We included the required steps and data to deploy and run this benchmark with Slurm in HPC environments (singularity images) in the directory _benchmarks_simple/deployment/slurm_.
  - Batch running: It includes the script _benchmarks_simple/deployment/slurm/launcher_ that acts as a launcher to run multiple experiments with different configurations transparent from Slurm.
  - Nsight Systems profiling: The previous launcher script allows enabling NVIDIA Nsight Systems profiling with `--with-nsight` flag. The corresponding arguments can be included via the flag `--nsight-args`.
  - Nsight Compute profiling: Similarly, the launcher script allows enabling NVIDIA Nsight Compute profiling with `--with-nsight-compute` flag. The corresponding arguments can be included via the flag `--nsight-compute-args`.

We also made a few minor updates to the _.gitignore_ file, adding new rules to prevent certain newly generated output files from being tracked, and removing specific rules to allow the inclusion of selected result files that we wanted to upload to the repository.

### Code structure
In accordance with the code changes, all updated content is inside the folders _benchmarks_ (online benchmarking) and _benchmarks_simple_ (offline benchmarking) except the _.gitignore_. Inside these folders, there are located the benchmark codes and their corresponding results. The structure of the main new and updated files and folders are as follows:
- [file] .gitignore
- [dir] benchmarks: including benchmark code and results for online experiments
  - [file] benchmark_serving_with_metrics.py: updated online benchmark script
  - [file] concurrent_metrics_checker.py: collector of prometheus and nvidia-smi metrics
  - [dir] deployment/slurm: files for deploying and running the online experiments
    - [file] launcher.py: script to launch multiple experiments with different configurations transparent from Slurm
    - [file] Singularity*.def: singularity image definition files used for running the experiments
    - [file] slurm.sh: base slurm script used by the launcher to run experiments on Slurm
  - [dir] definitive_results: collection of results/figures/tables for all paper online experiments
    - [dir] background: corresponding to Figures 2 and 3 of the paper. The directory includes the raw results from all used experiments and the script `chart.py` that creates the figures.
    - [dir] advisor: corresponding to Figures 10, 11 and 12 of the paper. The directory includes the raw results from all used experiments and the scripts `chart.py` that create the figures.
    - [dir] replication: corresponding to Figure 13 and Table IV of the paper. The directory includes the raw results from all used experiments and the scripts `chart.py` that create the figure and table.
- [dir] benchmarks_simple: including benchmark code and results for offline experiments
  - [file] decoding_offline_profiler.py: updated offline benchmark script
  - [dir] deployment/slurm: files for deploying and running the offline experiments
    - [file] launcher.py: script to launch multiple experiments with different configurations transparent from Slurm
    - [file] Singularity*.def: singularity image definition files used for running the experiments
    - [file] slurm.sh: base slurm script used by the launcher to run experiments on Slurm
  - [dir] definitive_results/analysis: collection of results/figures/tables for all paper offline experiments
    - [dir] prefill_vs_decode: corresponding to Figure 4 and Table 1. The directory includes the raw results from all used experiments and the scripts `chart.py` that create the figure and table.
    - [dir] decode_kernels: corresponding to Figures 5, 6 and 7. The directory includes the raw results from all used experiments and the scripts `chart.py` that create the figures.
    - [dir] attention_kernel: corresponding to Figure 1, 8 and 9 and Tables II and III. The directory includes the raw results from all used experiments and the scripts `chart.py` that create the figures and tables.

Each directory containing experiment results follows a consistent structure to ensure reproducibility and clarity:
- Command Tracking: Every set of experiment directories includes one or more .txt files listing the exact command-line instructions used to run the experiments. These invoke _launcher.py_ with varying input arguments, depending on the specific configuration.
- Output Files: Each specific experiment directory includes all output artifacts generated during the run. The exact files may vary with configuration, but the following are commonly present:
  - log_*.out: Standard output logs from the benchmark.
  - log_*.err: Error output logs from the benchmark.
  - server_out_*.log: Standard output from the server process.
  - server_err_*.log: Error output from the server process.

Note: Nsight Systems and Nsight Compute traces are not included in the repository due to their large size. However, all experiments can be easily reproduced by executing the corresponding commands found in the .txt config files.

### How to set up
We show how to reproduce the experiments that appear in the paper, where we use Singularity and Slurm, which must be installed prior to execution. Nevertheless, as the original vLLM code, everything can also be run with docker or plainly with Python. Follow the required steps:
- Create the base Docker image as the foundation for the Singularity image: `docker build -f Dockerfile -t vllm .`
- Create the Singularity image. If willing to use both online and offline modes, it is recommended to use the definition file found in the _benchmarks_simple_ deployment directory as appears in the following commands:
  - Default: `sudo singularity build vllm-benchmark-default.sif benchmarks_simple/deployment/slurm/SingularityBenchmark.def`
  - With Nsight: `sudo singularity build vllm-benchmark-default-nsight.sif benchmarks_simple/deployment/slurm/SingularityBenchmark_nsight.def`
- [OPT] If willing to use Nvidia Nsight Systems / Compute or MPS, the desired technologies should be installed as well prior to execution in the corresponding nodes.

### How to run
Once all required technologies are installed and the Singularity image has been built, you can easily reproduce the experiments from the paper using the commands provided in the .txt configuration files. For instance, to reproduce the LLaMA-2-7B results shown in Figures 2 and 3, refer to the file _benchmarks/definitive_results/background/llama-2-7b/config-55373.txt_. This file contains a ready-to-run command. Simply execute it from the root directory of the repository, and the corresponding experiments will be submitted to the Slurm queue for execution.

```
EXP_CONTAINER_IMAGE=/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/images/vllm-benchmark-default-nsight.sif \ 
PYTHONPATH=. \ 
python3 benchmarks/deployment/slurm/launcher.py 
--user bsc98 \ 
--queue acc_bsccs \ 
--max-duration 05:15:00 \ 
--results-path benchmarks/definitive_results/background/llama-2-7b \ 
--default-server-args "{'--disable-log-requests': '', '--model': '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-2-7b', '--max-model-len': '2048'}" \ 
--default-benchmark-args "{'--backend': 'openai', '--disable-tqdm': '', '--dataset-path': '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/data/ShareGPT_V3_unfiltered_cleaned_split.json', '--endpoint': '/v1/completions', '--model': '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-2-7b', '--save-result': '', '--num-prompts': '2000'}" \ 
--test-server-args "{'--max-num-seqs':['1', '2', '4', '8', '16', '32', '64', '96', '128', '192', '256', '512']}" \ 
--test-benchmark-args "{}"
```

As shown, the command invokes the launcher for the online benchmark. While the full list of arguments can be found in the corresponding launcher script, here is a summary of the key components defined in this example:

- `EXP_CONTAINER_IMAGE`: Environment variable specifying the Singularity image to use.
- `PYTHONPATH`: Set to the root of the repository to define the Python working directory.
- `--user`: Specifies the Slurm user account.
- `--queue`: Specifies the Slurm queue or partition.
- `--max-duration`: Sets the maximum allowed runtime for the Slurm job.
- `--results-path`: Path where all experiment results will be saved.
- `--default-server-args`: Arguments passed to the vLLM server for all experiments.
- `--default-benchmark-args`: Common benchmark arguments used across all experiments.
- `--test-server-args`: Server arguments that will vary across different experiments.
- `--test-benchmark-args`: Benchmark arguments that will vary across different experiments.

The launcher will automatically initiate an experiment for every combination of server and benchmark arguments defined via the `--test-server-args` and `--test-benchmark-args` flags. Each of these experiments will also include the default arguments specified by the `--default-server-args` and `--default-benchmark-args` flags. Every experiment is executed through the appropriate benchmark script, which also handles launching of the vLLM server.

### How to cite
If using these code modifications please cite this paper:
```
Recasens, P. G., Agullo, F., Zhu, Y., Wang, C., Lee, E. K., Tardieu, O., ... & Berral, J. L. (2025). Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference. arXiv preprint arXiv:2503.08311.
```
