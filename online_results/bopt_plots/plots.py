import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Tuple

def extract_experiment_metric(path: str) -> Dict[str, float]:
    output: Dict[str, float] = {}
    filenames: List[str] = glob.glob(os.path.join(path, 'openai-*.json'))
    filenames = [f for f in filenames if 'intermediate' not in f]
    
    if len(filenames) != 1:
        raise ValueError(f'More than one output result file or none {filenames} for path {path}')
    
    with open(filenames[-1]) as metrics_file:
        metrics: dict = json.load(metrics_file)
    
    output['throughput'] = float(metrics['request_throughput'])
    
    data = pd.read_csv(os.path.join(path, 'metrics_engine.csv'))
    time = data['time [s]'].to_numpy()
    num_running = data['num_requests_running'].to_numpy()
    duration = float(metrics['duration'])
    
    batch_size = np.mean(num_running[:np.searchsorted(time, duration)])
    batch_size_max = np.max(num_running)
    
    output.update({
        'batch_size_real_mean': batch_size,
        'batch_size_real_max': batch_size_max,
        'duration': duration,
        'mean_ttft_ms': float(metrics['mean_itl_ms']),
        'total_token_throughput': float(metrics['total_token_throughput'])
    })
    
    return output

def extract_results(path: str) -> List[Dict[str, float]]:
    results = []
    for subdir, dirs, _ in os.walk(path):
        for folder in dirs:
            try:
                metrics = extract_experiment_metric(os.path.join(path, folder))
                metrics['batch_size'] = int(folder.split('_')[1])
                results.append(metrics)
            except ValueError:
                continue
    results.sort(key=lambda x: x['batch_size'])  # Sort results by batch size

    return results

def plot_results(results: List[Dict[str, float]], output_path: str):
    batch_sizes = np.array([r['batch_size'] for r in results])
    throughputs = np.array([r['throughput'] for r in results])
    latencies = np.array([r['mean_ttft_ms'] for r in results])
    
    dT_dlogB = np.diff(throughputs) / np.diff(np.log2(batch_sizes))
    dL_dlogB = np.diff(latencies) / np.diff(np.log2(batch_sizes))
    
    relative_improvement_throughput = np.diff(throughputs) / throughputs[:-1]
    relative_improvement_latency = np.diff(latencies) / latencies[:-1]
    
    L_threshold = 0.10  # 10% increase threshold
    epsilon = 0.4  # 40% improvement cutoff
    plateau_indices = np.where((relative_improvement_throughput < epsilon) & (relative_improvement_latency > L_threshold))[0]
    B_opt_latency = batch_sizes[plateau_indices[0] + 1] if len(plateau_indices) > 0 else batch_sizes[-1]
    
    plt.rcParams.update({
        'font.size': 15,
        'axes.grid': True,
        'grid.linestyle': '--',
        'figure.figsize': (7, 5)
    })
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    colors_list = ['#0072B2', '#E69F00']
    
    axs[0].plot(latencies, throughputs, 'co-')
    axs[0].axvline(x=latencies[np.where(batch_sizes == B_opt_latency)[0][0]], color=colors_list[0], linestyle='--', label=f"Optimal Batch Size: {B_opt_latency}")
    axs[0].set_xlabel("Latency (ms)")
    axs[0].set_ylabel("Throughput (tokens/sec)")
    axs[0].set_title("Throughput vs Latency Trade-off")
    axs[0].legend()
    
    axs[1].plot(batch_sizes[:-1], dT_dlogB, 'ro-', label="dT/dlogB", color=colors_list[1])
    axs[1].axvline(x=B_opt_latency, linestyle='--', label=f"Optimal Batch Size: {B_opt_latency}", color=colors_list[0])
    axs[1].set_xscale("log")
    axs[1].set_xlabel("Batch Size")
    axs[1].set_ylabel("dT/dlogB (Throughput)")
    axs[1].set_title("First Derivative of Throughput & Latency")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path)

def main():
    model = "opt-1.3b"
    results_mean = extract_results(f'../output_length/_390/{model}')
    plot_results(results_mean, f'./plots_{model}.pdf')

if __name__ == '__main__':
    main()
