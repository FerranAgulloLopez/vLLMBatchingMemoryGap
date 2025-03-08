import os
import argparse
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
    
    data = pd.read_csv(os.path.join(path, 'metrics_engine_server_0.csv'))
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


def plot_results(results: List[Dict[str, float]], output_path: str, epsilon: float = 0.2, L: int = 5):
    batch_sizes = np.array([r['batch_size'] for r in results])
    throughputs = np.array([r['throughput'] for r in results])
    latencies = np.array([r['mean_ttft_ms'] for r in results])

    idx_32 = np.argmin(np.abs(batch_sizes - 32))
    latency_32 = latencies[idx_32]
    latency_threshold = L * latency_32  # Define the latency limit

    print(f"Latency at B=32: {latency_32:.2f} ms, Latency Threshold: {latency_threshold:.2f} ms")

    dT_dB = np.array([r['throughput'] for r in results]) / (np.array([r['batch_size'] for r in results])*throughputs[0])
    plateau_indices = np.where(dT_dB > epsilon)[0]
    valid_latency_indices = np.where(latencies <= latency_threshold)[0]
    valid_indices = np.intersect1d(plateau_indices, valid_latency_indices)
    B_opt = batch_sizes[np.max(valid_indices)]  # Largest batch size within latency limit

    print(f"Optimal Batch Size (B_opt): {B_opt}")

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 13,
        'axes.titlesize': 19,
        'axes.labelsize': 19,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'lines.linewidth': 2.0,
        'mathtext.default': 'regular',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.4,
        'figure.figsize': (7, 5)  # Consistent size for single plot
    })
    colors_list = ['#0072B2', '#E69F00', '#009E73', '#D55E00']
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))


    # Throughput vs Latency Trade-off
    axs[0].plot(latencies, throughputs, 'co-')
    axs[0].axvline(
        x=latencies[np.where(batch_sizes == B_opt)[0][0]], 
        color=colors_list[2], linestyle='--', label=f"Optimal Batch: {B_opt}"
    )
    axs[0].set_xlabel("Latency (ms)")
    axs[0].set_ylabel("Throughput (tokens/sec)")
    axs[0].set_title("Throughput vs Latency Trade-off")
    axs[0].legend()

    axs[1].plot(batch_sizes, dT_dB, 'ro-', color=colors_list[1])
    axs[1].axvline(x=B_opt, linestyle='--', label=f"Optimal Batch: {B_opt}", color=colors_list[2])
    axs[1].axhline(y=epsilon, color=colors_list[3], linestyle="--", label=f"Threshold ε={epsilon}")
    axs[1].set_xlabel("Batch Size")
    axs[1].set_ylabel("Throughput Gain")
    axs[1].set_title("Throughput Gain per Batch Increase")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Optimize throughput-latency trade-off using F_beta.")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--l", type=float, default=5)
    args = parser.parse_args()

    model = "opt-1.3b"
    results_mean = extract_results(
        f'/gpfs/scratch/bsc98/bsc098949/vLLMServingPlateau/benchmarks/definitive_results/background/{model}'
    )

    # Pass the user-specified beta to the plot function
    plot_results(results_mean, f'./plots_{model}.pdf', args.epsilon, args.l)


if __name__ == '__main__':
    main()