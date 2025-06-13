import os
import json
import glob
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Total memory available per GPU (64 GB)
TOTAL_GPU_MEMORY = 64 * 1024  # In MB

# Colors for the stacked bar plot
COLORS = ['#4CAF50', '#FF9800', '#2196F3', '#BDBDBD']  # Model Weights, Optimal KV Cache, Extra KV Cache, Unused Memory

# Predefined model information (memory usage in GB)
models_info = {
    "opt-1.3b": {'kv_cache_usage_gb': 54.48, 'model_weight_usage_gb': 3.11},
    "opt-2.7b": {'kv_cache_usage_gb': 51.97, 'model_weight_usage_gb': 5.61},
    "llama-2-7b": {'kv_cache_usage_gb': 44.36, 'model_weight_usage_gb': 13.15},
    "llama-2-13b": {'kv_cache_usage_gb': 32.61, 'model_weight_usage_gb': 24.89},
}

# Path to the results folder
RESULTS_PATH = "../_390/"  # Adjust this path based on your folder structure


def extract_experiment_metric(path: str) -> Dict[str, float]:
    """Extract relevant metrics from the experiment results."""
    output: Dict[str, float] = {}
    filenames: List[str] = glob.glob(os.path.join(path, 'openai-*.json'))
    filenames = [f for f in filenames if 'intermediate' not in f]

    if len(filenames) != 1:
        raise ValueError(f'More than one output result file or none {filenames} for path {path}')

    with open(filenames[0]) as metrics_file:
        metrics: dict = json.load(metrics_file)

    # Compute throughput
    output['throughput'] = float(metrics['request_throughput'])

    # Compute mean batch size
    data = pd.read_csv(os.path.join(path, 'metrics_engine.csv'))
    time = data['time [s]'].to_numpy()
    num_running = data['num_requests_running'].to_numpy()
    duration = float(metrics['duration'])

    output['num_preemptions_total'] = max(data['num_preemptions_total'].to_numpy())
    output['num_requests_running'] = np.max(data['num_requests_running'].to_numpy())
    output['num_requests_waiting'] = np.max(data['num_requests_waiting'].to_numpy())
    output['model_forward_total_time'] = np.mean(data['model_forward_total_time'].to_numpy())
    output['gpu_cache_usage_perc'] = np.max(data['gpu_cache_usage_perc'].to_numpy())
    output['mean_ttft_ms'] = float(metrics['mean_ttft_ms'])
    output['mean_tpot_ms'] = float(metrics['mean_tpot_ms'])

    batch_size = sum(num_running[time < duration]) / len(time[time < duration]) if len(time) > 0 else 0
    output['batch_size_real_mean'] = batch_size
    output['batch_size_real_max'] = np.max(num_running[time < duration]) if len(time) > 0 else 0

    if len(filenames) != 1:
        raise ValueError(f'Expected one output result file, found {len(filenames)} for path {path}')

    with open(filenames[0]) as metrics_file:
        metrics = json.load(metrics_file)

    # Read CSV to extract GPU cache usage percentage
    data = pd.read_csv(os.path.join(path, 'metrics_engine.csv'))
    output['gpu_cache_usage_perc'] = np.max(data['gpu_cache_usage_perc'].to_numpy())

    # Extract throughput and batch size
    output['batch_size'] = output['batch_size_real_max']

    return output


def compute_optimal_batch_size(results: List[Dict[str, float]]) -> int:
    """Compute the optimal batch size based on throughput and latency conditions."""
    if not results:
        return 0

    # Sort results by batch size
    results = sorted(results, key=lambda x: x['batch_size'])

    throughputs = np.array([r['throughput'] for r in results])
    batch_sizes = np.array([r['batch_size'] for r in results])
    latencies = np.array([r['mean_tpot_ms'] for r in results])

    L = 3
    idx_32 = np.argmin(np.abs(batch_sizes - 32))
    latency_32 = latencies[idx_32]
    latency_threshold = L * latency_32  # Define the latency limit
    epsilon = 0.3
    dT_dB = np.diff(throughputs) / np.diff(batch_sizes)
    plateau_indices = np.where(dT_dB < epsilon)[0]
    valid_latency_indices = np.where(latencies <= latency_threshold)[0]
    valid_indices = np.intersect1d(plateau_indices, valid_latency_indices)
    b_opt = batch_sizes[np.max(valid_latency_indices)]  # Largest batch size within latency limit

    return b_opt


def extract_results(path: str) -> Dict[str, Dict[str, float]]:
    """Extract metrics for each model and calculate optimal batch size."""
    results_by_model = {}

    for model in models_info.keys():
        model_path = os.path.join(path, model)
        results = []

        for subdir, _, _ in os.walk(model_path):
            try:
                metrics = extract_experiment_metric(subdir)
                results.append(metrics)
            except (ValueError, FileNotFoundError):
                continue

        if results:
            b_opt = compute_optimal_batch_size(results)
            optimal_cache_usage_perc = next(
                (r['gpu_cache_usage_perc'] for r in results if r['batch_size'] == b_opt), 0
            )

            kv_cache_full = models_info[model]['kv_cache_usage_gb']
            print(f"KV Cache Usage for {model}: {kv_cache_full} GB")
            print(f"Optimal Cache Usage for {model}: {optimal_cache_usage_perc}%")
            kv_cache_optimal = kv_cache_full * optimal_cache_usage_perc
            kv_cache_extra = kv_cache_full - kv_cache_optimal

            results_by_model[model] = {
                'model_weight_usage_gb': models_info[model]['model_weight_usage_gb'],
                'kv_cache_full_gb': kv_cache_full,
                'kv_cache_optimal_gb': kv_cache_optimal,
                'kv_cache_extra_gb': kv_cache_extra,
                'b_opt': b_opt
            }

    return results_by_model


def plot_memory_usage(results_by_model: Dict[str, Dict[str, float]]):
    """Plot the memory usage breakdown per model and show the percentage of free space gained."""
    models = list(results_by_model.keys())
    model_weights = [results_by_model[model]['model_weight_usage_gb'] for model in models]
    kv_cache_optimal = [results_by_model[model]['kv_cache_optimal_gb'] for model in models]
    kv_cache_extra = [results_by_model[model]['kv_cache_extra_gb'] for model in models]

    # Calculate unused memory and percentage gains
    unused_memory = [64 - (mw + opt + extra) for mw, opt, extra in zip(model_weights, kv_cache_optimal, kv_cache_extra)]
    gains_percentage = [(extra / 64) * 100 for extra in kv_cache_extra]

    # Print the results with percentage gain
    for model, gain, percentage in zip(models, kv_cache_extra, gains_percentage):
        print(f"Gains for {model}: {gain:.2f} GB ({percentage:.2f}% of total GPU memory)")

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size':9,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'lines.linewidth': 2.0,
        'mathtext.default': 'regular',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.4,
        'figure.figsize': (7, 5)  # Consistent size for single plot
    })

    colors_list = ['#0072B2', '#E69F00', '#009E73', '#D55E00']
    markers = ['o', 's', 'D', '^']  # Example markers for legend

    bar_width = 0.5
    fig, ax = plt.subplots(figsize=(5, 2))

    # Plot stacked bars
    ax.bar(models, model_weights, bar_width, label='Model Weights', color=colors_list[0], edgecolor='black')
    ax.bar(models, kv_cache_optimal, bar_width, bottom=model_weights, label='KV Cache', color=colors_list[1], edgecolor='black')
    ax.bar(models, kv_cache_extra, bar_width, bottom=np.array(model_weights) + np.array(kv_cache_optimal),
           label='Extra KV Cache', color=colors_list[2], edgecolor='black')
    ax.bar(models, unused_memory, bar_width,
           bottom=np.array(model_weights) + np.array(kv_cache_optimal) + np.array(kv_cache_extra),
           label='Other', color=colors_list[3], edgecolor='black')

    ax.set_ylabel('Memory Usage (GB)', fontsize=10)
    ax.set_ylim(0, 64)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)

    # Create custom legend elements
    legend_handles = [
        Line2D([0], [0], color=colors_list[i], linestyle='-', linewidth=3, 
               label=label, markeredgewidth=1, markeredgecolor='black')
        for i, label in enumerate(['Model Weights', 'KV Cache', 'Extra KV Cache', 'Other'])
    ]

    # Move the legend outside of the plot
    fig.legend(handles=legend_handles, loc='upper center', ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.55, 1.2))

    plt.tight_layout()
    plt.savefig('gpu_memory_distribution_with_bopt_real_data.pdf', bbox_inches='tight', dpi=400)
    plt.show()


def main():
    results_by_model = extract_results(RESULTS_PATH)
    if results_by_model:
        plot_memory_usage(results_by_model)
    else:
        print("No valid results found for the given models.")


if __name__ == '__main__':
    main()
