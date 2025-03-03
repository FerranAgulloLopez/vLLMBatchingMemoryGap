import os
import json
import glob
from typing import List, Tuple, Dict, Set
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Dynamic color mapping for output lengths
OUTPUT_LENGTH_COLOR_MAPPING = {
    130: 'blue',
    260: 'green',
    390: 'red',
    520: 'purple'
}
SORTED_OUTPUT_LENGTHS = sorted(OUTPUT_LENGTH_COLOR_MAPPING.keys())  # Ensure consistent order

def extract_experiment_metric(path: str) -> Dict[str, float]:
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
    # output['model_forward_total_time'] = np.mean(data['model_forward_total_time'].to_numpy())
    output['gpu_cache_usage_perc'] = np.max(data['gpu_cache_usage_perc'].to_numpy())

    batch_size = sum(num_running[time < duration]) / len(time[time < duration]) if len(time) > 0 else 0
    output['batch_size_real_mean'] = batch_size
    output['batch_size_real_max'] = np.max(num_running[time < duration]) if len(time) > 0 else 0
    output['duration'] = duration
    output['mean_ttft_ms'] = float(metrics['mean_ttft_ms'])
    output['mean_tpot_ms'] = float(metrics['mean_tpot_ms'])
    output['total_token_throughput'] = float(metrics['total_token_throughput'])

    return output

def extract_results(path: str) -> List[Dict[str, float]]:
    results = []
    for subdir, dirs, _ in os.walk(path):
        for folder in dirs:
            try:
                batch_size = int(folder.split('_')[1])
                num_prompts = int(folder.split('__')[-1])
                metrics = extract_experiment_metric(os.path.join(path, folder))
                metrics['batch_size'] = batch_size
                metrics['num_prompts'] = num_prompts
                results.append(metrics)
            except (ValueError, IndexError):
                continue
    return results

def plot_results(results_by_model: Dict[str, Dict[int, List[Dict[str, float]]]], path: str) -> None:
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    # Ensure axis limits are consistent across all subplots
    all_batch_sizes = []
    all_throughputs = []
    for results_by_output_len in results_by_model.values():
        for results in results_by_output_len.values():
            all_batch_sizes.extend([r['gpu_cache_usage_perc'] for r in results])
            all_throughputs.extend([r['total_token_throughput'] for r in results])

    x_min, x_max = min(all_batch_sizes), max(all_batch_sizes)
    y_min, y_max = min(all_throughputs), max(all_throughputs)

    a = 0
    for ax, (model, results_by_output_len) in zip(axes, results_by_model.items()):
        i = 0
        ax.set_title(f'{model}')
        for output_len, results in results_by_output_len.items():
            batch_sizes = [r['gpu_cache_usage_perc']*100 for r in results]
            throughputs = [r['total_token_throughput'] for r in results]

            # Sort by batch size for a smooth curve
            sorted_indices = np.argsort(batch_sizes)
            batch_sizes = np.array(batch_sizes)[sorted_indices]
            print(batch_sizes)
            throughputs = np.array(throughputs)[sorted_indices]

            ax.plot(batch_sizes, throughputs, marker='o', linestyle='solid',
                    label=f'{output_len} tokens', color=colors_list[i])
            i += 1

        if a == 2 or a == 3:
            ax.set_xlabel('GPU cache usage [%]')

        if a == 0 or a == 2:
            ax.set_ylabel('Throughput (token/s)')
        # ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y_min, y_max)
        ax.set_xlim(0,101)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', which='major')
        # ax.grid(True)
        a += 1

    plt.tight_layout()
    # plt.suptitle('', fontsize=20, y=1.02)
    plt.savefig(os.path.join(path, 'throughput_comparison_all_models.pdf'), format='pdf', bbox_inches='tight')
    plt.show()


def main():
    models = ["opt-1.3b", "opt-2.7b", "llama-2-7b", "llama-2-13b"]
    output_lengths = [130, 260, 390, 520]

    results_by_model = {}
    for model in models:
        results_by_output_len = {}
        for output_len in output_lengths:
            results = extract_results(f'../_{output_len}/{model}')
            results_by_output_len[output_len] = results
        results_by_model[model] = results_by_output_len

    plot_results(results_by_model, '.')

if __name__ == '__main__':
    main()
