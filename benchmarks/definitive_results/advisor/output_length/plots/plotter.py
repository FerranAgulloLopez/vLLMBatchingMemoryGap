import os
import json
import glob
from typing import List, Tuple, Dict, Set
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd

LABEL_COLOR_MAPPING = {
    500: 'blue',
    1000: 'green',
    2000: 'red'
}
SORTED_LABELS = sorted(LABEL_COLOR_MAPPING.keys())  # Ensure consistent order of labels


def extract_experiment_metric(path: str) -> Dict[str, float]:
    output: Dict[str, float] = {}
    filenames: List[str] = glob.glob(os.path.join(path, 'openai-*.json'))
    for i in range(len(filenames) - 1, -1, -1):
        if 'intermediate' in filenames[i]:
            del filenames[i]
    if len(filenames) != 1:
        raise ValueError(f'More than one output result file or none {filenames} for path {path}')
    with open(filenames[-1]) as metrics_file:
        metrics: dict = json.load(metrics_file)

    # compute throughput
    output['throughput'] = float(metrics['request_throughput'])

    # compute mean batch size
    data = pd.read_csv(os.path.join(path, 'metrics_engine.csv'))
    time = data['time [s]'].to_numpy()
    num_running = data['num_requests_running'].to_numpy()
    duration = float(metrics['duration'])
    output['num_preemptions_total'] = max(data['num_preemptions_total'].to_numpy())
    output['num_requests_running'] = np.max(data['num_requests_running'].to_numpy())
    output['num_requests_waiting'] = np.max(data['num_requests_waiting'].to_numpy())
    output['model_forward_total_time'] = np.mean(data['model_forward_total_time'].to_numpy())
    output['gpu_cache_usage_perc'] = np.max(data['gpu_cache_usage_perc'].to_numpy())

    batch_size = 0
    index = 0
    batch_size_max = 0
    while index < len(num_running) and time[index] < duration:
        batch_size += num_running[index]
        batch_size_max = max(batch_size_max,num_running[index] )
        index += 1
    if index > 0:
        batch_size /= index
    output['batch_size_real_mean'] = batch_size
    output['batch_size_real_max'] = batch_size_max
    output['duration'] = float(metrics['duration'])
    output['mean_ttft_ms'] = float(metrics['mean_ttft_ms'])
    output['mean_tpot_ms'] = float(metrics['mean_tpot_ms'])
    output['total_token_throughput'] = float(metrics['total_token_throughput'])

    return output


def extract_results(path: str) -> List[Dict[str, float]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['batch_size', 'num_prompts']
    results = []
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            batch_size: int = int(folder.split('_')[1])
            num_prompts: int = int(folder.split('__')[-1])
            try:
                metrics = extract_experiment_metric(os.path.join(path, folder))
            except ValueError:
                metrics = {}
            metrics['batch_size'] = batch_size
            metrics['num_prompts'] = num_prompts
            _id = create_id(metrics, id_metrics)
            if _id in collected_ids:
                raise ValueError('Repeated results')
            collected_ids.add(_id)
            results.append(metrics)
    return results


def plot_results(
        results: List[Dict[str, float]],
        path: str
) -> None:
    def __prepare_lines(results: List[Dict[str, float]], x_axis: str, y_axis: str, selection: str) -> List[Tuple[int, List[int], List[float]]]:
        output_tmp: Dict[int, Tuple[List[int], List[float]]] = {label: ([], [], []) for label in SORTED_LABELS}
        for item in results:
            selection_id = int(item[selection])  # Ensure numerical keys
            if selection_id not in output_tmp:
                continue  # Skip labels not in the mapping
            output_tmp[selection_id][0].append(item["batch_size"])
            print(item)
            output_tmp[selection_id][1].append(item[x_axis])
            output_tmp[selection_id][2].append(item[y_axis])

        output: List[Tuple[int, List[int], List[float]]] = []
        for key in SORTED_LABELS:
            batch_size, x_values, y_values = output_tmp[key]
            sorted_indices = np.argsort(x_values)
            x_line = [x_value for index, x_value in enumerate(x_values) if x_value is not None and y_values[index] is not None]
            y_line = [y_value for index, y_value in enumerate(y_values) if y_value is not None and x_values[index] is not None]
            x_line = [x_value for _, x_value in sorted(zip(batch_size, x_line)) ]
            y_line = [y_value for _, y_value in sorted(zip(batch_size, y_line))]
            output.append((key, x_line, y_line))
        return output

    # Use the global label and color mapping in plotting
    def __plot_lines(ax, label_x, label_y, lines):
        for key, x_line, y_line in lines:
            color = LABEL_COLOR_MAPPING[key]
            ax.plot(
                x_line, y_line,
                marker='o', linestyle='solid',
                label=f"{key}",
                color=color
            )
            for i, (x, y) in enumerate(zip(x_line, y_line)):
                if key == 2000:
                    ax.annotate(
                        f"{2**i}",        # Label includes the value as 2 raised to the power of the index
                        (x, y),           # Coordinates of the point
                        textcoords="offset points",
                        xytext=(5, 12),    # Offset to prevent overlap with the marker
                        ha='center',
                        fontsize=10
                    )
        ax.set_ylabel(label_y, fontsize=12)
        ax.set_xlabel(label_x, fontsize=12)
        ax.legend(loc='center right', fontsize=12)

    # Plot throughput/latency
    results1 = __prepare_lines(results, 'mean_tpot_ms', 'total_token_throughput', 'num_prompts')
    results2 = __prepare_lines(results, 'mean_ttft_ms', 'total_token_throughput', 'num_prompts')

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    __plot_lines(axs[0], 'Mean TPOT (ms)', 'Throughput (token/s)', results1)
    __plot_lines(axs[1], 'Mean TTFT (ms)', 'Throughput (token/s)', results2)

    plt.savefig(os.path.join(path, 'throughput'), bbox_inches='tight')

    # Plot throughput/real batch size
    results1 = __prepare_lines(results, 'batch_size_real_max', 'total_token_throughput', 'num_prompts')
    results2 = __prepare_lines(results, 'batch_size_real_max', 'mean_tpot_ms', 'num_prompts')
    results3 = __prepare_lines(results, 'batch_size_real_max', 'mean_ttft_ms', 'num_prompts')

    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    __plot_lines(axs[0], 'Max real batch size', 'Throughput (token/s)', results1)
    __plot_lines(axs[1], 'Max real batch size', 'Mean TPOT (ms)', results2)
    __plot_lines(axs[2], 'Max real batch size', 'Mean TTFT (ms)', results3)

    plt.savefig(os.path.join(path, 'throughput2'), bbox_inches='tight')

    # Plot gpu cache usage
    results1 = __prepare_lines(results, 'gpu_cache_usage_perc', 'total_token_throughput', 'num_prompts')
    results2 = __prepare_lines(results, 'num_requests_waiting', 'num_requests_running', 'num_prompts')
    plot_bachsize_real = __prepare_lines(results, 'gpu_cache_usage_perc', 'batch_size_real_max', 'num_prompts')

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    # __plot_lines(axs[0], 'GPU Cache Usage (%)', 'Throughput (token/s)', results1)

    for k, (key, x_line, y_line) in enumerate(results1):
        if key != 500:
            continue
        color = LABEL_COLOR_MAPPING[key]
        axs[0].plot(
            x_line, y_line,
            marker='o', linestyle='solid',
            label=f"{key}",
            color=color
        )
        for i, (x, y) in enumerate(zip(x_line, y_line)):
            if key == 500:
                bs = int(plot_bachsize_real[k][2][i])
                axs[0].annotate(
                    f"{bs}",     
                    (x, y),           # Coordinates of the point
                    textcoords="offset points",
                    xytext=(5, 12),    # Offset to prevent overlap with the marker
                    ha='center',
                    fontsize=10
                )
    axs[0].set_ylabel('Throughput (token/s)', fontsize=12)
    axs[0].set_xlabel('GPU Cache Usage (%)', fontsize=12)
    axs[0].legend(loc='center right', fontsize=12)

    __plot_lines(axs[1], 'Requests Waiting', 'Requests Running', results2)
    plt.savefig(os.path.join(path, 'serving_metrics'), bbox_inches='tight')

    # Third set of bar plots
    results1 = __prepare_lines(results, 'batch_size', 'batch_size_real_mean', 'num_prompts')
    results2 = __prepare_lines(results, 'batch_size', 'batch_size_real_max', 'num_prompts')

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    bar_width = 0.3
    for index, (label_x, label_y, lines) in enumerate(
            [
                ('Batch Size', 'Mean Batch Size', results1),
                ('Batch Size', 'Max Batch Size', results2),
            ]):
        for i, (key, x_line, y_line) in enumerate(lines):
            color = LABEL_COLOR_MAPPING[key]
            x_positions = np.arange(len(x_line)) # + i * bar_width
            x_labels = [f"{2**x}" for x in range(0,len(x_line))]
            max_values = [2**x for x in range(0, len(y_line))]
            adjusted_positions = [pos + (i * bar_width) for pos in x_positions]
            axs[index].bar(adjusted_positions, max_values, width=bar_width, color='lightgrey')
            axs[index].bar(adjusted_positions, y_line, bar_width, label=f"{key}", color=color)
            axs[index].set_xticks([pos + (1 * bar_width) for pos in x_positions]) 
            axs[index].set_xticklabels(x_labels, rotation=45) 
        axs[index].set_xlabel(label_x, fontsize=12)
        axs[index].set_ylabel(label_y, fontsize=12)
        axs[index].legend(loc='upper left', fontsize=12)

    plt.savefig(os.path.join(path, 'serving_metrics2'), bbox_inches='tight')


def main():
    model = "opt-1.3b"
    results_mean: List[Dict[str, float]] = extract_results('../batching_results/{}'.format(model))
    plot_results(results_mean, './plots_{}'.format(model))


if __name__ == '__main__':
    main()
