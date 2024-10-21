import os
import json
import glob
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np

MAX_MEMORY_MB = 65247
MODEL_SIZE_MB = 12556.2


def extract_experiment_metrics(path: str) -> Tuple[float, float, float, float]:
    filenames: List[str] = glob.glob(os.path.join(path, 'openai-*.json'))
    for i in range(len(filenames) - 1, -1, -1):
        if 'intermediate' in filenames[i]:
            del filenames[i]
    if len(filenames) != 1:
        raise ValueError(f'More than one output result file or none {filenames} for path {path}')
    with open(filenames[0]) as metrics_file:
        metrics: dict = json.load(metrics_file)
    throughput: float = float(metrics['output_throughput'])
    mean_itl: float = float(metrics['mean_itl_ms'])
    median_itl: float = float(metrics['median_itl_ms'])
    p99_itl: float = float(metrics['p99_itl_ms'])
    return throughput, mean_itl, median_itl, p99_itl


def extract_maximum_batch_size(path: str) -> int:
    if not os.path.exists(os.path.join(path, 'num_running.npy')):
        raise ValueError('Array not found')
    num_running = np.load(os.path.join(path, 'num_running.npy'))
    return np.max(num_running)


def extract_results(path: str) -> Dict[str, Dict[float, float]]:
    results = {
        'throughput': {},
        'mean_itl': {},
        'median_itl': {},
        'p99_itl': {}
    }
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            memory_utilization: float = float(folder.split('_')[1])
            available_memory_size: float = (MAX_MEMORY_MB - MODEL_SIZE_MB) * memory_utilization
            available_memory_size /= 1000
            try:
                throughput, mean_itl, median_itl, p99_itl = extract_experiment_metrics(os.path.join(path, folder))
            except ValueError:
                throughput, mean_itl, median_itl, p99_itl = None, None, None, None
            if available_memory_size in results['throughput']:
                raise ValueError('Repeated results')
            if available_memory_size in results['mean_itl']:
                raise ValueError('Repeated results')
            if available_memory_size in results['median_itl']:
                raise ValueError('Repeated results')
            if available_memory_size in results['p99_itl']:
                raise ValueError('Repeated results')
            results['throughput'][available_memory_size] = throughput
            results['mean_itl'][available_memory_size] = mean_itl
            results['median_itl'][available_memory_size] = median_itl
            results['p99_itl'][available_memory_size] = p99_itl
    return results


def extract_results_batch(path: str) -> Dict[str, Dict[int, float]]:
    results = {
        'throughput': {},
        'mean_itl': {},
        'median_itl': {},
        'p99_itl': {}
    }
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            try:
                maximum_batch: int = extract_maximum_batch_size(os.path.join(path, folder))
            except ValueError:
                continue
            try:
                throughput, mean_itl, median_itl, p99_itl = extract_experiment_metrics(os.path.join(path, folder))
            except ValueError:
                throughput, mean_itl, median_itl, p99_itl = None, None, None, None
            if maximum_batch in results['throughput']:
                raise ValueError('Repeated results')
            if maximum_batch in results['mean_itl']:
                raise ValueError('Repeated results')
            if maximum_batch in results['median_itl']:
                raise ValueError('Repeated results')
            if maximum_batch in results['p99_itl']:
                raise ValueError('Repeated results')
            results['throughput'][maximum_batch] = throughput
            results['mean_itl'][maximum_batch] = mean_itl
            results['median_itl'][maximum_batch] = median_itl
            results['p99_itl'][maximum_batch] = p99_itl
    return results


def plot_results_together(
        S_results_mean: Dict[str, Dict[float, float]],
        S_results_p25: Dict[str, Dict[float, float]],
        S_results_p75: Dict[str, Dict[float, float]],
        metrics_to_show: List[str],
        path: str,
        title: str,
) -> None:
    def __prepare_lines(results: Dict[str, Dict[float, float]]) -> Dict[str, Tuple[List[float], List[float]]]:
        output: Dict[str, Tuple[List[float], List[float]]] = {}
        for metric, metric_results in results.items():
            available_memory = [adapters for adapters, metric_value in metric_results.items() if metric_value is not None]
            adapter_metrics = [metric_value for adapters, metric_value in metric_results.items() if metric_value is not None]
            adapter_metrics = [x for _, x in sorted(zip(available_memory, adapter_metrics))]
            available_memory.sort()
            output[metric] = (available_memory, adapter_metrics)
        return output

    S_results_p25 = __prepare_lines(S_results_p25)
    S_results_mean = __prepare_lines(S_results_mean)
    S_results_p75 = __prepare_lines(S_results_p75)

    values_to_show: List[str, dict] = [
        ('Small request length', S_results_p25),
        ('Medium request length', S_results_mean),
        ('Large request length', S_results_p75)
    ]

    fig, axs = plt.subplots(3, 1, figsize=(1 * 5, 3 * 3), sharex=True)
    fig.subplots_adjust(hspace=0.2)

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    for index_y, (label, values_S) in enumerate(values_to_show):
        for metric in metrics_to_show:
            x_line = values_S[metric][0]
            y_line = values_S[metric][1]
            line = axs[index_y].plot(
                x_line,
                y_line,
                marker='o',
                linestyle='solid',
                label=f'{metric}'
            )

            '''x_final_point = x_line[-1] + (x_line[-1] - x_line[-2])
            y_final_point = y_line[-1] + (y_line[-1] - y_line[-2])
            axs[index_x].plot(
                x_line[-1:] + [x_final_point],
                y_line[-1:] + [y_final_point],
                marker='x',
                markevery=[1],
                linestyle='dotted',
                color=line[0].get_color()
            )'''

            axs[index_y].set_title(f'{label}')
            if index_y == len(values_to_show) - 1:
                axs[index_y].set_xlabel('KV Cache (GB)')

            axs[index_y].legend(loc='upper right', fontsize=12)
            axs[index_y].set_ylabel('output tokens p/s', fontsize=10)

    plt.savefig(os.path.join(path, f'S_{title}'), bbox_inches='tight')


def plot_results_together_batch(
        S_results_mean: Dict[str, Dict[int, float]],
        S_results_p25: Dict[str, Dict[int, float]],
        S_results_p75: Dict[str, Dict[int, float]],
        metric_to_show: str,
        path: str,
        title: str,
) -> None:
    def __prepare_lines(results: Dict[str, Dict[int, float]]) -> Dict[str, Tuple[List[int], List[float]]]:
        output: Dict[str, Tuple[List[int], List[float]]] = {}
        for metric, metric_results in results.items():
            available_memory = [adapters for adapters, metric_value in metric_results.items() if metric_value is not None]
            adapter_metrics = [metric_value for adapters, metric_value in metric_results.items() if metric_value is not None]
            adapter_metrics = [x for _, x in sorted(zip(available_memory, adapter_metrics))]
            available_memory.sort()
            output[metric] = (available_memory, adapter_metrics)
        return output

    S_results_p25 = __prepare_lines(S_results_p25)
    S_results_mean = __prepare_lines(S_results_mean)
    S_results_p75 = __prepare_lines(S_results_p75)

    values_to_show: List[str, dict] = [
        ('small request length', S_results_p25),
        ('medium request length', S_results_mean),
        ('large request length', S_results_p75)
    ]

    fig, axs = plt.subplots(1, 1, figsize=(1 * 5, 1 * 3), sharex=True)
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    for index_y, (label, values_S) in enumerate(values_to_show):
        x_line = values_S[metric_to_show][0]
        y_line = values_S[metric_to_show][1]
        line = axs.plot(
            x_line,
            y_line,
            marker='o',
            linestyle='solid',
            label=f'{label}'
        )

        axs.set_xlabel('maximum batch size')
        axs.legend(loc='upper right', fontsize=12)
        axs.set_ylabel('output tokens p/s', fontsize=10)

    plt.savefig(os.path.join(path, f'S_{title}'), bbox_inches='tight')


def plot_results_together_batch_memory(
        S_results_mean_memory: Dict[str, Dict[float, float]],
        S_results_mean_batch: Dict[str, Dict[int, float]],
        S_results_p25_memory: Dict[str, Dict[float, float]],
        S_results_p25_batch: Dict[str, Dict[int, float]],
        S_results_p75_memory: Dict[str, Dict[float, float]],
        S_results_p75_batch: Dict[str, Dict[int, float]],
        metric_to_show: str,
        path: str,
        title: str,
) -> None:
    def __prepare_lines(results: Dict[str, Dict[int, float]]) -> Dict[str, Tuple[List[int], List[float]]]:
        output: Dict[str, Tuple[List[int], List[float]]] = {}
        for metric, metric_results in results.items():
            available_memory = [adapters for adapters, metric_value in metric_results.items() if metric_value is not None]
            adapter_metrics = [metric_value for adapters, metric_value in metric_results.items() if metric_value is not None]
            adapter_metrics = [x for _, x in sorted(zip(available_memory, adapter_metrics))]
            available_memory.sort()
            output[metric] = (available_memory, adapter_metrics)
        return output

    S_results_p25_memory = __prepare_lines(S_results_p25_memory)
    S_results_p25_batch = __prepare_lines(S_results_p25_batch)
    S_results_mean_memory = __prepare_lines(S_results_mean_memory)
    S_results_mean_batch = __prepare_lines(S_results_mean_batch)
    S_results_p75_memory = __prepare_lines(S_results_p75_memory)
    S_results_p75_batch = __prepare_lines(S_results_p75_batch)

    values_to_show: List[str, dict] = [
        ('small request length', S_results_p25_memory, S_results_p25_batch),
        ('medium request length', S_results_mean_memory, S_results_mean_batch),
        ('large request length', S_results_p75_memory, S_results_p75_batch)
    ]

    fig, axs = plt.subplots(1, 1, figsize=(1 * 5, 1 * 3), sharex=True)
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    for index_y, (label, values_S_memory, values_S_batch) in enumerate(values_to_show):
        x_line = values_S_batch[metric_to_show][0]
        y_line = values_S_memory[metric_to_show][0]
        line = axs.plot(
            x_line,
            y_line,
            marker='o',
            linestyle='solid',
            label=f'{label}'
        )

        axs.set_xlabel('maximum batch size')
        axs.legend(loc='upper right', fontsize=12)
        axs.set_ylabel('KV Cache (GB)', fontsize=10)

    plt.savefig(os.path.join(path, f'S_{title}'), bbox_inches='tight')


def main():
    S_results_mean_memory: Dict[str, Dict[float, float]] = extract_results('llama-2-7b/mean_dataset')
    S_results_p25_memory: Dict[str, Dict[float, float]] = extract_results('llama-2-7b/p25_dataset')
    S_results_p75_memory: Dict[str, Dict[float, float]] = extract_results('llama-2-7b/p75_dataset')
    plot_results_together(S_results_mean_memory, S_results_p25_memory, S_results_p75_memory, ['throughput'], '.', 'default_throughput')
    plot_results_together(S_results_mean_memory, S_results_p25_memory, S_results_p75_memory, ['mean_itl', 'median_itl'], '.', 'default_itl')
    S_results_mean_batch: Dict[str, Dict[int, float]] = extract_results_batch('llama-2-7b/mean_dataset')
    S_results_p25_batch: Dict[str, Dict[int, float]] = extract_results_batch('llama-2-7b/p25_dataset')
    S_results_p75_batch: Dict[str, Dict[int, float]] = extract_results_batch('llama-2-7b/p75_dataset')
    plot_results_together_batch(S_results_mean_batch, S_results_p25_batch, S_results_p75_batch, 'throughput', '.', 'default_throughput_batch')
    plot_results_together_batch_memory(S_results_mean_memory, S_results_mean_batch, S_results_p25_memory, S_results_p25_batch, S_results_p75_memory, S_results_p75_batch, 'throughput', '.', 'default_throughput_batch_memory')


if __name__ == '__main__':
    main()
