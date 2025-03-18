import os
import csv
import json
import glob
from typing import List, Tuple, Dict, Set
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def extract_experiment_metric(path: str) -> Dict[str, float]:
    output: Dict[str, float] = {}

    # load global execution summary
    filenames: List[str] = glob.glob(os.path.join(path, 'openai-infqps-*.json'))
    if len(filenames) != 1:
        raise ValueError(f'More than one output result file or none {filenames} for path {path}')
    with open(filenames[0]) as metrics_file:
        execution_summary: dict = json.load(metrics_file)

    # load metric extracted values
    metric_values: Dict[str, List[float]] = {}
    with open(os.path.join(path, 'metrics_engine_server_0.csv'), newline='') as file:
        reader = csv.reader(file, delimiter=',')
        header = next(reader)
        header_indexes: Dict[int, str] = {}
        for index, item in enumerate(header):
            metric_values[item] = []
            header_indexes[index] = item
        for row in reader:
            for index, item in enumerate(row):
                metric_values[header_indexes[index]].append(float(item))

    # check if preemption
    if np.any(np.asarray(metric_values['num_preemptions_total']) > 0):
        raise ValueError

    # compute mean batch size
    output['batch_size'] = float(np.mean(np.asarray(metric_values['num_requests_running'])))

    # compute throughput
    output['throughput'] = float(execution_summary['total_token_throughput'])

    # compute latency
    output['latency'] = float(execution_summary['mean_itl_ms'])

    # compute max KV cache utilization
    output['kv_cache'] = np.max(np.asarray(metric_values['gpu_cache_usage_perc'])) * 100

    return output


def extract_results(path: str, model: str) -> List[Dict[str, float]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['model', 'set_batch_size']
    results = []
    rerun_errors: List[str] = []
    unknown_errors: int = 0
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            if folder == '_':
                continue
            batch_size: int = int(folder.split('_')[1])
            try:
                metrics = extract_experiment_metric(os.path.join(path, folder))
            except Exception as e:
                error_message: str = f'WARNING! Error while extracting results -> {os.path.join(path, folder)}. '
                with open(os.path.join(path, folder, 'server_err_0.log')) as f:
                    error_log: str = f.read()
                    if 'ValueError: No available memory for the cache blocks' in error_log:
                        error_message += 'Not enough memory'
                    elif 'torch.cuda.OutOfMemoryError: CUDA out of memory' in error_log:
                        error_message += 'Not enough memory'
                    elif 'ValueError: The model\'s max seq len (4096) is larger than the maximum number of tokens that can be stored in KV cache' in error_log:
                        error_message += 'Not enough memory'
                    elif 'RuntimeError: CUDA error: uncorrectable ECC error encountered' in error_log:
                        error_message += 'ECC error'
                        rerun_errors.append(os.path.join(path, folder))
                    elif 'RuntimeError: CUDA error: an illegal memory access was encountered' in error_log:
                        error_message += 'Memory access error'
                        rerun_errors.append(os.path.join(path, folder))
                    elif '[Errno 98] error while attempting to bind on address' in error_log:
                        error_message += 'Port bind error'
                        rerun_errors.append(os.path.join(path, folder))
                    else:
                        print(e)
                        error_message += 'Unknown error'
                        unknown_errors += 1
                # print(error_message)
                metrics = {}
            metrics['model'] = model
            metrics['set_batch_size'] = batch_size  # in case of error
            _id = create_id(metrics, id_metrics)
            if _id in collected_ids:
                raise ValueError('Repeated results')
            collected_ids.add(_id)
            results.append(metrics)
    print(f'Unknown extraction errors: {unknown_errors}. Should be zero.')
    print(f'Rerun errors: {len(rerun_errors)}. Should be zero. Full list: {rerun_errors}')
    return results


def __prepare_lines(results: List[Dict[str, float]], x_axis: str, y_axis: str, selection: str) -> List[
    Tuple[str, List[int], List[float]]]:
    output_tmp: Dict[str, Tuple[List[int], List[float]]] = {}
    for item in results:
        selection_id = item[selection]
        if selection_id not in output_tmp:
            output_tmp[selection_id] = ([], [])
        if x_axis not in item:
            output_tmp[selection_id][0].append(None)
        else:
            output_tmp[selection_id][0].append(item[x_axis])
        if y_axis not in item:
            output_tmp[selection_id][1].append(None)
        else:
            output_tmp[selection_id][1].append(item[y_axis])
    output: List[Tuple[str, List[int], List[float]]] = []
    for key, (x_values, y_values) in output_tmp.items():
        x_line = [x_value for index, x_value in enumerate(x_values) if
                  x_value is not None and y_values[index] is not None]
        y_line = [y_value for index, y_value in enumerate(y_values) if
                  y_value is not None and x_values[index] is not None]
        y_line = [y_value for _, y_value in sorted(zip(x_line, y_line))]
        x_line.sort()
        output.append(
            (
                key,
                x_line,
                y_line
            )
        )
    output = [value for _, value in sorted(zip([value[0] for value in output], output))]

    return output


def plot_throughput_latency(
        all_model_results: List[Dict[str, float]],
        path: str,
        plot_type: str
) -> None:
    # Consistent plot settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 17,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'lines.linewidth': 2.0,
        'mathtext.default': 'regular',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.4,
        'figure.figsize': (14, 5)  # Adjusted for balanced horizontal layout
    })

    # Prepare results for throughput and latency
    cols = [
        __prepare_lines(all_model_results, 'batch_size', 'throughput', 'model'),
        __prepare_lines(all_model_results, 'batch_size', 'latency', 'model')
    ]

    # Create subplots for horizontal or vertical layout
    if plot_type == 'horizontal':
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, constrained_layout=True)
    elif plot_type == 'vertical':
        fig, axs = plt.subplots(2, 1, figsize=(7, 9), sharex=True, constrained_layout=True)
    else:
        raise ValueError("Invalid plot type. Choose 'horizontal' or 'vertical'.")

    # Ensure consistent plot background and spine formatting
    for ax in axs:
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.0)
        ax.tick_params(axis='both', colors='black', direction='in')

    # Plot configuration
    col_labels = ['Throughput (tokens/s)', 'Latency (ms)']
    markers = ['o', 's', 'D', '^']
    colors = ['#0072B2', '#E69F00', '#009E73', '#D55E00']

    for idx, (col_results, ax) in enumerate(zip(cols, axs)):
        desired_order = ['opt-1.3b', 'opt-2.7b', 'llama-2-7b', 'llama-2-13b']
        col_results = sorted(col_results, key=lambda x: desired_order.index(x[0]))

        for (model, x_line, y_line), marker, color in zip(col_results, markers, colors):
            line = ax.plot(
                x_line, y_line,
                marker=marker,
                markersize=6,
                color=color,
                label=model,
                linewidth=2.5,  # Increased line width
            )[0]

            # Extend dotted line if needed
            if x_line[-1] < 300:
                x_final_point = x_line[-1] + (x_line[-1] - x_line[-2]) / 2
                y_final_point = y_line[-1] + (y_line[-1] - y_line[-2]) / 2
                ax.plot(
                    x_line[-1:] + [x_final_point],
                    y_line[-1:] + [y_final_point],
                    marker='x',
                    markevery=[1],
                    linestyle='dotted',
                    color=line.get_color()
                )

        # Axis labels and limits
        ax.set_ylabel(col_labels[idx], labelpad=10, fontsize=24)
        ax.set_xlabel('Average Batch Size (reqs)', fontsize=24, labelpad=10)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)

    legend_handles = [
        Line2D([0], [0], color=colors[i], marker=markers[i], markersize=10, linestyle='-', linewidth=3, label=model, markeredgewidth=1, markeredgecolor='black')
        for i, (model, x_line, y_line) in enumerate(cols[0])
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=2, frameon=False, fontsize=22, bbox_to_anchor=(0.57, 1.25))

    # Save as high-resolution PDF
    output_path = os.path.join(path, 'background_throughput_latency.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=400)
    plt.show()


def plot_kv_cache(
        all_model_results: List[Dict[str, float]],
        path: str
) -> None:
    # Consistent plot settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 2.0,
        'mathtext.default': 'regular',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.4,
        'figure.figsize': (7, 5)  # Consistent size for single plot
    })

    # Prepare data
    all_model_results = __prepare_lines(
        all_model_results,
        'kv_cache',
        'throughput',
        'model'
    )

    desired_order = ['opt-1.3b', 'opt-2.7b', 'llama-2-7b', 'llama-2-13b']
    sorted_results = sorted(all_model_results, key=lambda x: desired_order.index(x[0]))

    # Create plot
    fig, ax = plt.subplots(figsize=(4.5, 2.5), constrained_layout=True, facecolor='white')
    ax.set_facecolor('white')

    # Ensure axis visibility
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)
    ax.tick_params(axis='both', colors='black', direction='in')

    # Define markers and colors
    markers = ['o', 's', 'D', '^']
    colors = ['#0072B2', '#E69F00', '#009E73', '#D55E00']

    # Plot each model's results
    for (model, x_line, y_line), marker, color in zip(sorted_results, markers, colors):
        ax.plot(
            x_line,
            y_line,
            marker=marker,
            markersize=5,
            color=color,
            label=model,
            linewidth=1.5
        )

    # Axis labels and ticks
    ax.set_xlabel('KV Cache Maximum Usage (%)', fontsize=13, labelpad=10)
    ax.set_ylabel('Throughput (tokens/s)', fontsize=13, labelpad=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    # ax.legend(loc='upper right', frameon=False, bbox_to_anchor=(1, 0.95))

    legend_handles = [
        Line2D([0], [0], color=colors[i], marker=markers[i], markersize=8, linestyle='-', linewidth=3, label=model, markeredgewidth=1, markeredgecolor='black')
        for i, (model, x_line, y_line) in enumerate(sorted_results)
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.57, 1.22))

    # Save as high-resolution PDF
    output_path = os.path.join(path, 'background_kv_cache_plot.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=400)
    plt.show()


def main():
    model_results: List[Dict[str, float]] = []
    for model in ['opt-1.3b', 'opt-2.7b', 'llama-2-7b', 'llama-2-13b']:
        model_results += extract_results(model, model)

    plot_throughput_latency(
        model_results,
        '.',
        'horizontal'
    )

    plot_kv_cache(
        model_results,
        '.'
    )


if __name__ == '__main__':
    main()
