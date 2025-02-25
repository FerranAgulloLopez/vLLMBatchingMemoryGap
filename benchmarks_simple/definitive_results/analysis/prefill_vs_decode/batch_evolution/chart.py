import os
import re
import glob
from typing import List, Tuple, Dict, Set, Any
import matplotlib.pyplot as plt


def extract_experiment_metric(path: str) -> Dict[str, float]:
    output: Dict[str, float] = {}

    # load log output
    filenames: List[str] = glob.glob(os.path.join(path, 'log_*.out'))
    if len(filenames) != 1:
        raise ValueError(f'More than one output result file or none {filenames} for path {path}')
    with open(filenames[0]) as file:
        log_out: str = file.read()

    # compute prefill time
    pattern = f'Prefill elapsed time: [+-]?([0-9]+([.][0-9]*)?|[.][0-9]+) seconds'
    found = re.search(pattern, log_out)
    if found is None:
        raise ValueError(f'Metric pattern not found on result log')
    output['prefill'] = float(found.group(1))

    # compute decode time
    pattern = f'Decode elapsed time: [+-]?([0-9]+([.][0-9]*)?|[.][0-9]+) seconds'
    found = re.search(pattern, log_out)
    if found is None:
        raise ValueError(f'Metric pattern not found on result log')
    output['decode'] = float(found.group(1))

    return output


def extract_results(path: str, model: str) -> List[Dict[str, Any]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['model', 'input_length', 'output_length', 'batch_size']
    results = []
    rerun_errors: List[str] = []
    unknown_errors: int = 0
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            if folder == '_':
                continue
            input_length: int = int(folder.split('_')[1])
            output_length: int = int(folder.split('_')[2])
            batch_size: int = int(folder.split('_')[3])
            try:
                metrics = extract_experiment_metric(os.path.join(path, folder))
            except Exception as e:
                error_message: str = f'WARNING! Error while extracting results -> {os.path.join(path, folder)}. '
                error_message += 'Unknown error'
                unknown_errors += 1
                # print(error_message)
                metrics = {}
            metrics['model'] = model
            metrics['input_length'] = input_length
            metrics['output_length'] = output_length
            metrics['batch_size'] = batch_size
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


def plot_time_evolution(
        all_model_results: List[Dict[str, float]],
        path: str
) -> None:
    plt.style.use('ggplot')

    _, prefill_x, prefill_y = __prepare_lines(
        all_model_results,
        'batch_size',
        'prefill',
        'model'
    )[0]

    _, decode_x, decode_y = __prepare_lines(
        all_model_results,
        'batch_size',
        'decode',
        'model'
    )[0]

    # Calculate global slowdown
    init_time = prefill_y[0] + decode_y[0]
    slowdown_y = [(prefill_y[i] + decode_y[i]) / init_time for i in range(len(prefill_x))]

    # Set plot style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2.5,
        'mathtext.default': 'regular',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'figure.figsize': (8, 6)
    })

    # Prepare plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_facecolor('white')
    colors = ['#0072B2', '#E69F00', '#009E73', '#D55E00']

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)  # Make the axis lines more prominent
    ax.tick_params(axis='both', colors='black', direction='in')

    prefill_x = [str(item) for item in prefill_x]
    bar_width = 0.6
    ax.bar(prefill_x, prefill_y, label='Prefill Time', color='#0072B2', edgecolor='black', width=bar_width)
    patches = ax.bar(prefill_x, decode_y, bottom=prefill_y, label='Decode Time', color='#E69F00', edgecolor='black', width=bar_width)

    for i in range(len(prefill_x)):
        total_time = prefill_y[i] + decode_y[i]
        percentage = decode_y[i] / total_time * 100
        # ax.text(i, total_time / 2, f'{percentage:.1f}%', ha='center', va='center', fontsize=11, color='black')

    ax2 = ax.twinx()
    for spine in ax2.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)
    ax2.plot(prefill_x, slowdown_y, marker='o', color='#D55E00', label='Global Slowdown', linewidth=2.5)
    ax2.set_ylabel('Global Slowdown', fontsize=14, color='black')
    ax2.tick_params(axis='y', colors='black', width=1.5)
    ax2.grid(False)
    # ax.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.3)
    # ax2.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.3)

    ax.set_xlabel('Average Batch Size (reqs)', fontsize=14, labelpad=10, color='black')
    ax.set_ylabel('Time (s)', fontsize=14, labelpad=10, color='black')
    
    ax.legend(loc='upper left', frameon=False)
    ax2.legend(loc='center left', frameon=False)

    # Save plot as high-resolution PDF
    output_path = 'prefill_decode_plot.pdf'
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=400)
    plt.show()

    '''col_labels = ['Throughput (toks/s)', 'Time Between Tokens (ms)']
    for model, x_line, y_line in col_results:
        line = axs[index_x].plot(
            x_line,
            y_line,
            marker='o',
            label=model
        )[0]
    axs[index_x].set_ylabel(col_labels[index_x], fontsize=10)
    axs[index_x].set_xlabel('Average batch size (reqs)', fontsize=10)

    handles, labels = axs[index_x].get_legend_handles_labels()
    handles = [handles[2], handles[3], handles[1], handles[0]]
    labels = [labels[2], labels[3], labels[1], labels[0]]
    axs[index_x].legend(handles, labels, loc='center right', fontsize=10)'''


def plot_kv_cache(
        all_model_results: List[Dict[str, Any]],
        path: str
) -> None:
    plt.style.use('ggplot')

    all_model_results = __prepare_lines(
        all_model_results,
        'kv_cache',
        'throughput',
        'model'
    )

    nrows = 1
    ncols = 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4), sharex=True)
    fig.tight_layout()
    # fig.subplots_adjust(wspace=0)

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    for model, x_line, y_line in all_model_results:
        line = axs.plot(
            x_line,
            y_line,
            marker='o',
            label=model
        )[0]
    axs.set_ylabel('Throughput (toks/s)', fontsize=10)
    axs.set_xlabel('KV cache maximum usage (%)', fontsize=10)

    handles, labels = axs.get_legend_handles_labels()
    handles = [handles[2], handles[3], handles[1], handles[0]]
    labels = [labels[2], labels[3], labels[1], labels[0]]
    axs.legend(handles, labels, loc='center right', fontsize=10)

    plt.savefig(os.path.join(path, f'background_kv_cache'), bbox_inches='tight')


def main():
    model_results: List[Dict[str, Any]] = []
    model = 'opt-2.7b'
    model_results += extract_results(model, model)

    plot_time_evolution(
        model_results,
        '.'
    )


if __name__ == '__main__':
    main()
