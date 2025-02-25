import os
import re
import glob
import sqlite3
import shlex
import subprocess
import pandas as pd
from typing import List, Tuple, Dict, Set, Any
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import numpy as np
import pickle

# FOR RUNNING SCRIPT, EXPORT FIRST nsys-rep REPORT WITH "nsys export --separate-strings yes --type sqlite .nsys-rep"
# MAYBE THE REPORT IS MISSING BECAUSE OF REPO SIZE CONSTRAINTS. IN THAT CASE, RERUN THE EXPS WITH THE PROVIDED CONFIG


def create_sqlite_databases(path: str):
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            filenames: List[str] = glob.glob(os.path.join(path, folder, 'log_*.err'))
            if len(filenames) != 1:
                raise ValueError(f'More than one output result file or none {filenames} for path {path}')
            with open(filenames[0], 'r') as fp:
                if len(fp.readlines()) > 24:
                    raise ValueError('Some error happened', filenames[0])

            command = f'nsys export --separate-strings yes --type sqlite --output {os.path.join(path, folder)}/.nsys-rep.sqlite --force-overwrite true {os.path.join(path, folder)}/.nsys-rep'
            print(command)
            result = subprocess.run(
                shlex.split(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            print(result.stdout)
            print(result.stderr)


def nsight_extract_gpu_metric_arrays(path: str, metrics: List[str], start: float, end: float):
    try:
        conn = sqlite3.connect(path)
        metric_names = ''
        for metric in metrics:
            if metric_names == '':
                metric_names += f"metricName == '{metric}'"
            else:
                metric_names += f" OR metricName == '{metric}'"
        query = f"""
                SELECT metricName, timestamp, value
                FROM GPU_METRICS
                JOIN TARGET_INFO_GPU_METRICS USING (metricId)
                WHERE ({metric_names}) AND timestamp > {start} AND timestamp < {end}
            """
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        raise e

    if df.empty:
        raise Exception('Nothing found in SQLite database')

    metric_results = {}
    for metric in metrics:
        rows = df.loc[df['metricName'] == metric].drop(columns=['metricName']).to_numpy()
        metric_results[metric] = {
            'x': rows[:, 0],
            'y': rows[:, 1]
        }

    return metric_results


def nsight_extract_prefill_decode_start_end_times(path: str):
    try:
        conn = sqlite3.connect(path)
        query = f"""
                SELECT text, start
                FROM NVTX_EVENTS
                WHERE eventType == 34
            """ # type 34 correspond to mark events
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        raise e

    if df.empty:
        raise Exception('Nothing found in SQLite database')

    df = df.to_numpy()

    prefill_start = None
    prefill_end = None
    decode_start = None
    decode_end = None
    for index in range(np.shape(df)[0]):
        if df[index, 0] == 'PrefillStart':
            prefill_start = float(df[index, 1])
        elif df[index, 0] == 'PrefillEnd':
            prefill_end = float(df[index, 1])
        elif df[index, 0] == 'DecodingStart':
            decode_start = float(df[index, 1])
        elif df[index, 0] == 'DecodingEnd':
            decode_end = float(df[index, 1])
    assert prefill_start is not None
    assert prefill_end is not None
    assert decode_start is not None
    assert decode_end is not None

    return prefill_start, prefill_end, decode_start, decode_end


def extract_experiment_metric(path: str) -> Dict[str, float]:
    output: Dict[str, float] = {}

    nsight_sqlite_file: str = os.path.join(path, '.nsys-rep.sqlite')

    # load log output
    filenames: List[str] = glob.glob(os.path.join(path, 'log_*.out'))
    if len(filenames) != 1:
        raise ValueError(f'More than one output result file or none {filenames} for path {path}')
    with open(filenames[0]) as file:
        log_out: str = file.read()

    # load error output
    filenames: List[str] = glob.glob(os.path.join(path, 'log_*.err'))
    if len(filenames) != 1:
        raise ValueError(f'More than one output result file or none {filenames} for path {path}')
    with open(filenames[0]) as file:
        log_err: str = file.read()

    # check no preemption
    pattern = f'ValueError: All requests cannot be processed at the same time with input parameters'
    found = re.search(pattern, log_err)
    if found is not None:
        raise ValueError(f'Preemption was present')

    # check if using flash attention as backend
    flash_attention: bool = True  # opt-2.7b model not compatible with flash backend -> Cannot use FlashAttention-2 backend for head size 80
    pattern = f'Using XFormers backend'
    found = re.search(pattern, log_out)
    if found is not None:
        flash_attention = False

    # find prefill/decode start/end
    prefill_start, prefill_end, decode_start, decode_end = nsight_extract_prefill_decode_start_end_times(nsight_sqlite_file)

    # extract GPU metrics
    gpu_metrics = [
        'SMs Active [Throughput %]',
        'Compute Warps in Flight [Throughput %]',
        'Unallocated Warps in Active SMs [Throughput %]',
        'DRAM Read Bandwidth [Throughput %]',
        'DRAM Write Bandwidth [Throughput %]'
    ]
    gpu_metrics_values_prefill_raw = nsight_extract_gpu_metric_arrays(nsight_sqlite_file, gpu_metrics, prefill_start, prefill_end)
    gpu_metrics_values_decode_raw = nsight_extract_gpu_metric_arrays(nsight_sqlite_file, gpu_metrics, decode_start, decode_end)

    # extract prefill and decode GPU metric values
    gpu_metrics_values_prefill = {}
    gpu_metrics_values_decode = {}
    for gpu_metric in gpu_metrics:
        y_values_prefill = gpu_metrics_values_prefill_raw[gpu_metric]['y']
        y_values_decode = gpu_metrics_values_decode_raw[gpu_metric]['y']
        gpu_metrics_values_prefill[gpu_metric] = y_values_prefill
        gpu_metrics_values_decode[gpu_metric] = y_values_decode
    output['gpu_metrics_values_prefill'] = gpu_metrics_values_prefill
    output['gpu_metrics_values_decode'] = gpu_metrics_values_decode

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
                print(os.path.join(path, folder), e)
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


def plot_batch_size_evolution(
        all_model_results: List[Dict[str, Any]],
        path: str
) -> None:
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 13,
        'axes.titlesize': 15,
        'axes.labelsize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'lines.linewidth': 2.0,
        'mathtext.default': 'regular',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.4,
        'figure.figsize': (7, 5)  # Consistent size for single plot
    })

    # Load results if not provided
    if all_model_results is None:
        with open('/gpfs/scratch/bsc98/bsc098949/vLLMServingPlateau/decode_kernels_batch_size_evolution', 'rb') as file:
            all_model_results = pickle.load(file)
    else:
        with open('/gpfs/scratch/bsc98/bsc098949/vLLMServingPlateau/decode_kernels_batch_size_evolution', 'wb') as file:
            pickle.dump(all_model_results, file)

    metrics = [
        'Compute Warps in Flight [Throughput %]',
        'DRAM Read Bandwidth [Throughput %]'
    ]
    metrics_labels = ['Compute Warps in Flight', 'DRAM Read Throughput']
    metrics_linestyles = ['solid', 'dashed']
    metric_colors = ['#0072B2', '#E69F00']
    light_colors = ['#66B2FF', '#FFCC80']  # Lighter shades for the max parts

    # Prepare figure
    fig = plt.figure(figsize=(10, 8), constrained_layout=True, facecolor='white')
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2])

    # Top subplot
    ax_top = fig.add_subplot(gs[1, :])
    batch_sizes = [1, 32, 64, 128, 256]
    bar_width = 0.35 # 0.35
    index = np.arange(len(batch_sizes))

    averages = {metric: [] for metric in metrics}
    maxima = {metric: [] for metric in metrics}
    for item in all_model_results:
        if item['batch_size'] in batch_sizes:
            for metric in metrics:
                avg_value = np.mean(item['gpu_metrics_values_decode'][metric])
                max_value = np.max(item['gpu_metrics_values_decode'][metric])
                averages[metric].append(avg_value)
                maxima[metric].append(max_value)

    for i, metric in enumerate(metrics):
        ax_top.bar(index + i * bar_width, maxima[metric], bar_width, color=metric_colors[i], hatch='//', edgecolor='black')
        ax_top.bar(index + i * bar_width, averages[metric], bar_width,
              label=f'{metrics_labels[i]}', edgecolor='black', color=metric_colors[i])

    ax_top.set_ylabel('Usage Proportion (%)')
    ax_top.set_ylim(0, 100)  # Set y-axis limit from 0 to 100
    ax_top.set_xlabel('Average Batch Size (reqs)')
    ax_top.set_xticks(index + bar_width / 2)
    ax_top.set_xticklabels(batch_sizes)
    # ax_top.legend(loc='upper left', frameon=True)

    # Bottom subplots for batch size 1 and 512
    def plot_time_series(ax, results, batch_size, start_index, end_index):
        for metric_index, metric in enumerate(metrics):
            y_line = results['gpu_metrics_values_decode'][metric][start_index:end_index]
            y_line = max_pool1d_strided(y_line, 5, stride=4)
            x_line = np.arange(len(y_line))
            ax.plot(x_line, y_line, linestyle=metrics_linestyles[metric_index],
                    color=metric_colors[metric_index], label=metrics_labels[metric_index])
        ax.set_xlabel(f'Time (batch size = {batch_size})')
        ax.set_ylabel('Usage Proportion (%)')
        # if batch_size == 1:
        #     ax.legend(loc='upper left', frameon=False)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)

    def max_pool1d_strided(input_array, kernel_len, stride):
        output_len = (len(input_array) - kernel_len) // stride + 1
        return np.array([np.max(input_array[i * stride:i * stride + kernel_len]) for i in range(output_len)])

    results_batch_size_1 = next(r for r in all_model_results if r['batch_size'] == 1)
    results_batch_size_512 = next(r for r in all_model_results if r['batch_size'] == 512)

    ax_bottom_left = fig.add_subplot(gs[0, 0])
    plot_time_series(ax_bottom_left, results_batch_size_1, 1, start_index=0, end_index=125)

    ax_bottom_right = fig.add_subplot(gs[0, 1])
    plot_time_series(ax_bottom_right, results_batch_size_512, 512, start_index=30, end_index=1080)
    ax_bottom_right.yaxis.set_ticklabels([])

    legend_handles = [Line2D([0], [0], color=metric_colors[i], lw=7, label=metrics_labels[i]) for i in range(len(metrics))]
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(metrics), frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.95))

    # Save plot
    output_path = os.path.join(path, 'decode_kernels_batch_size_evolution.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=400)
    plt.show()



def main():
    '''for model in ['opt-1.3b', 'opt-2.7b', 'llama-2-7b', 'llama-2-13b']:
        create_sqlite_databases(model)'''

    '''model_results: List[Dict[str, Any]] = []
    for model in ['opt-1.3b']:
        model_results += extract_results(model, model)

    plot_batch_size_evolution(
        model_results,
        '.'
    )'''

    plot_batch_size_evolution(
        None,
        '.'
    )


if __name__ == '__main__':
    main()
