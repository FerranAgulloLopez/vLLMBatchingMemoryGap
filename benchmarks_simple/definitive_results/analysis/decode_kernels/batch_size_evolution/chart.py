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
    plt.style.use('ggplot')

    if all_model_results is not None:
        import pickle
        with open('/home/ferran/Downloads/decode_kernels_batch_size_evolution', 'wb') as file:
            pickle.dump(all_model_results, file)
    else:
        import pickle
        with open('/home/ferran/Downloads/decode_kernels_batch_size_evolution', 'rb') as file:
            all_model_results = pickle.load(file)

    nrows = 2
    ncols = 1
    fig = plt.figure(figsize=(8, 6))
    plt.tight_layout()
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    fig.subplots_adjust(wspace=0.05, hspace=0.25)

    metrics = [
        'Compute Warps in Flight [Throughput %]',
        'DRAM Read Bandwidth [Throughput %]'
    ]
    metrics_labels = [
        'Compute Warps in Flight',
        'DRAM Read Throughput'
    ]
    metrics_linestyles = [
        'solid',
        'dashed'
    ]
    metric_colors = []

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    # top subplot
    axs_top = fig.add_subplot(gs[0, :])
    for item in all_model_results:
        for metric in metrics:
            item[f'{metric} Average'] = np.mean(item['gpu_metrics_values_decode'][metric])
            item[f'{metric} Max'] = np.max(item['gpu_metrics_values_decode'][metric])
    for metric_index, metric in enumerate(metrics):
        axs_top.add_line(Line2D([], [], color='none', label=metrics_labels[metric_index]))
        for label, metric_specific in [('max', f'{metric} Max'), ('average', f'{metric} Average')]:
            _, x_line, y_line = __prepare_lines(
                all_model_results,
                'batch_size',
                metric_specific,
                'model'
            )[0]
            line = axs_top.plot(
                x_line,
                y_line,
                marker='o',
                linestyle=metrics_linestyles[metric_index],
                label=label
            )[0]
            if label == 'average':
                metric_colors.append(line.get_color())
    axs_top.set_ylabel('Usage proportion (%)', fontsize=10)
    axs_top.set_xlabel('Average batch size (reqs)', fontsize=10)

    leg = axs_top.legend(loc='center right', fontsize=10)
    for item, label in zip(leg.legend_handles, leg.texts):
        if label._text in metrics_labels:
            width = item.get_window_extent(fig.canvas.get_renderer()).width
            label.set_ha('left')
            label.set_position((-2 * width, 0))

    # subplots bottom
    results_batch_size_1 = None
    results_batch_size_512 = None
    for results in all_model_results:
        if results['batch_size'] == 1:
            results_batch_size_1 = results
        elif results['batch_size'] == 512:
            results_batch_size_512 = results

    def max_pool1d_strided(input_array, kernel_len, stride):  # extracted from chatGPT4
        input_len = input_array.shape[0]
        output_len = (input_len - kernel_len) // stride + 1
        output = np.zeros(output_len)

        for i in range(output_len):
            region = input_array[i * stride:i * stride + kernel_len]
            output[i] = np.max(region)

        return output

    # subplot bottom left
    axs_bottom_left = fig.add_subplot(gs[1, 0])
    start_index = 0
    end_index = 125
    for metric_index, metric in enumerate(metrics):
        y_line = results_batch_size_1['gpu_metrics_values_decode'][metric][start_index:end_index]
        y_line = max_pool1d_strided(y_line, 5, stride=4)
        x_line = np.arange(np.shape(y_line)[0])
        line = axs_bottom_left.plot(
            x_line,
            y_line,
            marker='',
            linestyle=metrics_linestyles[metric_index],
            color=metric_colors[metric_index],
            label=metrics_labels[metric_index]
        )[0]
    axs_bottom_left.set_xlabel('Time (batch size = 1)', fontsize=10)
    axs_bottom_left.set_ylabel('Usage proportion (%)', fontsize=10)
    axs_bottom_left.xaxis.set_ticklabels([])
    axs_bottom_left.yaxis.set_ticks([0, 50, 100])
    leg = axs_bottom_left.legend(loc='center right', fontsize=10)

    # subplot bottom right
    axs_bottom_left = fig.add_subplot(gs[1, 1])
    start_index = 30
    end_index = 1080
    for metric_index, metric in enumerate(metrics):
        y_line = results_batch_size_512['gpu_metrics_values_decode'][metric][start_index:end_index]
        y_line = max_pool1d_strided(y_line, 5, stride=4)
        x_line = np.arange(np.shape(y_line)[0])
        line = axs_bottom_left.plot(
            x_line,
            y_line,
            marker='',
            linestyle=metrics_linestyles[metric_index],
            color=metric_colors[metric_index],
            label=metrics_labels[metric_index]
        )[0]
    axs_bottom_left.set_xlabel('Time (batch size = 512)', fontsize=10)
    axs_bottom_left.yaxis.set_ticklabels([])
    axs_bottom_left.xaxis.set_ticklabels([])
    leg = axs_bottom_left.legend(loc='center right', fontsize=10)

    plt.savefig(os.path.join(path, f'decode_kernels_batch_size_evolution'), bbox_inches='tight')


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
