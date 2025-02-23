import os
import re
import glob
import sqlite3
import pandas as pd
from typing import List, Tuple, Dict, Set, Any
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# FOR RUNNING SCRIPT, EXPORT FIRST nsys-rep REPORT WITH "nsys export --separate-strings yes --type sqlite .nsys-rep"
# MAYBE THE REPORT IS MISSING BECAUSE OF REPO SIZE CONSTRAINTS. IN THAT CASE, RERUN THE EXPS WITH THE PROVIDED CONFIG


def cut_metric_values_timewise(metric_x: np.ndarray, metric_y: np.ndarray, init_cut: float, end_cut: float):
    assert init_cut < end_cut
    assert metric_x[0] < init_cut < metric_x[-1]
    assert metric_x[0] < end_cut < metric_x[-1]
    init_index: int = None
    end_index: int = None
    iter_index: int = 0
    while (init_index is None or end_index is None) and iter_index < len(metric_x):
        if init_index is None and init_cut < metric_x[iter_index]:
            init_index = iter_index
        if end_index is None and end_cut < metric_x[iter_index]:
            end_index = iter_index
        iter_index += 1
    assert init_cut < end_cut
    return metric_x[init_index:end_index], metric_y[init_index:end_index]


def nsight_extract_gpu_metric_arrays(path: str, metrics: List[str]):
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
                WHERE ({metric_names})
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


def nsight_extract_kernel_start_values(path: str, kernel_name: str):
    try:
        conn = sqlite3.connect(path)
        query = f"""
                SELECT start, end
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                JOIN StringIds ON CUPTI_ACTIVITY_KIND_KERNEL.shortName = StringIds.id
                WHERE StringIds.value == '{kernel_name}' AND streamId == 7
            """
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        raise e

    if df.empty:
        raise Exception('Nothing found in SQLite database')

    df = df.to_numpy()

    return df[:, 0]


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

    # extract complete GPU metrics
    gpu_metrics = [
        'SMs Active [Throughput %]',
        'Compute Warps in Flight [Throughput %]',
        'Unallocated Warps in Active SMs [Throughput %]',
        'DRAM Read Bandwidth [Throughput %]',
        'DRAM Write Bandwidth [Throughput %]'
    ]
    gpu_metrics_values = nsight_extract_gpu_metric_arrays(nsight_sqlite_file, gpu_metrics)

    # find prefill start -> start of flash_fwd_kernel
    kernel_start_values = nsight_extract_kernel_start_values(nsight_sqlite_file, 'reshape_and_cache_flash_kernel' if flash_attention else 'reshape_and_cache_kernel')
    prefill_start = np.min(kernel_start_values)

    # find decode start and end -> start of flash_fwd_splitkv_kernel
    kernel_start_values = nsight_extract_kernel_start_values(nsight_sqlite_file, 'flash_fwd_splitkv_kernel' if flash_attention else 'paged_attention_v1_kernel')
    decode_start = np.min(kernel_start_values)
    decode_end = np.max(kernel_start_values)

    '''fig, ax = plt.subplots()
    meh = cut_metric_values_timewise(dram_read_x, dram_read_y, prefill_start, decode_start)
    ax.plot(meh[0], np.convolve(meh[1], [0.05] * 20, 'same'), label='DRAM read')
    meh = cut_metric_values_timewise(sm_unocc_x, sm_unocc_y, prefill_start, decode_start)
    ax.plot(meh[0], np.convolve(meh[1], [0.05] * 20, 'same'), label='SM UnOccupancy')
    ax.grid()
    ax.legend()
    fig.savefig("/home/ferran/Downloads/prefill.png")

    fig, ax = plt.subplots()
    meh = cut_metric_values_timewise(dram_read_x, dram_read_y, decode_start, decode_end)
    ax.plot(meh[0], meh[1], label='DRAM read')
    meh = cut_metric_values_timewise(sm_unocc_x, sm_unocc_y, decode_start, decode_end)
    ax.plot(meh[0], meh[1], label='SM UnOccupancy')
    ax.grid()
    ax.legend()
    fig.savefig("/home/ferran/Downloads/decode.png")'''

    # extract prefill and decode GPU metric values
    gpu_metrics_values_prefill = {}
    gpu_metrics_values_decode = {}
    for gpu_metric in gpu_metrics:
        x_values = gpu_metrics_values[gpu_metric]['x']
        y_values = gpu_metrics_values[gpu_metric]['y']
        _, y_values_prefill = cut_metric_values_timewise(x_values, y_values, prefill_start, decode_start)
        _, y_values_decode = cut_metric_values_timewise(x_values, y_values, decode_start, decode_end)
        y_values_prefill = y_values_prefill[y_values_prefill != 0]
        y_values_decode = y_values_decode[y_values_decode != 0]
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
        with open('/home/ferran/Downloads/meh', 'wb') as file:
            pickle.dump(all_model_results, file)
    else:
        import pickle
        with open('/home/ferran/Downloads/meh', 'rb') as file:
            all_model_results = pickle.load(file)

    nrows = 2
    ncols = 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4), sharey=True)
    fig.tight_layout()
    # fig.subplots_adjust(wspace=0)

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

    index_x = 0

    for item in all_model_results:
        for metric in metrics:
            item[f'{metric} Average'] = np.mean(item['gpu_metrics_values_decode'][metric])
            item[f'{metric} Max'] = np.max(item['gpu_metrics_values_decode'][metric])
    for metric_index, metric in enumerate(metrics):
        axs[index_x].add_line(Line2D([], [], color='none', label=metrics_labels[metric_index]))
        for label, metric_specific in [('max', f'{metric} Max'), ('average', f'{metric} Average')]:
            _, x_line, y_line = __prepare_lines(
                all_model_results,
                'batch_size',
                metric_specific,
                'model'
            )[0]
            line = axs[index_x].plot(
                x_line,
                y_line,
                marker='o',
                linestyle=metrics_linestyles[metric_index],
                label=label
            )[0]
            if label == 'average':
                metric_colors.append(line.get_color())
    axs[index_x].set_ylabel('Usage proportion (%)', fontsize=10)
    axs[index_x].set_xlabel('Average batch size (reqs)', fontsize=10)

    leg = axs[index_x].legend(loc='center right', fontsize=10)
    for item, label in zip(leg.legend_handles, leg.texts):
        if label._text in metrics_labels:
            width = item.get_window_extent(fig.canvas.get_renderer()).width
            label.set_ha('left')
            label.set_position((-2 * width, 0))

    index_x = 1

    iter_index = 0
    while all_model_results[iter_index]['batch_size'] != 512 and iter_index < len(all_model_results):
        iter_index += 1
    assert all_model_results[iter_index]['batch_size'] == 512

    results = all_model_results[iter_index]
    start_index = 0
    end_index = 750

    def conv1d_strided(input_array, kernel, stride):  # extracted from chatGPT4
        input_len = input_array.shape[0]
        kernel_len = kernel.shape[0]
        output_len = (input_len - kernel_len) // stride + 1
        output = np.zeros(output_len)

        for i in range(output_len):
            region = input_array[i * stride:i * stride + kernel_len]
            output[i] = np.sum(region * kernel)

        return output

    def max_pool1d_strided(input_array, kernel_len, stride):  # extracted from chatGPT4
        input_len = input_array.shape[0]
        output_len = (input_len - kernel_len) // stride + 1
        output = np.zeros(output_len)

        for i in range(output_len):
            region = input_array[i * stride:i * stride + kernel_len]
            output[i] = np.max(region)

        return output

    for metric_index, metric in enumerate(metrics):
        y_line = results['gpu_metrics_values_decode'][metric][start_index:end_index]
        y_line = max_pool1d_strided(y_line, 5, stride=4)
        x_line = np.arange(np.shape(y_line)[0])
        line = axs[index_x].plot(
            x_line,
            y_line,
            marker='o',
            linestyle=metrics_linestyles[metric_index],
            color=metric_colors[metric_index],
            label=metrics_labels[metric_index]
        )[0]
    axs[index_x].set_xlabel('Time', fontsize=10)
    if nrows > 1:
        axs[index_x].set_ylabel('Usage proportion (%)', fontsize=10)
    axs[index_x].xaxis.set_ticklabels([])

    leg = axs[index_x].legend(loc='center right', fontsize=10)

    plt.savefig(os.path.join(path, f'decode_kernels_batch_size_evolution'), bbox_inches='tight')


def main():
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
