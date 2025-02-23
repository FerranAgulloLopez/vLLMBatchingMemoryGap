import os
import re
import glob
import sqlite3
import pandas as pd
from typing import List, Tuple, Dict, Set, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
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


def nsight_extract_kernels_start_end_values_timewise(path: str, start_time: float, end_time: float):
    try:
        conn = sqlite3.connect(path)
        query = f"""
                SELECT StringIds.value, start, end
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                JOIN StringIds ON CUPTI_ACTIVITY_KIND_KERNEL.shortName = StringIds.id
                WHERE streamId == 7 AND start > {start_time} AND end < {end_time}
            """
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        raise e

    if df.empty:
        raise Exception('Nothing found in SQLite database')

    df = df.to_numpy()

    return df


def extract_experiment_metric(path: str) -> Dict[str, Any]:
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

    # find decode start and end -> start of flash_fwd_splitkv_kernel
    kernel_start_values = nsight_extract_kernel_start_values(nsight_sqlite_file, 'flash_fwd_splitkv_kernel' if flash_attention else 'paged_attention_v1_kernel')
    decode_start = np.min(kernel_start_values)
    decode_end = np.max(kernel_start_values)

    # manually extract first decode step
    decode_start = decode_start - 3427350
    decode_end = decode_start + (decode_end - decode_start) / 338 - 4569800

    '''fig, ax = plt.subplots()
    meh = cut_metric_values_timewise(gpu_metrics_values['DRAM Read Bandwidth [Throughput %]']['x'], gpu_metrics_values['DRAM Read Bandwidth [Throughput %]']['y'], decode_start, decode_end)
    ax.plot(meh[0], meh[1], label='DRAM read')
    meh = cut_metric_values_timewise(gpu_metrics_values['Compute Warps in Flight [Throughput %]']['x'], gpu_metrics_values['Compute Warps in Flight [Throughput %]']['y'], decode_start, decode_end)
    ax.plot(meh[0], meh[1], label='SM UnOccupancy')
    ax.grid()
    ax.legend()
    fig.savefig("/home/ferran/Downloads/decode.png")'''

    # extract decode GPU metric values
    gpu_metrics_values_decode = {}
    for gpu_metric in gpu_metrics:
        x_values = gpu_metrics_values[gpu_metric]['x']
        y_values = gpu_metrics_values[gpu_metric]['y']
        x_values_decode, y_values_decode = cut_metric_values_timewise(x_values, y_values, decode_start, decode_end)
        gpu_metrics_values_decode[gpu_metric] = {
            'x': x_values_decode,
            'y': y_values_decode
        }
    output['gpu_metrics_values_decode'] = gpu_metrics_values_decode

    # extract kernel executions
    output['kernels'] = {}
    kernels_start_end = nsight_extract_kernels_start_end_values_timewise(nsight_sqlite_file, decode_start, decode_end)
    for index in range(np.shape(kernels_start_end)[0]):
        kernel_name: str = str(kernels_start_end[index, 0])
        kernel_start: float = float(kernels_start_end[index, 1])
        kernel_end: float = float(kernels_start_end[index, 2])
        if kernel_name not in output['kernels']:
            output['kernels'][kernel_name] = []
        output['kernels'][kernel_name].append((kernel_start, kernel_end))

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
            if batch_size != 160:
                continue
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


def plot_decode_timewise(
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
    # fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4), sharex=True)
    fig, axs = plt.subplot_mosaic('A;B', gridspec_kw=dict(height_ratios=[1, 0.5]), sharex=True)
    # fig.tight_layout()
    # fig.subplots_adjust(hspace=0, wspace=0)

    metrics = [
        'Compute Warps in Flight [Throughput %]',
        'Unallocated Warps in Active SMs [Throughput %]',
        'DRAM Read Bandwidth [Throughput %]'
    ]
    metrics_labels = [
        'Compute Warps in Flight',
        'Unallocated Warps in Active SMs',
        'DRAM Read Throughput'
    ]

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    results = all_model_results[0]
    start_index = 50
    end_index = 80
    start_time = results['gpu_metrics_values_decode'][metrics[0]]['x'][start_index]
    end_time = results['gpu_metrics_values_decode'][metrics[0]]['x'][end_index]

    index_y = 'A'
    for metric_index, metric in enumerate(metrics):
        y_line = results['gpu_metrics_values_decode'][metric]['y'][start_index:end_index]
        # y_line = max_pool1d_strided(y_line, 5, stride=4)
        x_line = results['gpu_metrics_values_decode'][metric]['x'][start_index:end_index]
        line = axs[index_y].plot(
            x_line,
            y_line,
            marker='o',
            label=metrics_labels[metric_index]
        )[0]
    axs[index_y].set_ylabel('Usage proportion (%)', fontsize=10)
    axs[index_y].xaxis.set_ticklabels([])
    axs[index_y].legend(loc='center', bbox_to_anchor=(0.5, 0.3), fontsize=10)

    index_y = 'B'

    # group kernels
    grouping_labels = {
        'matrix_multiplication': 'matrix multiplication',
        'attention': 'attention mechanism',
        'sort': 'sort and others',
        'device': 'device',
        'scatter_gather': 'scatter and gather',
        'softmax': 'softmax',
        'reduce': 'reduce',
        'norm': 'normalization',
        'unknown': 'unknown',
        'other': 'other'
    }
    kernel_grouping = {
        'sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_cublas': grouping_labels['matrix_multiplication'],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas': grouping_labels['matrix_multiplication'],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize256x128x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas': grouping_labels['matrix_multiplication'],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas': grouping_labels['matrix_multiplication'],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x256x32_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas': grouping_labels['matrix_multiplication'],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x64x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas': grouping_labels['matrix_multiplication'],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x64x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas': grouping_labels['matrix_multiplication'],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_on_kernel__5x_cublas': grouping_labels['matrix_multiplication'],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x128x64_warpgroupsize1x1x1_execute_segment_k_on_kernel__5x_cublas': grouping_labels['matrix_multiplication'],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x256x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas': grouping_labels['matrix_multiplication'],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x128x32_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas': grouping_labels['matrix_multiplication'],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize256x128x64_warpgroupsize2x1x1_execute_segment_k_on_kernel__5x_cublas': grouping_labels['matrix_multiplication'],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x256x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas': grouping_labels['matrix_multiplication'],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize256x128x32_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas': grouping_labels['matrix_multiplication'],
        'DeviceRadixSortExclusiveSumKernel': grouping_labels['sort'],
        'DeviceRadixSortHistogramKernel': grouping_labels['sort'],
        'DeviceRadixSortOnesweepKernel': grouping_labels['sort'],
        'DeviceSegmentedRadixSortKernel': grouping_labels['sort'],
        'DeviceScanInitKernel': grouping_labels['device'],
        'DeviceScanKernel': grouping_labels['device'],
        'Kernel': grouping_labels['unknown'],
        '_scatter_gather_elementwise_kernel': grouping_labels['scatter_gather'],
        'cunn_SoftMaxForward': grouping_labels['softmax'],
        'distribution_elementwise_grid_stride_kernel': grouping_labels['scatter_gather'],
        'elementwise_kernel': grouping_labels['other'],
        'fill_index_and_segment_kernel': grouping_labels['other'],
        'flash_fwd_kernel': grouping_labels['attention'],
        'flash_fwd_splitkv_kernel': grouping_labels['attention'],
        'flash_fwd_splitkv_combine_kernel': grouping_labels['attention'],
        'indexSelectLargeIndex': grouping_labels['other'],
        'indexSelectSmallIndex': grouping_labels['other'],
        'index_elementwise_kernel': grouping_labels['other'],
        'paged_attention_v1_kernel': grouping_labels['attention'],
        'paged_attention_v2_kernel': grouping_labels['attention'],
        'paged_attention_v2_reduce_kernel': grouping_labels['attention'],
        'reduce_kernel': grouping_labels['reduce'],
        'reshape_and_cache_kernel': grouping_labels['attention'],
        'reshape_and_cache_flash_kernel': grouping_labels['attention'],
        'sort_postprocess_kernel': grouping_labels['sort'],
        'splitKreduce_kernel': grouping_labels['reduce'],
        'tensor_kernel_scan_innermost_dim': grouping_labels['other'],
        'unrolled_elementwise_kernel': grouping_labels['other'],
        'vectorized_elementwise_kernel': grouping_labels['other'],
        'vectorized_layer_norm_kernel': grouping_labels['other'],
        'fill_reverse_indices_kernel': grouping_labels['other'],
        'act_and_mul_kernel': grouping_labels['other'],
        'fused_add_rms_norm_kernel': grouping_labels['norm'],
        'rms_norm_kernel': grouping_labels['norm'],
        'rotary_embedding_kernel': grouping_labels['other']
    }
    grouped_kernels = {}
    for kernel_label, kernel_starts_ends in results['kernels'].items():
        new_label = kernel_grouping[kernel_label]
        if new_label not in grouped_kernels:
            grouped_kernels[new_label] = []
        grouped_kernels[new_label] += kernel_starts_ends

    # filter executions by time
    for kernel_label, kernel_starts_ends in grouped_kernels.items():
        filtered_executions = []
        for kernel_start, kernel_end in kernel_starts_ends:
            if kernel_start > start_time and kernel_end < end_time:
                filtered_executions.append((kernel_start, kernel_end))
        grouped_kernels[kernel_label] = filtered_executions

    # plot kernels
    legend_patches = {}
    important_kernel_types = ['matrix multiplication', 'attention mechanism']
    colors = ['#348ABD', '#E24A33']
    for level_index, kernel_label in enumerate(important_kernel_types):
        legend_patches[kernel_label] = patches.Patch(color=colors[level_index], label=kernel_label)
        for kernel_execution_index, (kernel_start, kernel_end) in enumerate(grouped_kernels[kernel_label]):
            rect = patches.FancyBboxPatch(
                (kernel_start, len(important_kernel_types) - level_index + 1 - 0.3),
                kernel_end - kernel_start,
                0.6,
                boxstyle='round,pad=0.1',
                edgecolor='black',
                facecolor=colors[level_index],
                linewidth=0
            )
            axs[index_y].add_patch(rect)

    axs[index_y].set_xlabel('Time', fontsize=10)
    axs[index_y].set_ylim((1, 4))
    axs[index_y].get_yaxis().set_visible(False)
    axs[index_y].xaxis.set_ticklabels([])
    axs[index_y].legend(handles=legend_patches.values(), loc='center', fontsize=10)

    plt.savefig(os.path.join(path, f'decode_kernels_timewise'), bbox_inches='tight')


def main():
    model_results: List[Dict[str, Any]] = []
    model: str = 'llama-2-7b'
    model_results += extract_results(model, model)

    plot_decode_timewise(
        model_results,
        '.'
    )

    '''plot_decode_timewise(
        None,
        '.'
    )'''


if __name__ == '__main__':
    main()
