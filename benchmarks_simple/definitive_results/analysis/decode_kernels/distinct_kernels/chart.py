import os
import re
import glob
import pickle
import sqlite3
import pandas as pd
from typing import List, Tuple, Dict, Set, Any
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import shlex
from copy import deepcopy

# FOR RUNNING SCRIPT, EXPORT FIRST nsys-rep REPORT WITH "nsys export --separate-strings yes --type sqlite .nsys-rep"
# MAYBE THE REPORT IS MISSING BECAUSE OF REPO SIZE CONSTRAINTS. IN THAT CASE, RERUN THE EXPS WITH THE PROVIDED CONFIG


def str_to_bool(string: str):
    return string.lower() in ['true', '1', 't', 'y', 'yes']

CREATE_SQLITE_DATABASES = str_to_bool(os.getenv('CREATE_SQLITE_DATABASES', False))
LOAD_PICKLE = str_to_bool(os.getenv('LOAD_PICKLE', False))
PICKLE_ROOT_PATH = os.getenv('PICKLE_ROOT_PATH', None)
assert PICKLE_ROOT_PATH is not None


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


def nsight_extract_kernels_duration_on_window(path: str, start_time: float, end_time: float):
    try:
        conn = sqlite3.connect(path)
        query = f"""
                SELECT StringIds.value, SUM(end - start) duration
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                JOIN StringIds ON CUPTI_ACTIVITY_KIND_KERNEL.shortName = StringIds.id
                WHERE streamId == 7 AND start > {start_time} AND end < {end_time}
                GROUP BY StringIds.value
            """
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        raise e

    if df.empty:
        raise Exception('Nothing found in SQLite database')

    df = df.to_numpy()

    return df


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

    # find decode start and end
    _, _, decode_start, decode_end = nsight_extract_prefill_decode_start_end_times(nsight_sqlite_file)

    # compute decode time
    output['decode_time'] = decode_end - decode_start

    # extract kernel durations during decoding
    kernel_durations = nsight_extract_kernels_duration_on_window(nsight_sqlite_file, decode_start, decode_end)
    output['kernel_durations'] = {}
    for index in range(np.shape(kernel_durations)[0]):
        output['kernel_durations'][str(kernel_durations[index, 0])] = kernel_durations[index, 1]

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
    # group kernels
    grouping_labels = {
        'matrix_multiplication': 'Matrix multiplication',
        'attention': 'Attention mechanism',
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
    for index, model_results in enumerate(all_model_results):
        all_model_results[index]['kernel_durations_grouped'] = {}
        for kernel_label, kernel_duration in model_results['kernel_durations'].items():
            new_label = kernel_grouping[kernel_label]
            if new_label not in all_model_results[index]['kernel_durations_grouped']:
                all_model_results[index]['kernel_durations_grouped'][new_label] = 0
            all_model_results[index]['kernel_durations_grouped'][new_label] += kernel_duration
        del all_model_results[index]['kernel_durations']

    # include cpu time
    for index, model_results in enumerate(all_model_results):
        decode_time: float = model_results['decode_time']
        all_kernel_time: float = 0
        for kernel_label, kernel_duration in model_results['kernel_durations_grouped'].items():
            all_kernel_time += float(kernel_duration)
        all_model_results[index]['kernel_durations_grouped']['CPU time'] = decode_time - all_kernel_time

    # compute proportion and find important kernels
    minimum_proportion = 20
    important_kernels = set()
    filter_out = set()
    for index, model_results in enumerate(all_model_results):
        decode_time: float = model_results['decode_time']  # transform to ns
        for kernel_label, kernel_duration in model_results['kernel_durations_grouped'].items():
            kernel_duration: float = float(kernel_duration)
            kernel_proportion: float = kernel_duration / decode_time * 100
            all_model_results[index]['kernel_durations_grouped'][kernel_label] = kernel_proportion

            if kernel_proportion > minimum_proportion:
                important_kernels.add(kernel_label)
            else:
                filter_out.add(kernel_label)
    filter_out = filter_out.difference(important_kernels)
    print(f'Important kernels. Count: {len(important_kernels)}. List: {important_kernels}')

    # filter out not important kernels
    for index, model_results in enumerate(all_model_results):
        for filter_out_kernel in filter_out:
            if filter_out_kernel in model_results['kernel_durations_grouped']:
                del model_results['kernel_durations_grouped'][filter_out_kernel]

    # move important kernels outside
    for index, model_results in enumerate(all_model_results):
        for kernel_label, kernel_proportion in model_results['kernel_durations_grouped'].items():
            model_results[kernel_label] = kernel_proportion
        del model_results['kernel_durations_grouped']

    # prepare lines
    kernel_results = []
    for important_kernel in important_kernels:
        kernel_results.append(
            (
                important_kernel,
                __prepare_lines(
                    all_model_results,
                    'batch_size',
                    important_kernel,
                    'model'
                )
            )
        )

    # plot
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 13,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
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
    fig, ax = plt.subplots(layout='constrained', figsize=(6, 4))
    width = 0.15  # the width of the bars

    bottom = {}
    xaxis_labels = []
    print(kernel_results)
    for kernel_label, kernel_lines in kernel_results:
        multiplier = 0
        for (model, x_line, y_line) in kernel_lines:
            offset = width * multiplier
            # x_line = np.asarray(x_line)
            if model not in bottom:
                bottom[model] = np.zeros(len(y_line))
            y_line = np.asarray(y_line)

            rects = ax.bar(np.arange(len(x_line)) + offset, y_line, width, label=f'{model} {kernel_label}', bottom=bottom[model])
            multiplier += 1
            bottom[model] += y_line
            if len(x_line) > len(xaxis_labels):
                xaxis_labels = x_line
    ax.set_ylabel('Time proportion (%)')
    ax.set_xlabel('Average batch size (reqs)')
    ax.set_xticks(np.arange(len(xaxis_labels)), xaxis_labels)
    handles, labels = ax.get_legend_handles_labels()
    '''handles = [handles[1], handles[0], handles[2]]
    labels = [labels[1], labels[0], labels[2]]'''
    ax.legend(handles, labels, loc='upper right', fontsize=10)

    output_path = os.path.join(path, 'decode_kernels_distinct_kernels.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=400)


def extract_longer_matrix_multiplication(
        all_model_results: List[Dict[str, Any]],
        path: str
) -> None:
    plt.style.use('ggplot')

    # sum matrix multiplication kernels durations
    matrix_multiplication_kernels_durations = {
        'sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_cublas': 0,
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas': 0,
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize256x128x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas': 0,
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas': 0,
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x256x32_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas': 0,
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x64x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas': 0,
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x64x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas': 0,
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_on_kernel__5x_cublas': 0,
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x128x64_warpgroupsize1x1x1_execute_segment_k_on_kernel__5x_cublas': 0,
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x256x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas': 0,
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x128x32_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas': 0,
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize256x128x64_warpgroupsize2x1x1_execute_segment_k_on_kernel__5x_cublas': 0,
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x256x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas': 0,
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize256x128x32_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas': 0
    }
    matrix_multiplication_kernels_list = {
        'sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_cublas': [],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas': [],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize256x128x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas': [],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas': [],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x256x32_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas': [],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x64x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas': [],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x64x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas': [],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_on_kernel__5x_cublas': [],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x128x64_warpgroupsize1x1x1_execute_segment_k_on_kernel__5x_cublas': [],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x256x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas': [],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x128x32_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas': [],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize256x128x64_warpgroupsize2x1x1_execute_segment_k_on_kernel__5x_cublas': [],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x256x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas': [],
        'sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize256x128x32_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas': []
    }
    matrix_multiplication_kernels_times = matrix_multiplication_kernels_durations.copy()
    to_remove = set()
    for index, model_results in enumerate(all_model_results):
        if model_results['model'] == 'opt-1.3b':
            for kernel_label in matrix_multiplication_kernels_durations.keys():
                if kernel_label in model_results['kernel_durations']:
                    matrix_multiplication_kernels_durations[kernel_label] += model_results['kernel_durations'][kernel_label]
                    matrix_multiplication_kernels_times[kernel_label] += 1
                    matrix_multiplication_kernels_list[kernel_label] += [model_results['batch_size']]
                else:
                    to_remove.add(kernel_label)
            '''for kernel_label, kernel_duration in model_results['kernel_durations'].items():
                if kernel_label in matrix_multiplication_kernels_durations:
                    matrix_multiplication_kernels_durations[kernel_label] += kernel_duration'''
    '''for kernel_label in to_remove:
        del matrix_multiplication_kernels_durations[kernel_label]'''

    # sort
    sorted_values_durations = [(key, value) for value, key in sorted(zip(matrix_multiplication_kernels_durations.values(), matrix_multiplication_kernels_durations.keys()), reverse=True)]
    sorted_values_times = [(key, value) for value, key in sorted(zip(matrix_multiplication_kernels_times.values(), matrix_multiplication_kernels_times.keys()), reverse=True)]

    # show
    print(sorted_values_durations)
    print('----------------------')
    print(sorted_values_times)
    print('----------------------')
    print(matrix_multiplication_kernels_list)


def main():
    if CREATE_SQLITE_DATABASES:
        for model in ['opt-1.3b', 'opt-2.7b', 'llama-2-7b', 'llama-2-13b']:
            create_sqlite_databases(model)

    model_results: List[Dict[str, Any]] = []
    if LOAD_PICKLE:
        with open(os.path.join(PICKLE_ROOT_PATH, 'decode_kernels_distinct_kernels'), 'rb') as file:
            model_results = pickle.load(file)
    else:
        for model in ['opt-1.3b', 'opt-2.7b', 'llama-2-7b', 'llama-2-13b']:
            model_results += extract_results(model, model)
        with open(os.path.join(PICKLE_ROOT_PATH, 'decode_kernels_distinct_kernels'), 'wb') as file:
            pickle.dump(model_results, file)

    plot_batch_size_evolution(
        deepcopy(model_results),
        '.'
    )

    # # useful for a later on plot
    # extract_longer_matrix_multiplication(
    #     model_results,
    #     '.'
    # )


if __name__ == '__main__':
    main()
