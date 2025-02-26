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
from matplotlib.lines import Line2D

# FOR RUNNING SCRIPT, EXPORT FIRST nsys-rep REPORT WITH "nsys export --separate-strings yes --type sqlite .nsys-rep"
# MAYBE THE REPORT IS MISSING BECAUSE OF REPO SIZE CONSTRAINTS. IN THAT CASE, RERUN THE EXPS WITH THE PROVIDED CONFIG


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


def nsight_extract_prefill_first_decode_step_start_end_time(path: str):
    try:
        conn = sqlite3.connect(path)
        query = f"""
                SELECT text, start, end
                FROM NVTX_EVENTS
                WHERE eventType == 60 AND text == 'DecodeStep0'
            """ # type 34 correspond to mark events
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        raise e

    if df.empty:
        raise Exception('Nothing found in SQLite database')

    df = df.to_numpy()

    return df[0, 1], df[0, 2]


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

    # find decode first step start/end
    decode_first_step_start, decode_first_step_end = nsight_extract_prefill_first_decode_step_start_end_time(nsight_sqlite_file)

    # extract first decode step GPU metrics
    gpu_metrics = [
        'SMs Active [Throughput %]',
        'Compute Warps in Flight [Throughput %]',
        'Unallocated Warps in Active SMs [Throughput %]',
        'DRAM Read Bandwidth [Throughput %]',
        'DRAM Write Bandwidth [Throughput %]'
    ]
    gpu_metrics_values_decode_raw = nsight_extract_gpu_metric_arrays(nsight_sqlite_file, gpu_metrics, decode_first_step_start, decode_first_step_end)
    output['gpu_metrics_values_first_decode_step'] = gpu_metrics_values_decode_raw

    '''fig, ax = plt.subplots()
    meh = gpu_metrics_values_decode_raw['DRAM Read Bandwidth [Throughput %]']['x'], gpu_metrics_values_decode_raw['DRAM Read Bandwidth [Throughput %]']['y']
    ax.plot(meh[0], meh[1], label='DRAM read')
    meh = gpu_metrics_values_decode_raw['Compute Warps in Flight [Throughput %]']['x'], gpu_metrics_values_decode_raw['Compute Warps in Flight [Throughput %]']['y']
    ax.plot(meh[0], meh[1], label='SM UnOccupancy')
    ax.grid()
    ax.legend()
    fig.savefig("/home/ferran/Downloads/first_decode_step.png")'''

    # extract kernel executions
    output['kernels'] = {}
    kernels_start_end = nsight_extract_kernels_start_end_values_timewise(nsight_sqlite_file, decode_first_step_start, decode_first_step_end)
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


def __group_kernels(results: Dict[str, Any]) -> Dict[str, List[Tuple[float, float]]]:
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
    grouped_kernels: Dict[str, List[Tuple[float, float]]] = {}
    for kernel_label, kernel_starts_ends in results['kernels'].items():
        new_label = kernel_grouping[kernel_label]
        if new_label not in grouped_kernels:
            grouped_kernels[new_label] = []
        grouped_kernels[new_label] += kernel_starts_ends
    return grouped_kernels


def __cut_kernels_by_time(grouped_kernels: Dict[str, List[Tuple[float, float]]], start_time, end_time):
    # filter executions by time
    for kernel_label, kernel_starts_ends in grouped_kernels.items():
        filtered_executions = []
        for kernel_start, kernel_end in kernel_starts_ends:
            if kernel_start > start_time and kernel_end < end_time:
                filtered_executions.append((kernel_start, kernel_end))
        grouped_kernels[kernel_label] = filtered_executions
    return grouped_kernels


def plot_decode_timewise(
        all_model_results: List[Dict[str, Any]],
        path: str
) -> None:
    if all_model_results is not None:
        import pickle
        with open('/gpfs/scratch/bsc98/bsc098949/vLLMServingPlateau/decode_kernels_time_wise', 'wb') as file:
            pickle.dump(all_model_results, file)
    else:
        import pickle
        with open('/gpfs/scratch/bsc98/bsc098949/vLLMServingPlateau/decode_kernels_time_wise', 'rb') as file:
            all_model_results = pickle.load(file)

    # extract specific results for batch sizes 1 and 160
    results_1 = None
    results_160 = None
    for model_results in all_model_results:
        if model_results['batch_size'] == 1:
            results_1 = model_results
        elif model_results['batch_size'] == 160:
            results_160 = model_results
    assert results_1 is not None
    assert results_160 is not None

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
    # define figure
    # fig = plt.figure(figsize=(8, 6), constrained_layout=True, facecolor='white')
    # gs = GridSpec(nrows=2, ncols=2, figure=fig, height_ratios=[1, 0.5], hspace=0.05, wspace=0)

    fig = plt.figure(figsize=(8, 6), constrained_layout=True, facecolor='white')
    gs = GridSpec(2, 2, height_ratios=[1, 0.75], wspace=0.1, hspace=0.3)

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

    # define start and end indexes of plots inside the first decode step
    start_index_1 = 50
    end_index_1 = 65
    start_index_160 = 50
    end_index_160 = 80

    # top plots
    for index_results, results, start_index, end_index in [(0, results_1, start_index_1, end_index_1), (1, results_160, start_index_160, end_index_160)]:
        axs = fig.add_subplot(gs[0, index_results])
        for metric_index, metric in enumerate(metrics):
            y_line = results['gpu_metrics_values_first_decode_step'][metric]['y'][start_index:end_index]
            x_line = results['gpu_metrics_values_first_decode_step'][metric]['x'][start_index:end_index]
            line = axs.plot(
                x_line,
                y_line,
                marker='',
                label=metrics_labels[metric_index],
                linewidth=2
            )[0]
        axs.yaxis.set_ticks([0, 20, 40, 60, 80, 100])
        axs.xaxis.set_ticklabels([])
        axs.get_xaxis().set_visible(False)

        axs.set_ylim(bottom=0)
        axs.set_xlim((x_line[0], x_line[-1]))

        if index_results == 0:
            axs.set_ylabel('Usage proportion (%)', fontsize=10)
            axs.legend(loc='center', bbox_to_anchor=(1, 1.2), fontsize=10)  # TODO refactor
        else:
            axs.yaxis.set_ticklabels([])

    # bottom plots
    for index_results, results, start_index, end_index in [(0, results_1, start_index_1, end_index_1), (1, results_160, start_index_160, end_index_160)]:
        axs = fig.add_subplot(gs[1, index_results])

        # define start end times in seconds (not in indexes)
        start_time = next(iter(results['gpu_metrics_values_first_decode_step'].values()))['x'][start_index]
        end_time = next(iter(results['gpu_metrics_values_first_decode_step'].values()))['x'][end_index]

        # group kernels by generic names and filter by time
        grouped_kernels = __group_kernels(results)
        grouped_kernels = __cut_kernels_by_time(grouped_kernels, start_time, end_time)

        # plot
        legend_patches = {}
        important_kernel_types = ['Matrix multiplication', 'Attention mechanism']
        colors = ['#348ABD', '#E24A33']
        for level_index, kernel_label in enumerate(important_kernel_types):
            legend_patches[kernel_label] = patches.Patch(color=colors[level_index], label=kernel_label, linewidth=0.1)
            
            for kernel_start, kernel_end in grouped_kernels[kernel_label]:
                rect = patches.FancyBboxPatch(
                    (kernel_start, len(important_kernel_types) - level_index + 1 - 0.3),
                    kernel_end - kernel_start,
                    0.6,
                    boxstyle='round,pad=0.1',
                    edgecolor='black',
                    facecolor=colors[level_index],
                    linewidth=0
                )
                axs.add_patch(rect)
        

        if index_results == 0:
            axs.set_xlabel('Time - Batch size 1', fontsize=10)
            axs.set_ylabel('Kernel timeline', fontsize=10)
        
        if index_results == 1:
            axs.set_xlabel('Time - Batch size 160', fontsize=10)

        axs.set_ylim((1, 4))
        axs.set_xlim((start_time, end_time))
        axs.get_yaxis().set_visible(True)
        axs.xaxis.set_ticklabels([])
        axs.yaxis.set_ticklabels([])

    # Add legend below both bottom subplots
    fig.legend(handles=list(legend_patches.values()), loc='upper center', ncol=2, fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.47))


    output_path = os.path.join(path, 'decode_kernels_timewise.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=400)
    

def main():
    '''model_results: List[Dict[str, Any]] = []
    model: str = 'llama-2-7b'
    model_results += extract_results(model, model)

    plot_decode_timewise(
        model_results,
        '.'
    )'''

    plot_decode_timewise(
        None,
        '.'
    )


if __name__ == '__main__':
    main()
