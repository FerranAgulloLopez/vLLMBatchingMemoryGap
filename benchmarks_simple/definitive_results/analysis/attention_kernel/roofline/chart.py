import os
import pickle
import re
import glob
from decimal import Decimal
from typing import List, Tuple, Dict, Set, Any
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# MAYBE THE NCU REPORT IS MISSING BECAUSE OF REPO SIZE CONSTRAINTS. IN THAT CASE, RERUN THE EXPS WITH THE PROVIDED CONFIG

def str_to_bool(string: str):
    return string.lower() in ['true', '1', 't', 'y', 'yes']

LOAD_PICKLE = str_to_bool(os.getenv('LOAD_PICKLE', False))
PICKLE_ROOT_PATH = os.getenv('PICKLE_ROOT_PATH', None)
assert PICKLE_ROOT_PATH is not None
if not LOAD_PICKLE:
    # This following package comes from the NCU installation, add the corresponding path to the PYTHONPATH env variable, or check the official documentation
    # in our specific case PYTHONPATH=PYTHONPATH:/usr/local/NVIDIA-Nsight-Compute/extras/python/
    import ncu_report


def extract_experiment_metric(path: str) -> Dict[str, Any]:
    output: Dict[str, Any] = {}

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

    # load NCU report
    nsight_compute_file: str = os.path.join(path, 'ncu_profile.ncu-rep')
    ncu_metrics = ncu_report.load_report(nsight_compute_file)
    ncu_metrics = ncu_metrics.range_by_idx(0)

    # extract NCU metrics for all profiles
    output['ncu_metrics'] = {
        'achieved_work': [],
        'achieved_traffic': [],
        'peak_work_single_precision': [],
        'peak_work_double_precision': [],
        'peak_traffic': []
    }
    for index in range(ncu_metrics.num_actions()):
        ncu_profile_metrics = ncu_metrics.action_by_idx(index)

        # extract achieve work
        output['ncu_metrics']['achieved_work'].append(
            (
                    ncu_profile_metrics.metric_by_name('smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed').as_double() +
                    ncu_profile_metrics.metric_by_name('smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed').as_double() +
                    ncu_profile_metrics.metric_by_name('derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2').as_double()
            ) * ncu_profile_metrics.metric_by_name('smsp__cycles_elapsed.avg.per_second').as_double()
        )

        # extract achieved traffic
        output['ncu_metrics']['achieved_traffic'].append(ncu_profile_metrics.metric_by_name('dram__bytes.sum.per_second').as_double())

        # extract peak work single precision
        output['ncu_metrics']['peak_work_single_precision'].append(
            ncu_profile_metrics.metric_by_name('derived__sm__sass_thread_inst_executed_op_dfma_pred_on_x2').as_double() *
            ncu_profile_metrics.metric_by_name('sm__cycles_elapsed.avg.per_second').as_double()
        )

        # extract peak work double precision
        output['ncu_metrics']['peak_work_double_precision'].append(
            ncu_profile_metrics.metric_by_name('derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2').as_double() *
            ncu_profile_metrics.metric_by_name('sm__cycles_elapsed.avg.per_second').as_double()
        )

        # extract peak traffic
        output['ncu_metrics']['peak_traffic'].append(
            ncu_profile_metrics.metric_by_name('dram__bytes.sum.peak_sustained').as_double() *
            ncu_profile_metrics.metric_by_name('dram__cycles_elapsed.avg.per_second').as_double()
        )

    # average NCU metrics for all profiles
    for metric_name, metric_values in output['ncu_metrics'].items():
        output['ncu_metrics'][metric_name] = float(np.mean(np.asarray(metric_values)))

    return output


def extract_results(path: str, model: str, kernel: str) -> List[Dict[str, Any]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['model', 'kernel', 'input_length', 'output_length', 'batch_size']
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
            metrics['kernel'] = kernel
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
    # average peak metrics for all results
    peak_work_single_precision: List[float] = []
    peak_work_double_precision: List[float] = []
    peak_traffic: List[float] = []
    for results in all_model_results:
        peak_work_single_precision.append(results['ncu_metrics']['peak_work_single_precision'])
        peak_work_double_precision.append(results['ncu_metrics']['peak_work_double_precision'])
        peak_traffic.append(results['ncu_metrics']['peak_traffic'])
    peak_work_single_precision = float(np.mean(np.asarray(peak_work_single_precision)))
    peak_work_double_precision = float(np.mean(np.asarray(peak_work_double_precision)))
    peak_traffic = float(np.mean(np.asarray(peak_traffic)))

    # filter only results of opt-1.3b model
    left_all_model_results = []
    for results in all_model_results:
        if results['model'] == 'opt-1.3b':
            left_all_model_results.append(deepcopy(results))

    # extract individual metrics
    for results in left_all_model_results:
        results['achieved_work'] = results['ncu_metrics']['achieved_work']
        results['achieved_traffic'] = results['ncu_metrics']['achieved_traffic']
    achieved_work_left = __prepare_lines(
        left_all_model_results,
        'batch_size',
        'achieved_work',
        'kernel'
    )
    achieved_traffic_left = __prepare_lines(
        left_all_model_results,
        'batch_size',
        'achieved_traffic',
        'kernel'
    )
    achieved_work_flash = achieved_work_left[0]
    achieved_work_matmul = achieved_work_left[1]
    achieved_work_xformers = achieved_work_left[2]
    achieved_traffic_flash = achieved_traffic_left[0]
    achieved_traffic_matmul = achieved_traffic_left[1]
    achieved_traffic_xformers = achieved_traffic_left[2]
    assert achieved_work_flash[0] == 'flash'
    assert achieved_work_matmul[0] == 'matmul'
    assert achieved_work_xformers[0] == 'xformers'
    assert achieved_traffic_flash[0] == 'flash'
    assert achieved_traffic_matmul[0] == 'matmul'
    assert achieved_traffic_xformers[0] == 'xformers'

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import os

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 15,  # Slightly increased for readability
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'lines.linewidth': 3.0,  # Thicker lines for clarity
        'mathtext.default': 'regular',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,  # More visible but subtle
        'figure.figsize': (7.5, 5.5)  # Slightly adjusted for better aspect ratio
    })

    # Define colors & markers
    colors = ['#009E73', '#D55E00', '#CC79A7', '#0072B2', '#E69F00']
    markers = ['o', 's', 'D', '^', '*']

    # Create figure
    l = 300
    fig, axs = plt.subplots(figsize=(6, 4), constrained_layout=True)

    # Find intersection point
    intersection_x = peak_work_single_precision / peak_traffic  # Where y_compute = y_memory
    intersection_y = peak_work_single_precision

    # Peak work single precision (roofline)
    x_line = np.arange(intersection_x, l)
    y_line_compute = [peak_work_single_precision] * np.shape(x_line)[0]
    axs.plot(x_line, y_line_compute, label='Single precision roofline', linewidth=4, color=colors[0])

    # Peak memory bandwidth
    x_line = np.arange(0.01, intersection_x)
    y_line_memory = np.asarray([x * peak_traffic for x in x_line])
    axs.plot(x_line, y_line_memory, label='Memory bandwidth', linewidth=4, color=colors[1])
    axs.set_xlim(left=x_line[0])

    # Add vertical dashed line at intersection
    axs.axvline(intersection_x, linestyle='--', color='gray', linewidth=2)

    # Modify horizontal line to be dashed before intersection
    axs.plot(np.arange(0, intersection_x), [peak_work_single_precision] * len(np.arange(0.01, intersection_x)), linestyle='--', color='gray', linewidth=2)

    # Shading memory-bound and compute-bound regions
    x_fill = np.linspace(x_line[0], intersection_x, 100)
    y_fill = np.asarray([x * peak_traffic for x in x_fill])
    axs.fill_between(x_fill, y_fill, color='lightcoral', alpha=0.3)
    axs.fill_betweenx(np.linspace(0, intersection_y, 100), intersection_x, l, color='lightgreen', alpha=0.3)

    # Achieved work and traffic for different methods
    methods = {
        'FlashAttention kernel': (achieved_work_flash, achieved_traffic_flash, colors[2], markers[0]),
        'Matmul kernel': (achieved_work_matmul, achieved_traffic_matmul, colors[3], markers[1]),
        'Xformers kernel': (achieved_work_xformers, achieved_traffic_xformers, colors[4], markers[2])
    }

    for label, (work, traffic, color, marker) in methods.items():
        x_line = [work[2][0] / traffic[2][0], work[2][-1] / traffic[2][-1]]
        y_line = [work[2][0], work[2][-1]]
        
        axs.plot(
            x_line, y_line, label=label, color=color,
            marker=marker, markersize=10, markeredgewidth=1.5, markeredgecolor='black'
        )

        if label == "FlashAttention kernel":
            axs.text(x_line[1] * 1.25, y_line[1] * 0.9, 'MAX', color='black', fontsize=12)
        elif label == "Matmul kernel":
            axs.text(x_line[1] * 1, y_line[1] * 0.6, 'MAX', color='black', fontsize=12)
        elif label == "Xformers kernel":
            axs.text(x_line[1] * 0.4, y_line[1] * 1.1, 'MAX', color='black', fontsize=12)

    # Set log scales
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0], numticks=10))
    axs.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0], numticks=10))

    # Adjust y-limits
    y_max = peak_work_single_precision * 3
    axs.set_ylim(0, y_max)

    # Label axes
    axs.set_ylabel('Performance (Flop/s)', fontsize=16)
    axs.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=16)
    axs.text(0.15, peak_work_single_precision * 0.0008, 'Memory-bound', color='black', fontsize=11, fontweight='bold')
    axs.text(intersection_x * 1.15, peak_work_single_precision * 0.0008, 'Compute-bound', color='black', fontsize=11, fontweight='bold')

    legend_handles = [
        Line2D([0], [0], color=color, marker=marker, markersize=10, linestyle='-', linewidth=3, label=label, markeredgewidth=1.5, markeredgecolor='black')
        for label, (_, _, color, marker) in methods.items()
    ]
    legend_handles.append(Line2D([0], [0], color=colors[0], lw=4, linestyle='-', label='Single precision roofline'))
    legend_handles.append(Line2D([0], [0], color=colors[1], lw=4, linestyle='-', label='Memory bandwidth'))

    fig.legend(handles=legend_handles, loc='upper center', ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.57, 1.2))

    # Save as high-resolution vector graphic
    output_path = os.path.join(path, 'attention_kernel_roofline.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=400)

    plt.show()


def table_models(
        all_model_results: List[Dict[str, Any]],
        path: str
) -> None:
    plt.style.use('ggplot')

    # prepare data

    # average peak metrics for all results
    peak_work_single_precision: List[float] = []
    peak_work_double_precision: List[float] = []
    peak_traffic: List[float] = []
    for results in all_model_results:
        peak_work_single_precision.append(results['ncu_metrics']['peak_work_single_precision'])
        peak_work_double_precision.append(results['ncu_metrics']['peak_work_double_precision'])
        peak_traffic.append(results['ncu_metrics']['peak_traffic'])
    peak_work_single_precision = float(np.mean(np.asarray(peak_work_single_precision)))
    peak_work_double_precision = float(np.mean(np.asarray(peak_work_double_precision)))
    peak_traffic = float(np.mean(np.asarray(peak_traffic)))

    # get maximum batch size for every model
    model_max_batch_size = {}
    for results in all_model_results:
        model = results['model']
        batch_size = results['batch_size']
        if model not in model_max_batch_size:
            model_max_batch_size[model] = 0
        model_max_batch_size[model] = max(batch_size, model_max_batch_size[model])

    # filter only maximum and 1 batch sizes for every model and only xformers kernel
    max_all_model_results = []
    single_all_model_results = []
    for results in all_model_results:
        model = results['model']
        batch_size = results['batch_size']
        kernel = results['kernel']
        if kernel == 'xformers':
            if batch_size == model_max_batch_size[model]:
                max_all_model_results.append(deepcopy(results))
            elif batch_size == 1:
                single_all_model_results.append(deepcopy(results))

    # extract individual metrics
    for results in max_all_model_results:
        results['achieved_work'] = results['ncu_metrics']['achieved_work']
        results['achieved_traffic'] = results['ncu_metrics']['achieved_traffic']
    max_achieved_work_right = __prepare_lines(
        max_all_model_results,
        'batch_size',
        'achieved_work',
        'model'
    )
    max_achieved_traffic_right = __prepare_lines(
        max_all_model_results,
        'batch_size',
        'achieved_traffic',
        'model'
    )
    for results in single_all_model_results:
        results['achieved_work'] = results['ncu_metrics']['achieved_work']
        results['achieved_traffic'] = results['ncu_metrics']['achieved_traffic']

    single_achieved_work_right = __prepare_lines(
        single_all_model_results,
        'batch_size',
        'achieved_work',
        'model'
    )
    single_achieved_traffic_right = __prepare_lines(
        single_all_model_results,
        'batch_size',
        'achieved_traffic',
        'model'
    )

    # show
    def print_metric_value(value: float):
        return '%.2E' % Decimal(value)

    print('Common data')
    print('peak_work_single_precision', print_metric_value(peak_work_single_precision))
    print('peak_work_double_precision', print_metric_value(peak_work_double_precision))
    print('peak_traffic', print_metric_value(peak_traffic))
    print('Data by model')
    for index_model in range(len(max_achieved_traffic_right)):
        print('Model', single_achieved_work_right[index_model][0], max_achieved_work_right[index_model][0])
        model_label = max_achieved_traffic_right[index_model][0]
        print(model_label, single_achieved_work_right[index_model][0])
        assert model_label == max_achieved_work_right[index_model][0]
        assert model_label == single_achieved_work_right[index_model][0]
        assert model_label == single_achieved_traffic_right[index_model][0]
        max_batch_size = max_achieved_traffic_right[index_model][1]
        single_batch_size = single_achieved_traffic_right[index_model][1]
        single_model_achieved_traffic = single_achieved_traffic_right[index_model][2][0]
        single_model_achieved_work = single_achieved_work_right[index_model][2][0]
        max_model_achieved_traffic = max_achieved_traffic_right[index_model][2][0]
        max_model_achieved_work = max_achieved_work_right[index_model][2][0]
        print('model', model_label, 'batch size', single_batch_size, 'model_achieved_traffic', print_metric_value(single_model_achieved_traffic), 'model_achieved_work', print_metric_value(single_model_achieved_work))
        print('model', model_label, 'batch size', max_batch_size, 'model_achieved_traffic', print_metric_value(max_model_achieved_traffic), 'model_achieved_work', print_metric_value(max_model_achieved_work))


def main():
    model_results: List[Dict[str, Any]] = []
    if LOAD_PICKLE:
        with open(os.path.join(PICKLE_ROOT_PATH, 'attention_kernel_roofline'), 'rb') as file:
            model_results = pickle.load(file)
    else:
        model_results += extract_results('flash/opt-1.3b', 'opt-1.3b', 'flash')
        model_results += extract_results('xformers/opt-1.3b', 'opt-1.3b', 'xformers')
        model_results += extract_results('xformers/opt-2.7b', 'opt-2.7b', 'xformers')
        model_results += extract_results('xformers/llama-2-7b', 'llama-2-7b', 'xformers')
        model_results += extract_results('xformers/llama-2-13b', 'llama-2-13b', 'xformers')
        model_results += extract_results('matmul/opt-1.3b', 'opt-1.3b', 'matmul')

        with open(os.path.join(PICKLE_ROOT_PATH, 'attention_kernel_roofline'), 'wb') as file:
            pickle.dump(model_results, file)

    plot_decode_timewise(
        model_results,
        '.'
    )

    # table_models(
    #     model_results,
    #     '.'
    # )


if __name__ == '__main__':
    main()
