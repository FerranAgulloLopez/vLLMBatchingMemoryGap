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
# This following package comes from the NCU installation, add the corresponding path to the PYTHONPATH env variable, or check the official documentation
# in our specific case PYTHONPATH=PYTHONPATH:/usr/local/NVIDIA-Nsight-Compute/extras/python/
import ncu_report
from copy import deepcopy


# MAYBE THE NCU REPORT IS MISSING BECAUSE OF REPO SIZE CONSTRAINTS. IN THAT CASE, RERUN THE EXPS WITH THE PROVIDED CONFIG


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
    plt.style.use('ggplot')

    if all_model_results is not None:
        import pickle
        with open('/gpfs/scratch/bsc98/bsc098949/vLLMServingPlateau/attention_kernel_roofline', 'wb') as file:
            pickle.dump(all_model_results, file)
    else:
        import pickle
        with open('/gpfs/scratch/bsc98/bsc098949/vLLMServingPlateau/attention_kernel_roofline', 'rb') as file:
            all_model_results = pickle.load(file)

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

    # define figure
    nrows = 1
    ncols = 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, layout='constrained', figsize=(9, 6), sharey=True)

    # plot figure
    for index_x in range(ncols):
        # peak work single precision
        x_line = np.arange(0, 100)
        y_line = [peak_work_single_precision] * np.shape(x_line)[0]
        axs.plot(
            x_line,
            y_line,
            label='single precision roofline'
        )

        # peak work double precision
        x_line = np.arange(0, 100)
        y_line = [peak_work_double_precision] * np.shape(x_line)[0]
        axs.plot(
            x_line,
            y_line,
            label='double precision roofline'
        )

        # peak traffic
        x_line = np.arange(0.01, 100)
        # y_line = np.log(np.asarray([x_line[index] * peak_traffic for index in range(np.shape(x_line)[0])]))
        y_line = np.asarray([x_line[index] * peak_traffic for index in range(np.shape(x_line)[0])])
        axs.plot(
            x_line,
            y_line,
            label='memory bandwidth'
        )

        # achieved work and traffic
        x_line = [achieved_work_flash[2][index] / achieved_traffic_flash[2][index] for index in range(len(achieved_work_flash[2]))]
        y_line = achieved_work_flash[2]
        axs.plot(
            x_line,
            y_line,
            marker='o',
            label='achieved flash'
        )
        x_line = [achieved_work_matmul[2][index] / achieved_traffic_matmul[2][index] for index in range(len(achieved_work_matmul[2]))]
        y_line = achieved_work_matmul[2]
        axs.plot(
            x_line,
            y_line,
            marker='o',
            label='achieved matmul'
        )
        x_line = [achieved_work_xformers[2][index] / achieved_traffic_xformers[2][index] for index in range(len(achieved_work_xformers[2]))]
        y_line = achieved_work_xformers[2]
        axs.plot(
            x_line,
            y_line,
            marker='o',
            label='achieved xformers'
        )

        # label stuff
        axs.set_xscale('log')
        axs.set_yscale('log')
        # axs.set_xticks([0.1, 10, 100])
        import matplotlib.ticker as ticker
        axs.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0], numticks=10))
        axs.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0], numticks=10))
        axs.set_ylabel('Performance (Flop/s)')
        axs.set_xlabel('Arithmetic Intensity (FLOP/byte)')
        # axs.set_ylim(0, 0.2e13)
        axs.legend(loc='upper right', fontsize=10)

    output_path = os.path.join(path, 'attention_kernel_roofline.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=400)


def table_models(
        all_model_results: List[Dict[str, Any]],
        path: str
) -> None:
    plt.style.use('ggplot')

    if all_model_results is not None:
        import pickle
        with open('/gpfs/scratch/bsc98/bsc098949/vLLMServingPlateau', 'wb') as file:
            pickle.dump(all_model_results, file)
    else:
        import pickle
        with open('/gpfs/scratch/bsc98/bsc098949/vLLMServingPlateau', 'rb') as file:
            all_model_results = pickle.load(file)

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

    # filter only maximum batch size for every model and only xformers kernel
    right_all_model_results = []
    for results in all_model_results:
        model = results['model']
        batch_size = results['batch_size']
        kernel = results['kernel']
        if batch_size == model_max_batch_size[model] and kernel == 'xformers':
            right_all_model_results.append(deepcopy(results))

    # extract individual metrics
    for results in right_all_model_results:
        results['achieved_work'] = results['ncu_metrics']['achieved_work']
        results['achieved_traffic'] = results['ncu_metrics']['achieved_traffic']
    achieved_work_right = __prepare_lines(
        right_all_model_results,
        'batch_size',
        'achieved_work',
        'model'
    )
    achieved_traffic_right = __prepare_lines(
        right_all_model_results,
        'batch_size',
        'achieved_traffic',
        'model'
    )

    # show
    print('Common data')
    print('peak_work_single_precision', peak_work_single_precision)
    print('peak_work_double_precision', peak_work_double_precision)
    print('peak_traffic', peak_traffic)
    print('Data by model')
    for index_model in range(len(achieved_traffic_right)):
        model_label = achieved_traffic_right[index_model][0]
        assert model_label == achieved_work_right[index_model][0]
        model_achieved_traffic = achieved_traffic_right[index_model][2][0]
        model_achieved_work = achieved_work_right[index_model][2][0]
        print('model', model_label, 'model_achieved_traffic', model_achieved_traffic, 'model_achieved_work', model_achieved_work)


def main():
    model_results: List[Dict[str, Any]] = []
    model_results += extract_results('flash/opt-1.3b', 'opt-1.3b', 'flash')
    model_results += extract_results('xformers/opt-1.3b', 'opt-1.3b', 'xformers')
    model_results += extract_results('xformers/opt-2.7b', 'opt-2.7b', 'xformers')
    model_results += extract_results('xformers/llama-2-7b', 'llama-2-7b', 'xformers')
    model_results += extract_results('xformers/llama-2-13b', 'llama-2-13b', 'xformers')
    model_results += extract_results('matmul/opt-1.3b', 'opt-1.3b', 'matmul')

    plot_decode_timewise(
        model_results,
        '.'
    )

    table_models(
        model_results,
        '.'
    )


    '''plot_decode_timewise(
        None,
        '.'
    )

    table_models(
        None,
        '.'
    )'''


if __name__ == '__main__':
    main()
