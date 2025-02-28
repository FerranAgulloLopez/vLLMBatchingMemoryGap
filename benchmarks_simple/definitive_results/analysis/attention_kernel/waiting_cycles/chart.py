import os
import re
import glob
import pickle
from decimal import Decimal
from typing import List, Tuple, Dict, Set, Any
import matplotlib.pyplot as plt
import numpy as np
# This following package comes from the NCU installation, add the corresponding path to the PYTHONPATH env variable, or check the official documentation
# in our specific case PYTHONPATH=PYTHONPATH:/usr/local/NVIDIA-Nsight-Compute/extras/python/
import ncu_report
from copy import deepcopy


# MAYBE THE NCU REPORT IS MISSING BECAUSE OF REPO SIZE CONSTRAINTS. IN THAT CASE, RERUN THE EXPS WITH THE PROVIDED CONFIG

def str_to_bool(string: str):
    return string.lower() in ['true', '1', 't', 'y', 'yes']

LOAD_PICKLE = str_to_bool(os.getenv('LOAD_PICKLE', False))
PICKLE_ROOT_PATH = os.getenv('PICKLE_ROOT_PATH', None)
assert PICKLE_ROOT_PATH is not None


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
        'total_cycles': [],
        'waiting_cycles': []
    }
    for index in range(ncu_metrics.num_actions()):
        ncu_profile_metrics = ncu_metrics.action_by_idx(index)

        # extract average total cycles per instruction
        output['ncu_metrics']['total_cycles'].append(ncu_profile_metrics.metric_by_name('smsp__average_warp_latency_per_inst_issued.ratio').as_double())

        # extract average cycles waiting for L1TEX data
        output['ncu_metrics']['waiting_cycles'].append(ncu_profile_metrics.metric_by_name('smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio').as_double())

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


def plot_decode_cycles(
        all_model_results: List[Dict[str, Any]],
        path: str
) -> None:
    plt.style.use('ggplot')

    # prepare data

    # extract important information
    flash_results = {}
    xformers_results = {}
    for results in all_model_results:
        model = results['model']
        kernel = results['kernel']
        batch_size = results['batch_size']
        total_cycles = results['ncu_metrics']['total_cycles']
        waiting_cycles = results['ncu_metrics']['waiting_cycles']
        proportion = waiting_cycles / total_cycles * 100
        if kernel == 'flash':
            if model not in flash_results:
                flash_results[model] = {
                    '1': None,
                    'max': None
                }
            if batch_size == 1:
                flash_results[model]['1'] = proportion
            else:
                flash_results[model]['max'] = proportion
        elif kernel == 'xformers':
            if model not in xformers_results:
                xformers_results[model] = {
                    '1': None,
                    'max': None
                }
            if batch_size == 1:
                xformers_results[model]['1'] = proportion
            else:
                xformers_results[model]['max'] = proportion
    del all_model_results

    # define figure
    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, layout='constrained', figsize=(9, 3), sharey=True, sharex=True)

    # plot figure
    model_order = ['opt-1.3b', 'opt-2.7b', 'llama-2-7b', 'llama-2-13b']
    model_labels = ['OPT-1.3b', 'OPT-2.7b', 'LLaMA-2-7b', 'LLaMA-2-13b']
    for index_x, (label_results, results) in enumerate([('xformers attention', xformers_results), ('flash attention', flash_results)]):
        min_values = []
        max_values = []
        for model in model_order:
            if model not in results:
                min_values.append(None)
                max_values.append(None)
            else:
                min_values.append(results[model]['1'])
                max_values.append(results[model]['max'])

        axs[index_x].hlines(y=list(reversed(model_labels)), xmin=list(reversed(min_values)), xmax=list(reversed(max_values)), color='grey', zorder=1)
        axs[index_x].scatter(list(reversed(min_values)), list(reversed(model_labels)), label='batch size = 1')
        axs[index_x].scatter(list(reversed(max_values)), list(reversed(model_labels)), label='batch size = MAX')

        # label stuff
        axs[index_x].set_title(label_results)
        axs[index_x].set_xlabel('Stalled cycles waiting for data (%)')
        axs[index_x].legend(loc='center left', fontsize=10)

    plt.savefig(os.path.join(path, f'attention_kernel_waiting_cycles'), bbox_inches='tight')


def main():
    model_results: List[Dict[str, Any]] = []
    if LOAD_PICKLE:
        with open(os.path.join(PICKLE_ROOT_PATH, 'attention_kernel_waiting_cycles'), 'rb') as file:
            model_results = pickle.load(file)
    else:
        model_results += extract_results('flash/opt-1.3b', 'opt-1.3b', 'flash')
        model_results += extract_results('flash/llama-2-7b', 'llama-2-7b', 'flash')
        model_results += extract_results('flash/llama-2-13b', 'llama-2-13b', 'flash')
        model_results += extract_results('xformers/opt-1.3b', 'opt-1.3b', 'xformers')
        model_results += extract_results('xformers/opt-2.7b', 'opt-2.7b', 'xformers')
        model_results += extract_results('xformers/llama-2-7b', 'llama-2-7b', 'xformers')
        model_results += extract_results('xformers/llama-2-13b', 'llama-2-13b', 'xformers')

        with open(os.path.join(PICKLE_ROOT_PATH, 'attention_kernel_waiting_cycles'), 'wb') as file:
            pickle.dump(model_results, file)

    plot_decode_cycles(
        model_results,
        '.'
    )


if __name__ == '__main__':
    main()
