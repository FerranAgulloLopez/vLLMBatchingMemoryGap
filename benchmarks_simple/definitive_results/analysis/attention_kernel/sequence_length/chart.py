import os
import re
import glob
import pickle
from typing import List, Tuple, Dict, Set, Any
import matplotlib.pyplot as plt
import numpy as np


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


def plot_sequence_length(
        all_model_results: List[Dict[str, Any]],
        path: str
) -> None:
    # prepare data

    # extract important information
    prompt_results = {}
    output_results = {}
    for results in all_model_results:
        type = results['kernel']
        total_cycles = results['ncu_metrics']['total_cycles']
        waiting_cycles = results['ncu_metrics']['waiting_cycles']
        proportion = waiting_cycles / total_cycles * 100
        if type == 'increased_prompt_length':
            input_length = results['input_length']
            prompt_results[input_length] = proportion
        elif type == 'increased_output_length':
            output_length = results['output_length']
            output_results[output_length] = proportion
    del all_model_results

    # order
    prompt_x_line = list(prompt_results.keys())
    prompt_y_line = list(prompt_results.values())
    output_x_line = list(output_results.keys())
    output_y_line = list(output_results.values())
    prompt_y_line = [y for _, y in sorted(zip(prompt_x_line, prompt_y_line))]
    prompt_x_line = sorted(prompt_x_line)
    output_y_line = [y for _, y in sorted(zip(output_x_line, output_y_line))]
    output_x_line = sorted(output_x_line)

    # change x axis to proportion
    '''
    start = prompt_x_line[0]
    for index in range(len(prompt_x_line)):
        prompt_x_line[index] = f'x{prompt_x_line[index] / start}'
    start = output_x_line[0]
    for index in range(len(output_x_line)):
        output_x_line[index] = f'x{output_x_line[index] / start}'
    '''

    for index in range(len(prompt_x_line)):
        prompt_x_line[index] = str(prompt_x_line[index])
    for index in range(len(output_x_line)):
        output_x_line[index] = str(output_x_line[index])

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 13,
        'axes.titlesize': 15,
        'axes.labelsize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'lines.linewidth': 2.0,
        'mathtext.default': 'regular',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.4,
        'figure.figsize': (7, 5)  # Consistent size for single plot
    })

    # define figure
    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, layout='constrained', figsize=(8, 3.5), sharey=True, sharex=True)

    # plot figure
    for index_x, (label_results, x_line, y_line) in enumerate([('Prompt length', prompt_x_line, prompt_y_line), ('Output length', output_x_line, output_y_line)]):
        axs[index_x].bar(x_line, y_line, color='#0072B2', edgecolor='black')

        # label stuff
        axs[index_x].set_xlabel(label_results, fontsize=12)
        if index_x == 0:
            axs[index_x].set_ylabel('Idle cycles (%)', fontsize=13)

    output_path = os.path.join(path, 'attention_kernel_sequence_length.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=400)
    plt.show()


def main():
    model_results: List[Dict[str, Any]] = []
    if LOAD_PICKLE:
        with open(os.path.join(PICKLE_ROOT_PATH, 'attention_kernel_sequence_length'), 'rb') as file:
            model_results = pickle.load(file)
    else:
        model_results += extract_results('flash/increased_prompt_length/opt-1.3b', 'opt-1.3b', 'increased_prompt_length')
        model_results += extract_results('flash/increased_output_length/opt-1.3b', 'opt-1.3b', 'increased_output_length')

        with open(os.path.join(PICKLE_ROOT_PATH, 'attention_kernel_sequence_length'), 'wb') as file:
            pickle.dump(model_results, file)

    plot_sequence_length(
        model_results,
        '.'
    )


if __name__ == '__main__':
    main()
