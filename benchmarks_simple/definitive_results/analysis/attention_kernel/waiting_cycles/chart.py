import os
import re
import glob
import pickle
from typing import List, Dict, Set, Any
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


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
        'waiting_cycles': [],
        'l1_hit_rate': [],
        'l2_hit_rate': []
    }
    for index in range(ncu_metrics.num_actions()):
        ncu_profile_metrics = ncu_metrics.action_by_idx(index)

        # extract average total cycles per instruction
        output['ncu_metrics']['total_cycles'].append(ncu_profile_metrics.metric_by_name('smsp__average_warp_latency_per_inst_issued.ratio').as_double())

        # extract average cycles waiting for L1TEX data
        output['ncu_metrics']['waiting_cycles'].append(ncu_profile_metrics.metric_by_name('smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio').as_double())

        # extract l1 hit rate
        output['ncu_metrics']['l1_hit_rate'].append(
            100 *
            ncu_profile_metrics.metric_by_name('lts__t_sectors_lookup_hit.sum').as_double() /
            (
                ncu_profile_metrics.metric_by_name('lts__t_sectors_lookup_hit.sum').as_double() +
                ncu_profile_metrics.metric_by_name('lts__t_sectors_lookup_miss.sum').as_double()
            )

        )

        # extract l2 hit rate
        hits = \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_tex_mem_surface_op_ld_lookup_hit.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_tex_mem_texture_lookup_hit.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_lsu_mem_local_op_st_lookup_hit.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_tex_mem_surface_op_st_lookup_hit.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_lsu_mem_global_op_red_lookup_hit.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_tex_mem_surface_op_red_lookup_hit.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_hit.sum').as_double()

        misses = \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_lsu_mem_dshared_op_ld_lookup_miss.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_tex_mem_surface_op_ld_lookup_miss.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_tex_mem_texture_lookup_miss.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_lsu_mem_local_op_st_lookup_miss.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_lsu_mem_dshared_op_st_lookup_miss.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_tex_mem_surface_op_st_lookup_miss.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_lsu_mem_global_op_red_lookup_miss.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_lsu_mem_dshared_op_redas_lookup_miss.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_tex_mem_surface_op_red_lookup_miss.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_miss.sum').as_double() + \
            ncu_profile_metrics.metric_by_name('l1tex__t_sectors_pipe_tex_mem_surface_op_atom_lookup_miss.sum').as_double()

        output['ncu_metrics']['l2_hit_rate'].append(hits / (hits + misses) * 100)

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
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 13,
        'axes.titlesize': 15,
        'axes.labelsize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2.0,
        'mathtext.default': 'regular',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.4,
        'figure.figsize': (7, 5)  # Consistent size for single plot
    })

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
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, layout='constrained', figsize=(8, 3), sharey=True, sharex=False)

    # plot figure
    model_order = ['opt-1.3b', 'opt-2.7b', 'llama-2-7b', 'llama-2-13b']
    labels = ['OPT-1.3b', 'OPT-2.7b', 'LLaMA-2-7b', 'LLaMA-2-13b']
    x = np.arange(len(model_order))  # X positions for models
    bar_width = 0.4

    for index_x, (label_results, results) in enumerate([('Xformers Attention', xformers_results), ('Flash Attention', flash_results)]):
        min_values = []
        max_values = []

        for model in model_order:
            if model not in results:
                min_values.append(None)
                max_values.append(None)
            else:
                min_values.append(results[model]['1'])
                max_values.append(results[model]['max'])

        model_labels = [model for model, v in zip(labels, min_values) if v is not None]
        min_values = [v for v in min_values if v is not None]
        max_values = [v for v in max_values if v is not None]
        x = np.arange(len(model_labels))

        print(model_labels)

        axs[index_x].bar(x, max_values, bar_width, label='Batch size = MAX', color='#009E73', alpha=0.7, edgecolor='black', hatch='//')
        axs[index_x].bar(x, min_values, bar_width, label='Batch size = 1', color='#009E73', alpha=1, edgecolor='black')

        axs[index_x].set_title(label_results, fontsize=13)
        axs[index_x].set_ylabel('Idle cycles (%)', fontsize=13)
        axs[index_x].set_xticks(x)
        axs[index_x].set_xticklabels(model_labels, fontsize=12, rotation=30, ha="right")
        axs[index_x].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)

        axs[index_x].set_title(label_results)
        axs[index_x].set_ylim(0, 100)

    from matplotlib.patches import Patch
    mean_patch = Patch(facecolor='#009E73', edgecolor='black', label='Batch size = 1')
    max_patch = Patch(facecolor='#009E73', edgecolor='black', hatch='//', label='Batch size = MAX')
    legend_handles = [mean_patch, max_patch]
    fig.legend(handles=legend_handles, loc='upper center', ncol=2, frameon=False, fontsize=13, bbox_to_anchor=(0.5, 1.13))
    output_path = os.path.join(path, 'attention_kernel_waiting_cycles.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=400)


def cache_values(
        all_model_results: List[Dict[str, Any]],
        path: str
) -> None:
    plt.style.use('ggplot')

    # prepare data

    # get maximum batch size for every model
    model_max_batch_size = {}
    for results in all_model_results:
        model = results['model']
        batch_size = results['batch_size']
        if model not in model_max_batch_size:
            model_max_batch_size[model] = 0
        model_max_batch_size[model] = max(batch_size, model_max_batch_size[model])

    # filter only maximum batch size and 1 for every model and only xformers kernel
    right_all_model_results = []
    for results in all_model_results:
        model = results['model']
        batch_size = results['batch_size']
        kernel = results['kernel']
        if (batch_size == model_max_batch_size[model] or batch_size == 1) and kernel == 'xformers':
            right_all_model_results.append(deepcopy(results))
    del all_model_results

    # extract important metrics
    l1_hit_rate_1 = {}
    l1_hit_rate_max = {}
    l2_hit_rate_1 = {}
    l2_hit_rate_max = {}
    for results in right_all_model_results:
        model = results['model']
        l1_hit_rate = results['ncu_metrics']['l1_hit_rate']
        l2_hit_rate = results['ncu_metrics']['l2_hit_rate']
        batch_size = results['batch_size']
        if batch_size == 1:
            l1_hit_rate_1[model] = l1_hit_rate
            l2_hit_rate_1[model] = l2_hit_rate
        else:
            l1_hit_rate_max[model] = l1_hit_rate
            l2_hit_rate_max[model] = l2_hit_rate

    # show
    def print_metric_value(value: float):
        return '{:.2f}'.format(value)

    for model in l1_hit_rate_1.keys():
        print('Model', model, 'batch size', 1, 'l1 hit rate', print_metric_value(l1_hit_rate_1[model]), 'l2 hit rate', print_metric_value(l2_hit_rate_1[model]))
        print('Model', model, 'batch size', 'max', 'l1 hit rate', print_metric_value(l1_hit_rate_max[model]), 'l2 hit rate', print_metric_value(l2_hit_rate_max[model]))


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

    # cache_values(
    #     model_results,
    #     '.'
    # )


if __name__ == '__main__':
    main()
