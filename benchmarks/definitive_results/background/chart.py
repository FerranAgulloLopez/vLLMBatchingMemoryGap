import os
import csv
import json
import glob
from typing import List, Tuple, Dict, Set
import matplotlib.pyplot as plt
import numpy as np


def extract_experiment_metric(path: str) -> Dict[str, float]:
    output: Dict[str, float] = {}

    # load global execution summary
    filenames: List[str] = glob.glob(os.path.join(path, 'openai-infqps-*.json'))
    if len(filenames) != 1:
        raise ValueError(f'More than one output result file or none {filenames} for path {path}')
    with open(filenames[0]) as metrics_file:
        execution_summary: dict = json.load(metrics_file)

    # load metric extracted values
    metric_values: Dict[str, List[float]] = {}
    with open(os.path.join(path, 'metrics_engine_server_0.csv'), newline='') as file:
        reader = csv.reader(file, delimiter=',')
        header = next(reader)
        header_indexes: Dict[int, str] = {}
        for index, item in enumerate(header):
            metric_values[item] = []
            header_indexes[index] = item
        for row in reader:
            for index, item in enumerate(row):
                metric_values[header_indexes[index]].append(float(item))

    # check if preemption
    if np.any(np.asarray(metric_values['num_preemptions_total']) > 0):
        raise ValueError

    # compute mean batch size
    output['batch_size'] = float(np.mean(np.asarray(metric_values['num_requests_running'])))

    # compute throughput
    output['throughput'] = float(execution_summary['total_token_throughput'])

    # compute latency
    output['latency'] = float(execution_summary['mean_itl_ms'])

    # compute max KV cache utilization
    output['kv_cache'] = np.max(np.asarray(metric_values['gpu_cache_usage_perc'])) * 100

    return output


def extract_results(path: str, model: str) -> List[Dict[str, float]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['model', 'set_batch_size']
    results = []
    rerun_errors: List[str] = []
    unknown_errors: int = 0
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            if folder == '_':
                continue
            batch_size: int = int(folder.split('_')[1])
            try:
                metrics = extract_experiment_metric(os.path.join(path, folder))
            except Exception as e:
                error_message: str = f'WARNING! Error while extracting results -> {os.path.join(path, folder)}. '
                with open(os.path.join(path, folder, 'server_err_0.log')) as f:
                    error_log: str = f.read()
                    if 'ValueError: No available memory for the cache blocks' in error_log:
                        error_message += 'Not enough memory'
                    elif 'torch.cuda.OutOfMemoryError: CUDA out of memory' in error_log:
                        error_message += 'Not enough memory'
                    elif 'ValueError: The model\'s max seq len (4096) is larger than the maximum number of tokens that can be stored in KV cache' in error_log:
                        error_message += 'Not enough memory'
                    elif 'RuntimeError: CUDA error: uncorrectable ECC error encountered' in error_log:
                        error_message += 'ECC error'
                        rerun_errors.append(os.path.join(path, folder))
                    elif 'RuntimeError: CUDA error: an illegal memory access was encountered' in error_log:
                        error_message += 'Memory access error'
                        rerun_errors.append(os.path.join(path, folder))
                    elif '[Errno 98] error while attempting to bind on address' in error_log:
                        error_message += 'Port bind error'
                        rerun_errors.append(os.path.join(path, folder))
                    else:
                        error_message += 'Unknown error'
                        unknown_errors += 1
                # print(error_message)
                metrics = {}
            metrics['model'] = model
            metrics['set_batch_size'] = batch_size  # in case of error
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


def plot_throughput_latency(
        all_model_results: List[Dict[str, float]],
        path: str,
        type: str
) -> None:
    plt.style.use('ggplot')
    # plt.style.use('seaborn-v0_8')

    '''meh = __prepare_lines(
        all_model_results,
        'set_batch_size',
        'throughput',
        'model'
    )'''

    cols = []
    cols.append(__prepare_lines(
        all_model_results,
        'batch_size',
        'throughput',
        'model'
    ))
    cols.append(__prepare_lines(
        all_model_results,
        'batch_size',
        'latency',
        'model'
    ))

    if type == 'horizontal':
        nrows = 1
        ncols = 2
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4), sharex=True)
        fig.tight_layout()
        # fig.subplots_adjust(wspace=0)

        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)

        col_labels = ['Throughput (toks/s)', 'Latency (ms)']
        for index_x, col_results in enumerate(cols):
            for model, x_line, y_line in col_results:
                line = axs[index_x].plot(
                    x_line,
                    y_line,
                    marker='o',
                    label=model
                )[0]
                if len(x_line) < 10:
                    x_final_point = x_line[-1] + 10
                    y_final_point = y_line[-1] + (y_line[-1] - y_line[-2]) / 5
                    axs[index_x].plot(
                        x_line[-1:] + [x_final_point],
                        y_line[-1:] + [y_final_point],
                        marker='x',
                        markevery=[1],
                        linestyle='dotted',
                        color=line.get_color()
                    )
            axs[index_x].set_ylabel(col_labels[index_x], fontsize=10)
            axs[index_x].set_xlabel('Average batch size (reqs)', fontsize=10)

            handles, labels = axs[index_x].get_legend_handles_labels()
            handles = [handles[2], handles[3], handles[1], handles[0]]
            labels = [labels[2], labels[3], labels[1], labels[0]]
            axs[index_x].legend(handles, labels, loc='upper right', fontsize=10)
    elif type == 'vertical':
        nrows = 2
        ncols = 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4), sharex=True)
        fig.tight_layout()
        # fig.subplots_adjust(wspace=0)

        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)

        col_labels = ['Throughput (toks/s)', 'Time Between Tokens (ms)']
        for index_x, col_results in enumerate(cols):
            for model, x_line, y_line in col_results:
                line = axs[index_x].plot(
                    x_line,
                    y_line,
                    marker='o',
                    label=model
                )[0]
                if x_line[-1] < 300:
                    '''x_final_point = x_line[-1] + 10
                    y_final_point = y_line[-1] + (y_line[-1] - y_line[-2]) / 5'''
                    x_final_point = x_line[-1] + (x_line[-1] - x_line[-2]) / 2
                    y_final_point = y_line[-1] + (y_line[-1] - y_line[-2]) / 2
                    axs[index_x].plot(
                        x_line[-1:] + [x_final_point],
                        y_line[-1:] + [y_final_point],
                        marker='x',
                        markevery=[1],
                        linestyle='dotted',
                        color=line.get_color()
                    )
            axs[index_x].set_ylabel(col_labels[index_x], fontsize=10)
            axs[index_x].set_xlabel('Average batch size (reqs)', fontsize=10)

            handles, labels = axs[index_x].get_legend_handles_labels()
            handles = [handles[2], handles[3], handles[1], handles[0]]
            labels = [labels[2], labels[3], labels[1], labels[0]]
            axs[index_x].legend(handles, labels, loc='center right', fontsize=10)

    plt.savefig(os.path.join(path, f'background_throughput_latency'), bbox_inches='tight')


def plot_kv_cache(
        all_model_results: List[Dict[str, float]],
        path: str
) -> None:
    plt.style.use('ggplot')

    all_model_results = __prepare_lines(
        all_model_results,
        'kv_cache',
        'throughput',
        'model'
    )

    nrows = 1
    ncols = 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4), sharex=True)
    fig.tight_layout()
    # fig.subplots_adjust(wspace=0)

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    for model, x_line, y_line in all_model_results:
        line = axs.plot(
            x_line,
            y_line,
            marker='o',
            label=model
        )[0]
    axs.set_ylabel('Throughput (toks/s)', fontsize=10)
    axs.set_xlabel('KV cache maximum usage (%)', fontsize=10)

    handles, labels = axs.get_legend_handles_labels()
    handles = [handles[2], handles[3], handles[1], handles[0]]
    labels = [labels[2], labels[3], labels[1], labels[0]]
    axs.legend(handles, labels, loc='center right', fontsize=10)

    plt.savefig(os.path.join(path, f'background_kv_cache'), bbox_inches='tight')


def main():
    model_results: List[Dict[str, float]] = []
    for model in ['opt-1.3b', 'opt-2.7b', 'llama-2-7b', 'llama-2-13b']:
        model_results += extract_results(model, model)

    plot_throughput_latency(
        model_results,
        '.',
        'vertical'
    )

    plot_kv_cache(
        model_results,
        '.'
    )


if __name__ == '__main__':
    main()
