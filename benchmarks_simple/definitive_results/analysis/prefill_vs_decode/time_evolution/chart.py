import os
import re
import glob
from typing import List, Tuple, Dict, Set, Any
import matplotlib.pyplot as plt


def extract_experiment_metric(path: str) -> Dict[str, float]:
    output: Dict[str, float] = {}

    # load log output
    filenames: List[str] = glob.glob(os.path.join(path, 'log_*.out'))
    if len(filenames) != 1:
        raise ValueError(f'More than one output result file or none {filenames} for path {path}')
    with open(filenames[0]) as file:
        log_out: str = file.read()

    # compute prefill time
    pattern = f'Prefill elapsed time: [+-]?([0-9]+([.][0-9]*)?|[.][0-9]+) seconds'
    found = re.search(pattern, log_out)
    if found is None:
        raise ValueError(f'Metric pattern not found on result log')
    output['prefill'] = float(found.group(1))

    # compute decode time
    pattern = f'Decode elapsed time: [+-]?([0-9]+([.][0-9]*)?|[.][0-9]+) seconds'
    found = re.search(pattern, log_out)
    if found is None:
        raise ValueError(f'Metric pattern not found on result log')
    output['decode'] = float(found.group(1))

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


def plot_time_evolution(
        all_model_results: List[Dict[str, float]],
        path: str
) -> None:
    plt.style.use('ggplot')

    _, prefill_x, prefill_y = __prepare_lines(
        all_model_results,
        'batch_size',
        'prefill',
        'model'
    )[0]

    _, decode_x, decode_y = __prepare_lines(
        all_model_results,
        'batch_size',
        'decode',
        'model'
    )[0]

    nrows = 1
    ncols = 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4), sharex=True)
    fig.tight_layout()
    # fig.subplots_adjust(wspace=0)

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    prefill_x = [str(item) for item in prefill_x]
    decode_x = [str(item) for item in decode_x]

    axs2 = axs.twinx()
    init_time = prefill_y[0] + decode_y[0]
    slowdown_y = [(prefill_y[index] + decode_y[index]) / init_time for index in range(len(prefill_x))]
    axs2.plot(prefill_x, slowdown_y, marker='o', label='global slowdown', color='red')
    axs2.set_ylabel('Global slowdown')
    axs2.grid(False)
    # axs2.set_xticks([])
    # axs2.set_yticks([])
    axs2.legend(loc='upper right', fontsize=10)

    axs.bar(prefill_x, prefill_y, label='prefill time')
    patches = axs.bar(decode_x, decode_y, bottom=prefill_y, label='decode time')
    axs.bar_label(patches, ['{0:.1f}%'.format(decode_y[index] / (prefill_y[index] + decode_y[index]) * 100) for index in range(len(prefill_x))], label_type='center')

    axs.set_xlabel('Batch size (reqs)')
    axs.set_ylabel('Time (s)')
    axs.legend(loc='upper left', fontsize=10)

    '''col_labels = ['Throughput (toks/s)', 'Time Between Tokens (ms)']
    for model, x_line, y_line in col_results:
        line = axs[index_x].plot(
            x_line,
            y_line,
            marker='o',
            label=model
        )[0]
    axs[index_x].set_ylabel(col_labels[index_x], fontsize=10)
    axs[index_x].set_xlabel('Average batch size (reqs)', fontsize=10)

    handles, labels = axs[index_x].get_legend_handles_labels()
    handles = [handles[2], handles[3], handles[1], handles[0]]
    labels = [labels[2], labels[3], labels[1], labels[0]]
    axs[index_x].legend(handles, labels, loc='center right', fontsize=10)'''

    plt.savefig(os.path.join(path, f'decode_vs_prefill_time_evolution'), bbox_inches='tight')


def plot_kv_cache(
        all_model_results: List[Dict[str, Any]],
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
    model_results: List[Dict[str, Any]] = []
    model = 'opt-2.7b'
    model_results += extract_results(model, model)

    plot_time_evolution(
        model_results,
        '.'
    )


if __name__ == '__main__':
    main()
