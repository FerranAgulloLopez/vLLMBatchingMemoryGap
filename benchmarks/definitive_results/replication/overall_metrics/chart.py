import os
import csv
import json
import glob
from typing import List, Tuple, Dict, Set, Any
import numpy as np


def extract_experiment_metric(path: str, replicas: int) -> Dict[str, float]:
    output: Dict[str, float] = {}

    # check configured replicas
    filenames: List[str] = glob.glob(os.path.join(path, 'server_out_*.log'))
    if len(filenames) != replicas:
        raise ValueError(f'Server number does not coincide with set config')

    # load global execution summary
    filenames: List[str] = glob.glob(os.path.join(path, 'openai-infqps-*.json'))
    if len(filenames) != 1:
        raise ValueError(f'More than one output result file or none {filenames} for path {path}')
    with open(filenames[0]) as metrics_file:
        execution_summary: dict = json.load(metrics_file)

    # load metric extracted values
    metric_values_list: List[Dict[str, List[float]]] = []
    with open(os.path.join(path, 'metrics_engine_server_0.csv'), newline='') as file:
        metric_values: Dict[str, List[float]] = {}
        reader = csv.reader(file, delimiter=',')
        header = next(reader)
        header_indexes: Dict[int, str] = {}
        for index, item in enumerate(header):
            metric_values[item] = []
            header_indexes[index] = item
        for row in reader:
            for index, item in enumerate(row):
                metric_values[header_indexes[index]].append(float(item))
        metric_values_list.append(metric_values)

    # check if preemption
    for metric_values in metric_values_list:
        if np.any(np.asarray(metric_values['num_preemptions_total']) > 0):
            raise ValueError

    # compute throughput
    output['throughput'] = float(execution_summary['total_token_throughput'])

    # compute latency
    output['latency'] = float(execution_summary['mean_itl_ms'])

    # compute max KV cache utilization
    max_kv_cache_utilization: float = 0
    for metric_values in metric_values_list:
        max_kv_cache_utilization += float(np.max(np.asarray(metric_values['gpu_cache_usage_perc'])) * 100)
    max_kv_cache_utilization /= len(metric_values_list)
    output['kv_cache'] = max_kv_cache_utilization

    return output


def extract_results(path: str, model: str) -> List[Dict[str, Any]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['model', 'set_batch_size', 'replicas']
    results = []
    rerun_errors: List[str] = []
    unknown_errors: int = 0
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            if folder == '_':
                continue
            batch_size: int = int(folder.split('_')[1])
            replicas: int = int(folder.split('_')[3])
            try:
                metrics = extract_experiment_metric(os.path.join(path, folder), replicas)
            except Exception as e:
                error_message: str = f'WARNING! Error while extracting results -> {os.path.join(path, folder)}. '
                error_message += 'Unknown error'
                unknown_errors += 1
                # print(error_message)
                print(os.path.join(path, folder), e)
                metrics = {}
            metrics['model'] = model
            metrics['set_batch_size'] = batch_size
            metrics['replicas'] = replicas
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


def show_table(
        all_model_results: List[Dict[str, Any]],
        path: str
) -> None:
    # prepare data

    # order by model
    ordered_model_results: Dict[List[Dict[str, Any]]] = {}
    for results in all_model_results:
        model = results['model']
        if model not in ordered_model_results:
            ordered_model_results[model] = []
        ordered_model_results[model].append(results)

    # show
    def print_metric_value(value: float):
        return '{:.2f}'.format(value)
    for model, model_results in ordered_model_results.items():
        for results in model_results:
            replicas = results['replicas']
            throughput = results['throughput']
            latency = results['latency']
            kv_cache = results['kv_cache']
            batch_size = results['set_batch_size']
            print(
                'Model', model,
                'batch_size', batch_size,
                'replicas', replicas,
                'throughput', print_metric_value(throughput),
                'latency', print_metric_value(latency),
                'kv_cache', print_metric_value(kv_cache),
            )


def main():
    model_results: List[Dict[str, Any]] = []
    for model in ['opt-1.3b', 'opt-2.7b']:
        model_results += extract_results(model, model)

    show_table(
        model_results,
        '.'
    )


if __name__ == '__main__':
    main()
