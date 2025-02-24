import os
import re
import glob
import sqlite3
import pandas as pd
from typing import List, Tuple, Dict, Set, Any
import matplotlib.pyplot as plt
import numpy as np

# FOR RUNNING SCRIPT, EXPORT FIRST nsys-rep REPORT WITH "nsys export --separate-strings yes --type sqlite .nsys-rep"
# MAYBE THE REPORT IS MISSING BECAUSE OF REPO SIZE CONSTRAINTS. IN THAT CASE, RERUN THE EXPS WITH THE PROVIDED CONFIG


def cut_metric_values_timewise(metric_x: np.ndarray, metric_y: np.ndarray, init_cut: float, end_cut: float):
    assert init_cut < end_cut
    assert metric_x[0] < init_cut < metric_x[-1]
    assert metric_x[0] < end_cut < metric_x[-1]
    init_index: int = None
    end_index: int = None
    iter_index: int = 0
    while (init_index is None or end_index is None) and iter_index < len(metric_x):
        if init_index is None and init_cut < metric_x[iter_index]:
            init_index = iter_index
        if end_index is None and end_cut < metric_x[iter_index]:
            end_index = iter_index
        iter_index += 1
    assert init_cut < end_cut
    return metric_x[init_index:end_index], metric_y[init_index:end_index]


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

    # compute decode importance
    output['decode_importance'] = output['decode'] / (output['prefill'] + output['decode'])

    # check if using flash attention as backend
    flash_attention: bool = True  # opt-2.7b model not compatible with flash backend -> Cannot use FlashAttention-2 backend for head size 80
    pattern = f'Using XFormers backend'
    found = re.search(pattern, log_out)
    if found is not None:
        flash_attention = False

    # find prefill/decode start/end
    prefill_start, prefill_end, decode_start, decode_end = nsight_extract_prefill_decode_start_end_times(nsight_sqlite_file)

    # extract GPU metrics
    gpu_metrics = [
        'SMs Active [Throughput %]',
        'Compute Warps in Flight [Throughput %]',
        'Unallocated Warps in Active SMs [Throughput %]',
        'DRAM Read Bandwidth [Throughput %]',
        'DRAM Write Bandwidth [Throughput %]'
    ]
    gpu_metrics_values_prefill_raw = nsight_extract_gpu_metric_arrays(nsight_sqlite_file, gpu_metrics, prefill_start, prefill_end)
    gpu_metrics_values_decode_raw = nsight_extract_gpu_metric_arrays(nsight_sqlite_file, gpu_metrics, decode_start, decode_end)

    # to debug correct cut of the prefill and the decode
    '''fig, ax = plt.subplots()
    line_x, line_y = gpu_metrics_values_prefill['DRAM Read Bandwidth [Throughput %]']['x'], gpu_metrics_values_prefill['DRAM Read Bandwidth [Throughput %]']['y']
    ax.plot(line_x, np.convolve(line_y, [0.05] * 20, 'same'), label='DRAM read', marker='')
    line_x, line_y = gpu_metrics_values_prefill['Unallocated Warps in Active SMs [Throughput %]']['x'], gpu_metrics_values_prefill['Unallocated Warps in Active SMs [Throughput %]']['y']
    ax.plot(line_x, np.convolve(line_y, [0.05] * 20, 'same'), label='SM UnOccupancy', marker='')
    ax.grid()
    ax.legend()
    fig.savefig("/home/ferran/Downloads/prefill.png")

    fig, ax = plt.subplots()
    line_x, line_y = gpu_metrics_values_decode['DRAM Read Bandwidth [Throughput %]']['x'], gpu_metrics_values_decode['DRAM Read Bandwidth [Throughput %]']['y']
    ax.plot(line_x, np.convolve(line_y, [0.05] * 20, 'same'), label='DRAM read', marker='')
    line_x, line_y = gpu_metrics_values_decode['Unallocated Warps in Active SMs [Throughput %]']['x'], gpu_metrics_values_decode['Unallocated Warps in Active SMs [Throughput %]']['y']
    ax.plot(line_x, np.convolve(line_y, [0.05] * 20, 'same'), label='SM UnOccupancy', marker='')
    ax.grid()
    ax.legend()
    fig.savefig("/home/ferran/Downloads/decode.png")'''

    # extract prefill and decode GPU metric values
    gpu_metrics_values_prefill = {}
    gpu_metrics_values_decode = {}
    for gpu_metric in gpu_metrics:
        y_values_prefill = gpu_metrics_values_prefill_raw[gpu_metric]['y']
        y_values_decode = gpu_metrics_values_decode_raw[gpu_metric]['y']
        '''y_values_prefill = y_values_prefill[y_values_prefill != 0]
        y_values_decode = y_values_decode[y_values_decode != 0]'''
        gpu_metrics_values_prefill[gpu_metric] = y_values_prefill
        gpu_metrics_values_decode[gpu_metric] = y_values_decode
    output['gpu_metrics_values_prefill'] = gpu_metrics_values_prefill
    output['gpu_metrics_values_decode'] = gpu_metrics_values_decode

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


def print_table(
        all_model_results: List[Dict[str, Any]],
        path: str,
        type: str
) -> None:
    plt.style.use('ggplot')

    def print_metric_value(value: float):
        return '{:.2f}'.format(value)

    if all_model_results is not None:
        import pickle
        with open('/home/ferran/Downloads/prefill_vs_decode_summary_table', 'wb') as file:
            pickle.dump(all_model_results, file)
    else:
        import pickle
        with open('/home/ferran/Downloads/prefill_vs_decode_summary_table', 'rb') as file:
            all_model_results = pickle.load(file)

    if type == 'together':
        prefill_importance = [1 - item['decode_importance'] for item in all_model_results]
        print(f'Prefill Importance: {print_metric_value(float(np.mean(prefill_importance)))}')
        decode_importance = [item['decode_importance'] for item in all_model_results]
        print(f'Decode Importance: {print_metric_value(float(np.mean(decode_importance)))}')

        for metric in all_model_results[0]['gpu_metrics_values_prefill'].keys():
            prefill_sum_average = 0
            prefill_sum_max = 0
            decode_sum_max = 0
            decode_sum_average = 0
            count = 0
            for model_results in all_model_results:
                prefill_values = model_results['gpu_metrics_values_prefill'][metric]
                decode_values = model_results['gpu_metrics_values_decode'][metric]
                prefill_sum_average += np.mean(prefill_values)
                prefill_sum_max += np.max(prefill_values)
                decode_sum_average += np.mean(decode_values)
                decode_sum_max += np.max(decode_values)
                count += 1
            print(f'{metric} Average. Prefill: {print_metric_value(prefill_sum_average / count)}. Decode: {print_metric_value(decode_sum_average / count)}')
            print(f'{metric} Max. Prefill: {print_metric_value(prefill_sum_max / count)}. Decode: {print_metric_value(decode_sum_max / count)}')

    elif type == 'separated':
        output_models = []
        output_lines = {}
        output_lines['Importance'] = []
        for gpu_metric in all_model_results[0]['gpu_metrics_values_prefill'].keys():
            output_lines[gpu_metric] = {
                'average': [],
                'max': []
            }

        for model_results in all_model_results:
            output_models.append(model_results['model'])

            prefill_importance = print_metric_value(1 - model_results['decode_importance'])
            decode_importance = print_metric_value(model_results['decode_importance'])
            output_lines['Importance'].append(prefill_importance)
            output_lines['Importance'].append(decode_importance)

            for metric in all_model_results[0]['gpu_metrics_values_prefill'].keys():
                prefill_values = model_results['gpu_metrics_values_prefill'][metric]
                decode_values = model_results['gpu_metrics_values_decode'][metric]
                prefill_average = print_metric_value(float(np.mean(prefill_values)))
                prefill_max = print_metric_value(float(np.max(prefill_values)))
                decode_average = print_metric_value(float(np.mean(decode_values)))
                decode_max = print_metric_value(float(np.max(decode_values)))
                output_lines[metric]['average'].append(prefill_average)
                output_lines[metric]['average'].append(decode_average)
                output_lines[metric]['max'].append(prefill_max)
                output_lines[metric]['max'].append(decode_max)

        print(output_models)
        for line_label, line_values in output_lines.items():
            if line_label == 'Importance':
                str_values = ' & '.join(line_values)
                print(f'{line_label} -> {str_values}')
            else:
                str_values_average = ' & '.join(line_values['average'])
                print(f'{line_label} -> Average & {str_values_average}')
                str_values_max = ' & '.join(line_values['max'])
                print(f'{line_label} -> Max & {str_values_max}')


def main():
    model_results: List[Dict[str, Any]] = []
    for model in ['opt-1.3b', 'opt-2.7b', 'llama-2-7b', 'llama-2-13b']:
        model_results += extract_results(model, model)

    print_table(
        model_results,
        '.',
        'separated'
    )

    '''print_table(
        None,
        '.',
        'separated'
    )'''


if __name__ == '__main__':
    main()
