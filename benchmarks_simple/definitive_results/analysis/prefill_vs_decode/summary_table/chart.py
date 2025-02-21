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


def nsight_extract_gpu_metric_array(path: str, metric: str):
    try:
        conn = sqlite3.connect(path)
        query = f"""
                SELECT timestamp, value
                FROM GPU_METRICS
                JOIN TARGET_INFO_GPU_METRICS USING (metricId)
                WHERE (metricName == '{metric}')
            """
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        raise e

    if df.empty:
        raise Exception('Nothing found in SQLite database')

    df = df.to_numpy()
    metric_x = df[:, 0]
    metric_y = df[:, 1]

    return metric_x, metric_y


def nsight_extract_kernel_start_values(path: str, kernel_name: str):
    try:
        conn = sqlite3.connect(path)
        query = f"""
                SELECT start, end
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                JOIN StringIds ON CUPTI_ACTIVITY_KIND_KERNEL.shortName = StringIds.id
                WHERE StringIds.value == '{kernel_name}' AND streamId == 7
            """
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        raise e

    if df.empty:
        raise Exception('Nothing found in SQLite database')

    df = df.to_numpy()

    return df[:, 0]


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

    # extract complete SM occupancy and DRAM bandwidth
    dram_read_x, dram_read_y = nsight_extract_gpu_metric_array(nsight_sqlite_file, 'DRAM Read Bandwidth [Throughput %]')
    sm_unocc_x, sm_unocc_y = nsight_extract_gpu_metric_array(nsight_sqlite_file, 'Unallocated Warps in Active SMs [Throughput %]')

    # find prefill start -> start of flash_fwd_kernel
    kernel_start_values = nsight_extract_kernel_start_values(nsight_sqlite_file, 'reshape_and_cache_flash_kernel' if flash_attention else 'reshape_and_cache_kernel')
    prefill_start = np.min(kernel_start_values)

    # find decode start and end -> start of flash_fwd_splitkv_kernel
    kernel_start_values = nsight_extract_kernel_start_values(nsight_sqlite_file, 'flash_fwd_splitkv_kernel' if flash_attention else 'paged_attention_v1_kernel')
    decode_start = np.min(kernel_start_values)
    decode_end = np.max(kernel_start_values)

    '''fig, ax = plt.subplots()
    meh = cut_metric_values_timewise(dram_read_x, dram_read_y, prefill_start, decode_start)
    ax.plot(meh[0], np.convolve(meh[1], [0.05] * 20, 'same'), label='DRAM read')
    meh = cut_metric_values_timewise(sm_unocc_x, sm_unocc_y, prefill_start, decode_start)
    ax.plot(meh[0], np.convolve(meh[1], [0.05] * 20, 'same'), label='SM UnOccupancy')
    ax.grid()
    ax.legend()
    fig.savefig("/home/ferran/Downloads/prefill.png")

    fig, ax = plt.subplots()
    meh = cut_metric_values_timewise(dram_read_x, dram_read_y, decode_start, decode_end)
    ax.plot(meh[0], meh[1], label='DRAM read')
    meh = cut_metric_values_timewise(sm_unocc_x, sm_unocc_y, decode_start, decode_end)
    ax.plot(meh[0], meh[1], label='SM UnOccupancy')
    ax.grid()
    ax.legend()
    fig.savefig("/home/ferran/Downloads/decode.png")'''

    # extract prefill average and max SM occupancy
    _, sm_unocc_y_prefill = cut_metric_values_timewise(sm_unocc_x, sm_unocc_y, prefill_start, decode_start)
    sm_unocc_y_prefill = sm_unocc_y_prefill[sm_unocc_y_prefill != 0]
    output['prefill_average_unocc'] = float(np.mean(sm_unocc_y_prefill))
    output['prefill_max_unocc'] = float(np.max(sm_unocc_y_prefill))

    # extract prefill average and max DRAM bandwidth
    _, dram_read_y_prefill = cut_metric_values_timewise(dram_read_x, dram_read_y, prefill_start, decode_start)
    dram_read_y_prefill = dram_read_y_prefill[dram_read_y_prefill != 0]
    output['prefill_average_dram_read'] = float(np.mean(dram_read_y_prefill))
    output['prefill_max_dram_read'] = float(np.max(dram_read_y_prefill))

    # extract decode average and max SM occupancy
    _, sm_unocc_y_decode = cut_metric_values_timewise(sm_unocc_x, sm_unocc_y, decode_start, decode_end)
    sm_unocc_y_decode = sm_unocc_y_decode[sm_unocc_y_decode != 0]
    output['decode_average_unocc'] = float(np.mean(sm_unocc_y_decode))
    output['decode_max_unocc'] = float(np.max(sm_unocc_y_decode))

    # extract decode average and max DRAM bandwidth
    _, dram_read_y_decode = cut_metric_values_timewise(dram_read_x, dram_read_y, decode_start, decode_end)
    dram_read_y_decode = dram_read_y_decode[dram_read_y_decode != 0]
    output['decode_average_dram_read'] = float(np.mean(dram_read_y_decode))
    output['decode_max_dram_read'] = float(np.max(dram_read_y_decode))

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
        path: str
) -> None:
    plt.style.use('ggplot')

    if all_model_results is not None:
        import pickle
        with open('/home/ferran/Downloads/meh', 'wb') as file:
            pickle.dump(all_model_results, file)
    else:
        import pickle
        with open('/home/ferran/Downloads/meh', 'rb') as file:
            all_model_results = pickle.load(file)

    decode_importance = __prepare_lines(
        all_model_results,
        'batch_size',
        'decode_importance',
        'model'
    )

    sm_unocc_average = __prepare_lines(
        all_model_results,
        'prefill_average_unocc',
        'decode_average_unocc',
        'model'
    )

    sm_unocc_max = __prepare_lines(
        all_model_results,
        'prefill_max_unocc',
        'decode_max_unocc',
        'model'
    )

    dram_read_average = __prepare_lines(
        all_model_results,
        'prefill_average_dram_read',
        'decode_average_dram_read',
        'model'
    )

    dram_read_max = __prepare_lines(
        all_model_results,
        'prefill_max_dram_read',
        'decode_max_dram_read',
        'model'
    )

    decode_importance = [item[2] for item in decode_importance]
    sm_unocc_average_prefill = [item[1] for item in sm_unocc_average]
    sm_unocc_average_decode = [item[2] for item in sm_unocc_average]
    sm_unocc_max_prefill = [item[1] for item in sm_unocc_max]
    sm_unocc_max_decode = [item[2] for item in sm_unocc_max]
    dram_read_average_prefill = [item[1] for item in dram_read_average]
    dram_read_average_decode = [item[2] for item in dram_read_average]
    dram_read_max_prefill = [item[1] for item in dram_read_max]
    dram_read_max_decode = [item[2] for item in dram_read_max]

    print(f'Decode Importance: {float(np.mean(decode_importance))}')
    print(f'SM Unoccupancy Average. Prefill: {float(np.mean(sm_unocc_average_prefill))}. Decode: {float(np.mean(sm_unocc_average_decode))}')
    print(f'SM Unoccupancy Max. Prefill: {float(np.mean(sm_unocc_max_prefill))}. Decode: {float(np.mean(sm_unocc_max_decode))}')
    print(f'DRAM Read Average. Prefill: {float(np.mean(dram_read_average_prefill))}. Decode: {float(np.mean(dram_read_average_decode))}')
    print(f'DRAM Read Max. Prefill: {float(np.mean(dram_read_max_prefill))}. Decode: {float(np.mean(dram_read_max_decode))}')


def main():
    model_results: List[Dict[str, Any]] = []
    for model in ['opt-1.3b', 'opt-2.7b', 'llama-2-7b', 'llama-2-13b']:
        model_results += extract_results(model, model)

    print_table(
        model_results,
        '.'
    )

    '''print_table(
        None,
        '.'
    )'''


if __name__ == '__main__':
    main()
