import os
import re
import glob
import sqlite3
import shlex
import subprocess
import pandas as pd
from typing import List, Tuple, Dict, Set, Any
import numpy as np
import pickle

# FOR RUNNING SCRIPT, EXPORT FIRST nsys-rep REPORT WITH "nsys export --separate-strings yes --type sqlite .nsys-rep"
# MAYBE THE REPORT IS MISSING BECAUSE OF REPO SIZE CONSTRAINTS. IN THAT CASE, RERUN THE EXPS WITH THE PROVIDED CONFIG


def str_to_bool(string: str):
    return string.lower() in ['true', '1', 't', 'y', 'yes']

CREATE_SQLITE_DATABASES = str_to_bool(os.getenv('CREATE_SQLITE_DATABASES', False))
LOAD_PICKLE = str_to_bool(os.getenv('LOAD_PICKLE', False))
PICKLE_ROOT_PATH = os.getenv('PICKLE_ROOT_PATH', None)
assert PICKLE_ROOT_PATH is not None


def create_sqlite_databases(path: str):
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            filenames: List[str] = glob.glob(os.path.join(path, folder, 'log_*.err'))
            if len(filenames) != 1:
                raise ValueError(f'More than one output result file or none {filenames} for path {path}')
            with open(filenames[0], 'r') as fp:
                if len(fp.readlines()) > 24:
                    raise ValueError('Some error happened', filenames[0])

            command = f'nsys export --separate-strings yes --type sqlite --output {os.path.join(path, folder)}/.nsys-rep.sqlite --force-overwrite true {os.path.join(path, folder)}/.nsys-rep'
            print(command)
            result = subprocess.run(
                shlex.split(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            print(result.stdout)
            print(result.stderr)


def nsight_extract_benchmark_start_end_times(path: str):
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

    benchmark_start = None
    benchmark_end = None
    for index in range(np.shape(df)[0]):
        if df[index, 0] == 'BenchmarkStart':
            benchmark_start = float(df[index, 1])
        elif df[index, 0] == 'BenchmarkEnd':
            benchmark_end = float(df[index, 1])
    assert benchmark_start is not None
    assert benchmark_end is not None

    return benchmark_start, benchmark_end


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


def extract_experiment_metric(path: str, replicas: int) -> Dict[str, Any]:
    output: Dict[str, Any] = {}

    nsight_sqlite_file: str = os.path.join(path, '.nsys-rep.sqlite')

    # check configured replicas and load server logs
    filenames: List[str] = glob.glob(os.path.join(path, 'server_out_*.log'))
    if len(filenames) != replicas:
        raise ValueError(f'Server number does not coincide with set config')
    server_logs: List[str] = []
    for filename in filenames:
        with open(filename) as file:
            server_logs.append(file.read())

    # check no preemption
    for server_log in server_logs:
        pattern = 'is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space'
        found = re.search(pattern, server_log)
        if found is not None:
            raise ValueError(f'Preemption was present')

    # find benchmark start/end
    benchmark_start, benchmark_end = nsight_extract_benchmark_start_end_times(nsight_sqlite_file)

    # extract GPU metrics
    gpu_metrics = [
        'SMs Active [Throughput %]',
        'Compute Warps in Flight [Throughput %]',
        'Unallocated Warps in Active SMs [Throughput %]',
        'DRAM Read Bandwidth [Throughput %]',
        'DRAM Write Bandwidth [Throughput %]'
    ]
    output['gpu_metrics_values'] = nsight_extract_gpu_metric_arrays(nsight_sqlite_file, gpu_metrics, benchmark_start, benchmark_end)

    # extract CPU time - using the kernel execution directly would be better, but the generated nsight report would be too large
    cpu_time: float = 0
    values: np.ndarray = output['gpu_metrics_values']['SMs Active [Throughput %]']['y']
    times: np.ndarray = output['gpu_metrics_values']['SMs Active [Throughput %]']['x']
    assert np.shape(values)[0] == np.shape(times)[0]
    for index in range(1, np.shape(values)[0]):
        if values[index] == 0:
            cpu_time += times[index] - times[index - 1]
    output['cpu_time'] = (cpu_time / (benchmark_end - benchmark_start)) * 100

    return output


def extract_results(path: str, model: str) -> List[Dict[str, Any]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['model', 'set_batch_size', 'replicas', 'chunked']
    results = []
    rerun_errors: List[str] = []
    unknown_errors: int = 0
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            if folder == '_':
                continue
            batch_size: int = int(folder.split('_')[1])
            replicas: int = int(folder.split('_')[3])
            chunked: bool = True if 'chunked' in path else False
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
            metrics['chunked'] = chunked
            _id = create_id(metrics, id_metrics)
            if _id in collected_ids:
                raise ValueError('Repeated results')
            collected_ids.add(_id)
            results.append(metrics)
    print(f'Unknown extraction errors: {unknown_errors}. Should be zero.')
    print(f'Rerun errors: {len(rerun_errors)}. Should be zero. Full list: {rerun_errors}')
    return results


def show_table(
        all_model_results: List[Dict[str, Any]],
        path: str
) -> None:
    # prepare data

    # filter metrics
    metrics_to_show = {
        'Compute Warps in Flight [Throughput %]',
        'DRAM Read Bandwidth [Throughput %]'
    }
    for results in all_model_results:
        for metric_label in list(results['gpu_metrics_values'].keys()):
            if metric_label not in metrics_to_show:
                del results['gpu_metrics_values'][metric_label]

    # average metrics and prepare for good printing
    def print_metric_value(value: float):
        return '{:.2f}'.format(value)
    for results in all_model_results:
        results['cpu_time'] = print_metric_value(results['cpu_time'])
        for metric_label, metric_values in results['gpu_metrics_values'].items():
            results['gpu_metrics_values'][metric_label] = print_metric_value(float(np.mean(results['gpu_metrics_values'][metric_label]['y'])))

    # show
    for model_results in all_model_results:
        model = model_results['model']
        replicas = model_results['replicas']
        batch_size = model_results['set_batch_size']
        cpu_time = model_results['cpu_time']
        gpu_metrics = model_results['gpu_metrics_values']
        chunked = model_results['chunked']
        print(
            'Model', model,
            'batch size', batch_size,
            'chunked', chunked,
            'replicas', replicas,
            'CPU time', cpu_time,
            'GPU metrics', gpu_metrics
        )


def main():
    if CREATE_SQLITE_DATABASES:
        for model in ['opt-1.3b', 'opt-2.7b']:
            create_sqlite_databases(model)

    model_results: List[Dict[str, Any]] = []
    if LOAD_PICKLE:
        with open(os.path.join(PICKLE_ROOT_PATH, 'replication_gpu_metrics'), 'rb') as file:
            model_results = pickle.load(file)
    else:
        for model in ['opt-1.3b', 'opt-1.3b_chunked', 'opt-2.7b', 'opt-2.7b_chunked']:
            model_results += extract_results(model, model)
        with open(os.path.join(PICKLE_ROOT_PATH, 'replication_gpu_metrics'), 'wb') as file:
            pickle.dump(model_results, file)

    show_table(
        model_results,
        '.'
    )


if __name__ == '__main__':
    main()
