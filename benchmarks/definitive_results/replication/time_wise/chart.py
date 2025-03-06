import os
import re
import glob
import sqlite3
import shlex
import json
import subprocess
import pandas as pd
from typing import List, Tuple, Dict, Set, Any
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
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


def nsight_extract_kernels_start_end_values_timewise(path: str, start_time: float, end_time: float):
    try:
        conn = sqlite3.connect(path)
        query = f"""
                SELECT DISTINCT globalPid, start, end
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                WHERE streamId == 7 AND start > {start_time} AND end < {end_time}
            """
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        raise e

    if df.empty:
        raise Exception('Nothing found in SQLite database')

    df = df.to_numpy()

    return df


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

    # extract kernel start end times
    kernel_times = nsight_extract_kernels_start_end_values_timewise(nsight_sqlite_file, benchmark_start, benchmark_end)

    # order kernel times by replica
    replica_kernel_times = {}
    for index in range(np.shape(kernel_times)[0]):
        process_id: float = float(kernel_times[index, 0])
        start_time: float = float(kernel_times[index, 1])
        end_time: float = float(kernel_times[index, 2])
        if process_id not in replica_kernel_times:
            replica_kernel_times[process_id] = []
        replica_kernel_times[process_id].append((start_time, end_time))
    output['replica_kernel_times'] = list(replica_kernel_times.values())

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


def __cut_kernels_by_time(kernel_values: List[Tuple[float, float]], start_time, end_time):
    output_kernel_values: List[Tuple[float, float]] = []
    for kernel_start, kernel_end in kernel_values:
        if kernel_start > start_time and kernel_end < end_time:
            output_kernel_values.append((kernel_start, kernel_end))
    return output_kernel_values


def plot_timewise(
        all_model_results: List[Dict[str, Any]],
        path: str
) -> None:
    # prepare data

    # extract only kernels by replica information
    kernel_data: Dict[int, List[List[Tuple[float, float]]]] = {}
    for model_results in all_model_results:
        replicas = model_results['replicas']
        kernel_data[replicas] = model_results['replica_kernel_times']
    del all_model_results

    # normalize times to start counting from first kernel
    def find_minimum_replica_kernel_start_time(replica_kernels: List[Tuple[float, float]]):
        return min([start_time for (start_time, _) in replica_kernels])
    for replica_values in kernel_data.values():
        first_time: float = min([find_minimum_replica_kernel_start_time(values) for values in replica_values])
        for values in replica_values:
            for index in range(len(values)):
                values[index] = (values[index][0] - first_time, values[index][1] - first_time)

    # determine window to plot for each replica config
    def find_maximum_replica_kernel_start_time(replica_kernels: List[Tuple[float, float]]):
        return max([start_time for (start_time, _) in replica_kernels])
    start_index_by_replica = {
        1: min(
            [find_minimum_replica_kernel_start_time(replica_kernels) for replica_kernels in kernel_data[1]]),
        2: min(
            [find_minimum_replica_kernel_start_time(replica_kernels) for replica_kernels in kernel_data[2]]),
        4: min(
            [find_minimum_replica_kernel_start_time(replica_kernels) for replica_kernels in kernel_data[4]]),
    }
    end_index_by_replica = {
        1: max(
            [find_maximum_replica_kernel_start_time(replica_kernels) for replica_kernels in kernel_data[1]]),
        2: max(
            [find_maximum_replica_kernel_start_time(replica_kernels) for replica_kernels in kernel_data[2]]),
        4: max(
            [find_maximum_replica_kernel_start_time(replica_kernels) for replica_kernels in kernel_data[4]]),
    }
    time_window_by_replica = {
        1: (4202808886.5, (4202808886.5 + 26267555.540625)),
        2: (4202808886.5, (4202808886.5 + 26267555.540625)),
        4: (4202808886.5 , (4202808886.5 + 26267555.540625))
    }

    # filter kernels outside of the determined window
    for replica_label, replica_values in kernel_data.items():
        for index_values, values in enumerate(replica_values):
            kernel_data[replica_label][index_values] = __cut_kernels_by_time(
                values,
                time_window_by_replica[replica_label][0],
                time_window_by_replica[replica_label][1]
            )

    # filter out experiment with 4 replicas
    del kernel_data[4]

    # extract and order number of replica configs
    replicas_labels = sorted(kernel_data.keys())

    # define figure
    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 3), constrained_layout=True, facecolor='white', sharey=True)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 13,
        'axes.titlesize': 13,
        'axes.labelsize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'lines.linewidth': 2.0,
        'mathtext.default': 'regular',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.4,
        'figure.figsize': (7, 5)  # Consistent size for single plot
    })

    replica_colors = ['#348ABD', '#E24A33']

    # plot
    for index_x, replica_label in enumerate(replicas_labels):
        kernel_values = kernel_data[replica_label]
        legend_patches = []
        for replica_index, replica_values in enumerate(kernel_values):
            legend_label = ''
            if replica_index == 0:
                legend_label = 'first replica'
            elif replica_index == 1:
                legend_label = 'second replica'
            legend_patches.append(patches.Patch(color=replica_colors[replica_index], label=legend_label, linewidth=0.1))
            for aux_index, (kernel_start, kernel_end) in enumerate(replica_values):
                level_index = 3
                print(len(replica_values) - aux_index)
                rect = patches.FancyBboxPatch(
                    (kernel_start, level_index),
                    kernel_end - kernel_start,
                    0.6,
                    boxstyle='round,pad=0.1',
                    edgecolor='black',
                    linewidth=0,
                    facecolor=replica_colors[replica_index],
                )
                axs[index_x].add_patch(rect)

        if index_x == 0:
            axs[index_x].set_ylabel('Kernel timeline')
        elif index_x == 1:
            axs[index_x].legend(handles=legend_patches, frameon=True)

        axs[index_x].set_ylim(3 - 0.6, 3 + 1.2)
        axs[index_x].set_xlim(time_window_by_replica[replica_label][0], time_window_by_replica[replica_label][1])
        axs[index_x].yaxis.set_ticks([])
        axs[index_x].yaxis.set_ticklabels([])

        import matplotlib.ticker as ticker
        def ns_to_sec(x, pos):
            return f"{x * 1e-6:.1f}"
        axs[index_x].xaxis.set_major_formatter(ticker.FuncFormatter(ns_to_sec))
        if index_x == 0:
            axs[index_x].set_xlabel(f'Time (ms) - {replica_label} replica')
        elif index_x == 1:
            axs[index_x].set_xlabel(f'Time (ms) - {replica_label} replicas')
    output_path = os.path.join(path, 'replication_timewise.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=400)



def main():
    if CREATE_SQLITE_DATABASES:
        for model in ['opt-1.3b']:
            create_sqlite_databases(model)

    model_results: List[Dict[str, Any]] = []
    if LOAD_PICKLE:
        with open(os.path.join(PICKLE_ROOT_PATH, 'replication_timewise'), 'rb') as file:
            model_results = pickle.load(file)
    else:
        for model in ['opt-1.3b']:
            model_results += extract_results(model, model)
        with open(os.path.join(PICKLE_ROOT_PATH, 'replication_timewise'), 'wb') as file:
            pickle.dump(model_results, file)

    plot_timewise(
        model_results,
        '.'
    )


if __name__ == '__main__':
    main()
