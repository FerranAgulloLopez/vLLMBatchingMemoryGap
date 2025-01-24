import os
import json
from typing import List, Tuple, Dict, Set
import matplotlib.pyplot as plt
import numpy as np

CUDA_GRAPH_RUNNER: Dict[str, Set] = {
    'matrix_multiplications': {
        'sm90_xmma_gemm',
        'sm80_xmma_gemm',
    },
    'mem_transfers': {
        'Memcpy DtoD (Device -> Device)',
        'Memset (Unknown)',
        'memcpy128',
    },
    'vllm': {
        'void vllm::rms_norm_kernel',
        'void vllm::rotary_embedding_kernel',
        'void vllm::reshape_and_cache_kernel',
        'void vllm::paged_attention_v1_kernel',
        'void vllm::act_and_mul_kernel',
    },
}


def extract_experiment_metric(path: str) -> Dict[str, float]:
    output: Dict[str, float] = {}
    with open(os.path.join(path, 'time.json')) as metrics_file:
        times: dict = json.load(metrics_file)

    output_length: int = int(times['context']['output_len'])
    last_decode_step: str = f'decode_{output_length - 1}'
    cuda_graph_runner: dict = times[last_decode_step]['model_stats'][0]
    assert cuda_graph_runner['entry']['name'] == 'CUDAGraphRunner'

    # extract total cuda graph runner cuda time from the last decode step
    output['cuda_graph_runner_total_cuda_time'] = float(cuda_graph_runner['entry']['cuda_time_us'])

    # extract specific times from cuda graph runner and the last decode step
    output['cuda_graph_runner_matrix_multiplications_cuda_time'] = 0
    output['cuda_graph_runner_mem_transfers_cuda_time'] = 0
    output['cuda_graph_runner_vllm_cuda_time'] = 0
    # TODO refactor!!!
    for item in cuda_graph_runner['children']:
        item_name: str = item['entry']['name']
        cuda_time_us: float = item['entry']['cuda_time_us']
        breaked: bool = False
        for selector in CUDA_GRAPH_RUNNER['matrix_multiplications']:
            if selector in item_name:
                output['cuda_graph_runner_matrix_multiplications_cuda_time'] += cuda_time_us
                breaked = True
                break
        if breaked:
            continue
        for selector in CUDA_GRAPH_RUNNER['mem_transfers']:
            if selector in item_name:
                output['cuda_graph_runner_mem_transfers_cuda_time'] += cuda_time_us
                breaked = True
                break
        if breaked:
            continue
        for selector in CUDA_GRAPH_RUNNER['vllm']:
            if selector in item_name:
                output['cuda_graph_runner_vllm_cuda_time'] += cuda_time_us
                breaked = True
                break

    return output


def extract_results(path: str) -> List[Dict[str, float]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['prompt_len', 'output_len', 'batch_size']
    results = []
    subdir, dirs, files = next(os.walk(path))
    for folder in dirs:
        prompt_len: int = int(folder.split('_')[1])
        output_len: int = int(folder.split('_')[2])
        batch_size: int = int(folder.split('_')[3])
        try:
            metrics = extract_experiment_metric(os.path.join(path, folder))
        except FileNotFoundError:
            metrics = {}
        metrics['prompt_len'] = prompt_len
        metrics['output_len'] = output_len
        metrics['batch_size'] = batch_size
        _id = create_id(metrics, id_metrics)
        if _id in collected_ids:
            raise ValueError('Repeated results')
        collected_ids.add(_id)
        results.append(metrics)
    return results


def plot_results(
        results: List[Dict[str, float]],
        path: str
) -> None:
    def __prepare_lines(results: List[Dict[str, float]], x_axis: str, y_axis: str, selection: str) -> List[Tuple[str, List[int], List[float]]]:
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
            x_line = [x_value for index, x_value in enumerate(x_values) if x_value is not None and y_values[index] is not None]
            y_line = [y_value for index, y_value in enumerate(y_values) if y_value is not None and x_values[index] is not None]
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

    results_total_time = __prepare_lines(
        results,
        'batch_size',
        'cuda_graph_runner_total_cuda_time',
        'output_len'
    )

    nrows = 1
    ncols = 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4), sharex=True)
    fig.subplots_adjust(wspace=0)

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    for (key, x_line, y_line) in results_total_time:
        line = axs.plot(
            x_line,
            y_line,
            marker='o',
            linestyle='solid',
            label=key
        )
    axs.set_ylabel('time (us)', fontsize=10)
    axs.legend(loc='upper right', fontsize=12)
    axs.set_xlabel('batch size (r)')
    axs.set_title('Total cuda time evolution with batch size')

    plt.savefig(os.path.join(path, f'total_cuda_time'), bbox_inches='tight')

    nrows = 1
    ncols = 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4), sharex=True)
    fig.subplots_adjust(wspace=0)

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    for (key, x_line, y_line) in results_total_time:
        y_line_speedup = [item / y_line[0] for item in y_line]
        line = axs.plot(
            x_line,
            y_line_speedup,
            marker='o',
            linestyle='solid',
            label='real'
        )
        y_line_optimal = [1] * len(y_line)
        line = axs.plot(
            x_line,
            y_line_optimal,
            marker='o',
            linestyle='solid',
            label=f'optimal'
        )
        line = axs.plot(
            x_line,
            x_line,
            marker='o',
            linestyle='solid',
            label=f'worst'
        )
    axs.set_ylabel('time (us)', fontsize=10)
    axs.legend(loc='upper right', fontsize=12)
    axs.set_xlabel('batch size (r)')
    axs.set_ylim(0, 10)
    axs.set_title('Speedup total cuda time evolution with batch size')

    plt.savefig(os.path.join(path, f'total_cuda_time_speedup'), bbox_inches='tight')

    results_matrix_multiplications = __prepare_lines(
        results,
        'batch_size',
        'cuda_graph_runner_matrix_multiplications_cuda_time',
        'output_len'
    )
    results_mem_transfers = __prepare_lines(
        results,
        'batch_size',
        'cuda_graph_runner_mem_transfers_cuda_time',
        'output_len'
    )
    results_vllm = __prepare_lines(
        results,
        'batch_size',
        'cuda_graph_runner_vllm_cuda_time',
        'output_len'
    )

    fig, ax = plt.subplots(layout='constrained', figsize=(9, 6))
    width = 0.15  # the width of the bars
    multiplier = 0
    x_line_labels = results_total_time[0][1]
    x_line = np.arange(len(results_total_time[0][1]))
    for label, y_line in {
        'total CUDA time': results_total_time[0][2],
        'Multiplications time': results_matrix_multiplications[0][2],
        'vLLM stuff': results_vllm[0][2],
        'Memory transfer time': results_mem_transfers[0][2],
    }.items():
        offset = width * multiplier
        rects = ax.bar(x_line + offset, y_line, width, label=label)
        # ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_ylabel('time (us)')
    ax.set_xlabel('batch size (r)')
    ax.set_title('Cuda time evolution split by types')
    ax.set_xticks(x_line + width, x_line_labels)
    ax.legend(loc='upper left', ncols=3)

    plt.savefig(os.path.join(path, f'cuda_time_types'), bbox_inches='tight')



def main():
    results: List[Dict[str, float]] = extract_results('llama-2-7b')
    plot_results(results, '.')


if __name__ == '__main__':
    main()
