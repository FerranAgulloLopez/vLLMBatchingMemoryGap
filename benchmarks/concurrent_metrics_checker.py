import os
import re
import signal
import sys
import time
from typing import List, Tuple, Dict
import subprocess
from multiprocessing import Process
import requests


GPU_METRICS_NVIDIA_SMI = \
    [
        'uuid',
        'clocks_event_reasons.hw_slowdown',
        'clocks_event_reasons.sw_thermal_slowdown',
        'memory.total',
        'memory.reserved',
        'memory.used',
        'memory.free',
        'utilization.gpu',
        'utilization.memory',
    ]

GPU_METRICS_NVIDIA_SMI_DOM_SELECT = \
    [
        'p',
        'u',
        'c',
        'v',
        't',
    ]

GPU_METRICS_NVIDIA_SMI_DOM_GPM = \
    [

    ]

PROMETHEUS_METRICS = \
    [
        'scheduler_total_time',
        'model_forward_total_time',
        'gpu_cache_usage_perc',
        'num_requests_running',
        'num_requests_waiting',
        'num_preemptions_total',
        'request_success_total',
        'prompt_tokens_total',
        'generation_tokens_total',
    ]

WRITING_RATIO = 20


class ConcurrentMetricsChecker(Process):

    def __init__(self, output_path: str, metrics_api_url: str):
        global GPU_METRICS_NVIDIA_SMI, GPU_METRICS_NVIDIA_SMI_DOM_SELECT, GPU_METRICS_NVIDIA_SMI_DOM_GPM
        super(ConcurrentMetricsChecker, self).__init__()
        self.output_path: str = output_path
        self.metrics_api_url: str = metrics_api_url

        self.gpu_metrics_command_nvidia_smi: str = f'nvidia-smi --format=csv'
        if len(GPU_METRICS_NVIDIA_SMI) > 0:
            self.gpu_metrics_command_nvidia_smi += f' --query-gpu={",".join(GPU_METRICS_NVIDIA_SMI)}'
        self.gpu_metrics_command_nvidia_smi_dom: str = f'nvidia-smi dmon --count=1'
        if len(GPU_METRICS_NVIDIA_SMI_DOM_SELECT) > 0:
            self.gpu_metrics_command_nvidia_smi_dom += f' --select={"".join(GPU_METRICS_NVIDIA_SMI_DOM_SELECT)}'
        if len(GPU_METRICS_NVIDIA_SMI_DOM_GPM) > 0:
            self.gpu_metrics_command_nvidia_smi_dom += f' --gpm-metrics={",".join(GPU_METRICS_NVIDIA_SMI_DOM_GPM)}'

        self.rows_to_write: Dict[str, Tuple[object, List[str]]] = {}

        signal.signal(signal.SIGTERM, self.__signal_term_handler)

        self.cache_config = requests.get(self.metrics_api_url).text
        with open(os.path.join(self.output_path, f'cache_config.txt'), 'w') as text_file:
            text_file.write(self.cache_config)

    def run(self):
        global WRITING_RATIO, PROMETHEUS_METRICS
        start_time: float = time.perf_counter()

        try:
            # initialize engine metrics
            _id = 'metrics_engine'
            row: str = ','.join(PROMETHEUS_METRICS) + '\n'
            open_file = open(os.path.join(self.output_path, f'{_id}.csv'), 'a')
            self.rows_to_write[_id] = (
                open_file,
                ['time [s],' + row]
            )

            # initialize GPU metrics nvidia-smi
            rows: List[str] = self.__run_subprocess(self.gpu_metrics_command_nvidia_smi)
            number_gpu_devices: int = len(rows) - 1
            for gpu_device in range(number_gpu_devices):
                _id = f'metrics_gpu_device_nvidia_smi_{gpu_device}'
                open_file = open(os.path.join(self.output_path, f'{_id}.csv'), 'a')
                self.rows_to_write[_id] = (
                        open_file,
                        ['time [s],' + rows[0] + '\n']
                )

            # initialize GPU metrics nvidia-smi dom
            rows: List[str] = self.__run_subprocess(self.gpu_metrics_command_nvidia_smi_dom)
            for gpu_device in range(number_gpu_devices):
                _id = f'metrics_gpu_device_nvidia_smi_dom_{gpu_device}'
                open_file = open(os.path.join(self.output_path, f'{_id}.csv'), 'a')
                row_header_0: str = ' '.join(rows[0].replace('# ', '').split()).replace(' ', ',')
                row_header_1: str = ' '.join(rows[1].replace('# ', '').split()).replace(' ', ',')
                row_header_0.replace('# ', '')
                self.rows_to_write[_id] = (
                    open_file,
                    ['time,' + row_header_0 + '\n'] + ['s,' + row_header_1 + '\n']
                )

            # extract metrics in loop
            iterations: int = 0
            while True:
                start_iteration_time: float = time.perf_counter()
                try:
                    timestamp: float = time.perf_counter() - start_time

                    # extract engine metrics
                    _id = 'metrics_engine'
                    metrics_response: str = requests.get(self.metrics_api_url).text
                    row: List[str] = []
                    for metric_key in PROMETHEUS_METRICS:
                        metric_value: float = self.__find_prometheus_metric_value(
                            metric_key,
                            metrics_response
                        )
                        row.append(str(metric_value))
                    self.rows_to_write[_id][1].append(f'{timestamp},' + ','.join(row) + '\n')

                    # extract GPU metrics nvidia-smi
                    rows: List[str] = self.__run_subprocess(self.gpu_metrics_command_nvidia_smi)
                    for gpu_device in range(number_gpu_devices):
                        _id = f'metrics_gpu_device_nvidia_smi_{gpu_device}'
                        self.rows_to_write[_id][1].append(f'{timestamp},' + rows[gpu_device + 1] + '\n')

                    # extract GPU metrics nvidia-smi dom
                    rows: List[str] = self.__run_subprocess(self.gpu_metrics_command_nvidia_smi_dom)
                    for gpu_device in range(number_gpu_devices):
                        _id = f'metrics_gpu_device_nvidia_smi_dom_{gpu_device}'
                        row_values = ' '.join(rows[gpu_device + 2].split()).replace(' ', ',')
                        self.rows_to_write[_id][1].append(f'{timestamp},' + row_values + '\n')

                    # write metrics
                    iterations += 1
                    if iterations >= WRITING_RATIO:
                        self.__write_metrics()
                        iterations = 0
                except Exception as e:
                    print(f'Error while running usage checker: {e}')
                finally:
                    iteration_time: float = time.perf_counter() - start_iteration_time
                    if iteration_time < 1:
                        time.sleep(1 - iteration_time)

        except Exception as e:
            self.__close_files()
            raise e

    def __run_subprocess(self, command: str) -> List[str]:
        output = subprocess.check_output(command.split(), stderr=subprocess.STDOUT)
        return output.decode('ascii').split('\n')[:-1]

    def __find_prometheus_metric_value(self, label: str, metrics_response: str) -> float:
        try:
            pattern = f"vllm:{label}(.+?) ([+-]?([0-9]*[.])?[0-9]+)\n"
            value = re.search(pattern, metrics_response).group(2)
        except:
            # print(f'Pattern error with label {label}')
            return 0
        return float(value)

    def __signal_term_handler(self, signal, frame) -> None:
        self.__write_metrics()
        self.__close_files()
        sys.exit(0)

    def __write_metrics(self) -> None:
        for key, (file_open, rows) in self.rows_to_write.items():
            if len(rows) > 0:
                file_open.writelines(rows)
                self.rows_to_write[key] = (file_open, [])

    def __close_files(self) -> None:
        for file_open, _ in self.rows_to_write.values():
            if file_open is not None and not file_open.closed:
                file_open.close()
