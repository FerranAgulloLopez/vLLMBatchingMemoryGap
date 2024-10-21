import os
import re
import signal
import sys
import time
from multiprocessing import Process

import numpy as np
import requests


def find_prometheus_metric_value(label: str, metrics_response: str):
    try:
        pattern = f"vllm:{label}(.+?) ([+-]?([0-9]*[.])?[0-9]+)\n"
        value = re.search(pattern, metrics_response).group(2)
    except:
        # print(f'Pattern error with label {label}')
        return 0
    return float(value)


class ConcurrentMetricsChecker(Process):

    def __init__(self, output_path: str, metrics_api_url: str):
        super(ConcurrentMetricsChecker, self).__init__()
        self.output_path = output_path
        self.metrics_api_url = metrics_api_url
        signal.signal(signal.SIGTERM, self.__signal_term_handler)

        self.time = []
        self.gpu_cache_usage_perc = []
        self.num_running = []
        self.num_waiting = []
        self.num_preempted = []
        self.finished = []
        self.processed_tokens_prompt = []
        self.processed_tokens_generation = []
        self.running_by_adapter = {}
        self.waiting_by_adapter = {}

        self.cache_config = requests.get(self.metrics_api_url).text

    def run(self):
        start_time = time.perf_counter()
        while True:
            try:
                self.time.append(time.perf_counter() - start_time)
                metrics_response = requests.get(self.metrics_api_url).text

                self.gpu_cache_usage_perc.append(
                    find_prometheus_metric_value(
                        f"gpu_cache_usage_perc",
                        metrics_response
                    )
                )

                self.num_running.append(
                    find_prometheus_metric_value(
                        f"num_requests_running",
                        metrics_response
                    )
                )

                self.num_waiting.append(
                    find_prometheus_metric_value(
                        f"num_requests_waiting",
                        metrics_response
                    )
                )

                self.num_preempted.append(
                    find_prometheus_metric_value(
                        f"num_preemptions_total",
                        metrics_response
                    )
                )

                self.finished.append(
                    find_prometheus_metric_value(
                        f"request_success_total",
                        metrics_response
                    )
                )

                self.processed_tokens_prompt.append(
                    find_prometheus_metric_value(
                        f"prompt_tokens_total",
                        metrics_response
                    )
                )
                self.processed_tokens_generation.append(
                    find_prometheus_metric_value(
                        f"generation_tokens_total",
                        metrics_response
                    )
                )

            except Exception as e:
                print(f'Error while running usage checker: {e}')
            finally:
                time.sleep(1)

    def __signal_term_handler(self, signal, frame):
        self.__save_metrics()
        sys.exit(0)

    def __save_metrics(self):
        with open(os.path.join(self.output_path, f'cache_config.txt'), "w") as text_file:
            text_file.write(self.cache_config)
        np.save(
            os.path.join(self.output_path, f'time'),
            np.asarray(self.time)
        )
        np.save(
            os.path.join(self.output_path, f'gpu_cache_usage_perc'),
            np.asarray(self.gpu_cache_usage_perc)
        )
        np.save(
            os.path.join(self.output_path, f'num_running'),
            np.asarray(self.num_running)
        )
        np.save(
            os.path.join(self.output_path, f'num_waiting'),
            np.asarray(self.num_waiting)
        )
        np.save(
            os.path.join(self.output_path, f'num_preempted'),
            np.asarray(self.num_preempted)
        )
        np.save(
            os.path.join(self.output_path, f'finished'),
            np.asarray(self.finished)
        )
        np.save(
            os.path.join(self.output_path, f'processed_tokens_prompt'),
            np.asarray(self.processed_tokens_prompt)
        )
        np.save(
            os.path.join(self.output_path, f'processed_tokens_token'),
            np.asarray(self.processed_tokens_generation)
        )
