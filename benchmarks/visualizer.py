import argparse
import glob
import json
import os
import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_lines(lines, y_label, x_label, title, path, ylim=None, xlim=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    for line in lines:
        y, y_error, x, label = line
        ax.plot(x, y, label=label)
        ax.fill_between(x, y - y_error, y + y_error, alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
    if len(lines) < 10:
        ax.legend(loc='best', shadow=True)
    plt.savefig(path, bbox_inches='tight')


def main(output_path: str):
    time = np.load(os.path.join(output_path, 'time.npy'))
    processed_tokens_prompt = np.load(os.path.join(output_path, 'processed_tokens_prompt.npy'))
    processed_tokens_prompt_evolution = []
    for i in range(1, processed_tokens_prompt.shape[0]):
        processed_tokens_prompt_evolution.append(processed_tokens_prompt[i] - processed_tokens_prompt[i - 1])
    plot_lines(
        [
            (
                processed_tokens_prompt_evolution,
                np.zeros(len(processed_tokens_prompt_evolution)),
                time[:len(processed_tokens_prompt_evolution)],
                ''
            )
        ],
        'processed tokens prompt',
        'time (s)',
        f'Evolution of processed tokens prompt',
        os.path.join(output_path, 'processed_tokens_prompt.png')
    )
    processed_tokens_prompt = np.load(os.path.join(output_path, 'processed_tokens_token.npy'))
    processed_tokens_prompt_evolution = []
    for i in range(1, processed_tokens_prompt.shape[0]):
        processed_tokens_prompt_evolution.append(processed_tokens_prompt[i] - processed_tokens_prompt[i - 1])
    plot_lines(
        [
            (
                processed_tokens_prompt_evolution,
                np.zeros(len(processed_tokens_prompt_evolution)),
                time[:len(processed_tokens_prompt_evolution)],
                ''
            )
        ],
        'processed tokens generation',
        'time (s)',
        f'Evolution of processed tokens generation',
        os.path.join(output_path, 'processed_tokens_generation.png')
    )
    gpu_cache_usage_perc = np.load(os.path.join(output_path, 'gpu_cache_usage_perc.npy'))
    plot_lines(
        [
            (
                gpu_cache_usage_perc,
                np.zeros(len(gpu_cache_usage_perc)),
                time[:len(gpu_cache_usage_perc)],
                ''
            )
        ],
        'GPU usage (%)',
        'time (s)',
        f'GPU cache usage',
        os.path.join(output_path, 'gpu_cache_usage_perc.png')
    )
    num_running = np.load(os.path.join(output_path, 'num_running.npy'))
    num_waiting = np.load(os.path.join(output_path, 'num_waiting.npy'))
    num_preempted = np.load(os.path.join(output_path, 'num_preempted.npy'))
    num_finished = np.load(os.path.join(output_path, 'finished.npy'))
    plot_lines(
        [
            (
                num_running,
                np.zeros(len(num_running)),
                time[:len(num_running)],
                'running'
            ),
            (
                num_waiting,
                np.zeros(len(num_waiting)),
                time[:len(num_waiting)],
                'waiting'
            ),
            (
                num_preempted,
                np.zeros(len(num_preempted)),
                time[:len(num_preempted)],
                'preempted'
            ),
            (
                num_finished,
                np.zeros(len(num_finished)),
                time[:len(num_finished)],
                'finished'
            )
        ],
        '# requests',
        'time (s)',
        f'Evolution of requests',
        os.path.join(output_path, 'requests_evolution.png')
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualizer of other scripts\' outputs')
    parser.add_argument("--output-path", type=str, help="Path to output directory", required=True)
    args = parser.parse_args()
    main(args.output_path)
