from __future__ import print_function
import argparse
import json
import os.path
import random
import subprocess
import sys
from copy import deepcopy
from typing import List
import os


# path to root code directory in host and container
EXP_HOME_CODE_DIR = os.getenv('EXP_HOME_CODE_DIR', '.')
EXP_CONTAINER_CODE_DIR = os.getenv('EXP_CONTAINER_CODE_DIR', '/usr/local/lib/python3.10/dist-packages')

# path to Slurm executable
EXP_SLURM_EXECUTABLE = os.getenv('EXP_SLURM_EXECUTABLE', 'benchmarks/deployment/slurm/slurm.sh')

# path to benchmark executable
EXP_BENCHMARK_EXECUTABLE = os.getenv('EXP_BENCHMARK_EXECUTABLE', 'benchmarks/benchmark_serving_with_metrics.py')

# path to container image
EXP_CONTAINER_IMAGE = os.getenv('EXP_CONTAINER_IMAGE')
if EXP_CONTAINER_IMAGE is None:
    raise ValueError('Environment variable EXP_CONTAINER_IMAGE not specified')

# print ENV vars
print('Environment variable values:')
print('EXP_HOME_CODE_DIR:', EXP_HOME_CODE_DIR)
print('EXP_CONTAINER_CODE_DIR:', EXP_CONTAINER_CODE_DIR)
print('EXP_SLURM_EXECUTABLE:', EXP_SLURM_EXECUTABLE)
print('EXP_BENCHMARK_EXECUTABLE:', EXP_BENCHMARK_EXECUTABLE)
print('EXP_CONTAINER_IMAGE:', EXP_CONTAINER_IMAGE)
print('\n\n')

# parameters that cannot be modified (it could make the Job stop working)
ILLEGAL_PARAMETERS = {
    '--launch-server',
    '--server-args',
    '--results-path',
    '--result-dir',
    '--host',
    '--port'
}


def schedule_job(
    user: str,
    queue: str,
    specific_name: str,
    results_path: str,
    arguments: str,
    exp_max_duration: str,
    exclusive: bool,
    no_effect: bool
) -> None:
    global \
        EXP_HOME_CODE_DIR, \
        EXP_CONTAINER_CODE_DIR, \
        EXP_SLURM_EXECUTABLE, \
        EXP_CONTAINER_IMAGE, \
        EXP_BENCHMARK_EXECUTABLE, \
        EXP_SLURM_EXECUTABLE

    exp_results_path = os.path.join(results_path, specific_name)
    os.makedirs(exp_results_path, exist_ok=True)

    env = os.environ.copy()
    env["EXP_NAME"] = specific_name
    env["EXP_MAX_DURATION_SECONDS"] = exp_max_duration
    env["EXP_RESULTS_PATH"] = exp_results_path
    env["EXP_ARGS"] = arguments
    env["EXP_HOME_CODE_DIR"] = os.path.abspath(EXP_HOME_CODE_DIR)
    env["EXP_CONTAINER_CODE_DIR"] = EXP_CONTAINER_CODE_DIR
    env["EXP_BENCHMARK_EXECUTABLE"] = EXP_BENCHMARK_EXECUTABLE
    env["EXP_CONTAINER_IMAGE"] = EXP_CONTAINER_IMAGE

    command = f'cat {EXP_SLURM_EXECUTABLE} | envsubst > {exp_results_path}/launcher.sh'
    subprocess.run(command, env=env, shell=True)

    if exclusive:
        command = f'sbatch -A {user} -q {queue} --exclusive {exp_results_path}/launcher.sh'
    else:
        command = f'sbatch -A {user} -q {queue} {exp_results_path}/launcher.sh'
    if not no_effect:
        subprocess.run(command, shell=True)


def rec_select_args_combination(combination_name: str, current_combination: str, left_args: dict) -> (List[str], List[str]):
    if len(left_args) == 0:
        return [combination_name], [current_combination]
    left_args = deepcopy(left_args)
    selected_arg_key = next(iter(left_args))
    selected_arg_values = left_args.pop(selected_arg_key)
    left_combination_names = []
    left_combinations = []
    for arg_value in selected_arg_values:
        arg_value_name: str = arg_value
        if '/' in arg_value_name:
            arg_value_name = arg_value_name.split('/')[-1]
        output_names, output_args = rec_select_args_combination(f'{combination_name}_{arg_value_name}', f"{current_combination} {selected_arg_key}='{arg_value}'", left_args)
        left_combination_names += output_names
        left_combinations += output_args
    return left_combination_names, left_combinations


def main(
        user: str,
        queue: str,
        results_path: str,
        default_server_args: dict,
        default_benchmark_args: dict,
        test_server_args: dict,
        test_benchmark_args: dict,
        exp_max_duration: str,
        exclusive: bool,
        no_effect: bool
) -> None:
    global ILLEGAL_PARAMETERS

    # TODO uncomment asserts and test
    # assert any(arg in ILLEGAL_PARAMETERS for arg in default_server_args.keys()), f'The default server args cannot contain the following parameters: {ILLEGAL_PARAMETERS}'
    # assert any(arg in ILLEGAL_PARAMETERS for arg in default_benchmark_args.keys()), f'The default benchmark args cannot contain the following parameters: {ILLEGAL_PARAMETERS}'
    # assert any(arg in ILLEGAL_PARAMETERS for arg in test_server_args.keys()), f'The test server args cannot contain the following parameters: {ILLEGAL_PARAMETERS}'
    # assert any(arg in ILLEGAL_PARAMETERS for arg in test_benchmark_args.keys()), f'The test benchmark args cannot contain the following parameters: {ILLEGAL_PARAMETERS}'

    # assert any(arg in default_server_args for arg in test_server_args.keys()), f'The server args to test cannot be included as well in its default args'
    # assert any(arg in default_benchmark_args for arg in test_benchmark_args.keys()), f'The benchmark args to test cannot be included as well in its default args'

    # TODO assert that default dicts do not contain more than one value per item
    default_server_args = ' '.join([f"{arg}={default_server_args[arg]}" if default_server_args[arg] != '' else arg for arg in default_server_args.keys()])
    default_benchmark_args = ' '.join([f"{arg}='{default_benchmark_args[arg]}'" if default_benchmark_args[arg] != '' else arg for arg in default_benchmark_args.keys()])

    server_combination_names, server_args_combinations = rec_select_args_combination('', default_server_args, test_server_args)
    benchmark_combination_names, benchmark_args_combinations = rec_select_args_combination('', default_benchmark_args, test_benchmark_args)

    for server_index, server_args_combination in enumerate(server_args_combinations):
        for benchmark_index, benchmark_args_combination in enumerate(benchmark_args_combinations):
            combination_name = f'{server_combination_names[server_index]}_{benchmark_combination_names[benchmark_index]}'
            print(f'Name -> {combination_name}. Server Args -> {server_args_combination}. Benchmark Args -> {benchmark_args_combination}')

            random_port = random.randint(1024, 65535)
            arguments = f"--port='{random_port}' --result-dir='{os.path.join(results_path, combination_name)}' {benchmark_args_combination} --launch-server --server-args='--port={random_port} {server_args_combination}'"

            schedule_job(
                user,
                queue,
                combination_name,
                results_path,
                arguments,
                exp_max_duration,
                exclusive,
                no_effect
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launcher of vllm benchmarking experiments on Kubernetes')
    parser.add_argument('--user', type=str, help='Slurm user', required=True)
    parser.add_argument('--queue', type=str, help='Slurm queue', required=True)
    parser.add_argument('--results-path', type=str, help='Path to store results', required=True)
    parser.add_argument('--exclusive', action='store_true', default=False, help='Run the experiments in exclusive mode')
    parser.add_argument('--max-duration', type=str, default='00:05:00', help='Slurm queue')
    parser.add_argument('--default-server-args', type=str, help='Dictionary with the default vllm server args')
    parser.add_argument('--default-benchmark-args', type=str, help='Dictionary with the default benchmark args')
    parser.add_argument('--test-server-args', type=str, help='Dictionary with the vllm server args to test against')
    parser.add_argument('--test-benchmark-args', type=str, help='Dictionary with the benchmark args to test against')
    parser.add_argument('--no-effect', action='store_true', help='Do everything except the step of launching the experiment')
    args = parser.parse_args()

    default_server_args = json.loads(args.default_server_args.replace('\'', '"'))
    default_benchmark_args = json.loads(args.default_benchmark_args.replace('\'', '"'))
    test_server_args = json.loads(args.test_server_args.replace('\'', '"'))
    test_benchmark_args = json.loads(args.test_benchmark_args.replace('\'', '"'))

    os.makedirs(args.results_path, exist_ok=True)
    config_path = os.path.join(args.results_path, f'config-{str(random.randint(0, 100000))}.txt')
    with open(config_path, 'w') as config_file:
        config = 'PYTHONPATH=. python3 ' + ' '.join(sys.argv).replace('{', '"{').replace('}', '}"') + '\n'
        config_file.write(config)

    main(
        args.user,
        args.queue,
        args.results_path,
        default_server_args,
        default_benchmark_args,
        test_server_args,
        test_benchmark_args,
        args.max_duration,
        args.exclusive,
        args.no_effect
    )
