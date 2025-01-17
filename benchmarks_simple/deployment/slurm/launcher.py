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
EXP_CONTAINER_CODE_DIR = os.getenv('EXP_CONTAINER_CODE_DIR', '/usr/local/lib/python3.12/dist-packages')

# path to Slurm executable
EXP_SLURM_EXECUTABLE = os.getenv('EXP_SLURM_EXECUTABLE', 'benchmarks_simple/deployment/slurm/slurm.sh')

# path to benchmark executable
EXP_BENCHMARK_EXECUTABLE = os.getenv('EXP_BENCHMARK_EXECUTABLE', 'benchmarks_simple/decoding_offline_profiler.py')

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
ILLEGAL_PARAMETERS = {}


def schedule_job(
    user: str,
    queue: str,
    specific_name: str,
    default_env_vars: dict,
    results_path: str,
    arguments: str,
    exp_max_duration: str,
    exclusive: bool,
    no_effect: bool,
    with_nsight: bool,
    nsight_args: str
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
    env["EXP_HOME_CODE_DIR"] = os.path.abspath(EXP_HOME_CODE_DIR)
    env["EXP_CONTAINER_CODE_DIR"] = EXP_CONTAINER_CODE_DIR
    env["EXP_CONTAINER_IMAGE"] = EXP_CONTAINER_IMAGE

    # running env vars
    str_env_vars: str = ''
    default_env_vars['PYTHONPATH'] = EXP_CONTAINER_CODE_DIR
    for key_env_var, value_env_var in default_env_vars.items():
        str_env_vars += f' --env {key_env_var}={value_env_var}'
    env["EXP_ENV_VARS"] = str_env_vars

    # define command
    command = f'python3 {EXP_BENCHMARK_EXECUTABLE} {arguments}'
    if with_nsight:
        command = f'nsys profile --output {exp_results_path}/ {nsight_args} ' + command
    env["EXP_BENCHMARK_COMMAND"] = command

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
        default_args: dict,
        test_args: dict,
        default_env_vars: dict,
        exp_max_duration: str,
        exclusive: bool,
        no_effect: bool,
        with_nsight: bool,
        nsight_args: str
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
    default_args = ' '.join([f"{arg}={default_args[arg]}" if default_args[arg] != '' else arg for arg in default_args.keys()])
    combination_names, args_combinations = rec_select_args_combination('', default_args, test_args)

    for server_index, args_combination in enumerate(args_combinations):
        combination_name = combination_names[server_index]
        print(f'Name -> {combination_name}. Args -> {args_combination}')

        arguments = f"--result-dir='{os.path.join(results_path, combination_name)}' {args_combination}"

        schedule_job(
            user,
            queue,
            combination_name,
            default_env_vars,
            results_path,
            arguments,
            exp_max_duration,
            exclusive,
            no_effect,
            with_nsight,
            nsight_args
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launcher of vllm benchmarking experiments on Kubernetes')
    parser.add_argument('--user', type=str, help='Slurm user', required=True)
    parser.add_argument('--queue', type=str, help='Slurm queue', required=True)
    parser.add_argument('--results-path', type=str, help='Path to store results', required=True)
    parser.add_argument('--exclusive', action='store_true', default=False, help='Run the experiments in exclusive mode')
    parser.add_argument('--max-duration', type=str, default='00:05:00', help='Slurm queue')
    parser.add_argument('--default-args', type=str, help='Dictionary with the default args')
    parser.add_argument('--test-args', type=str, help='Dictionary with the args to test against')
    parser.add_argument('--no-effect', action='store_true', help='Do everything except the step of launching the experiment')
    parser.add_argument('--default-env-vars', type=str, help='Dictionary with the default env vars')
    parser.add_argument('--with-nsight', default=False, action='store_true', help='Launch with nsight profile')
    parser.add_argument('--nsight-args', default='', type=str, help='Additional nsight arguments')
    args = parser.parse_args()

    default_args = json.loads(args.default_args.replace('\'', '"'))
    test_args = json.loads(args.test_args.replace('\'', '"'))
    if args.default_env_vars is not None:
        default_env_vars = json.loads(args.default_env_vars.replace('\'', '"'))
    else:
        default_env_vars = {}

    os.makedirs(args.results_path, exist_ok=True)
    config_path = os.path.join(args.results_path, f'config-{str(random.randint(0, 100000))}.txt')
    with open(config_path, 'w') as config_file:
        config = f'EXP_CONTAINER_IMAGE={EXP_CONTAINER_IMAGE} PYTHONPATH=. python3 ' + ' '.join(sys.argv).replace('{', '"{').replace('}', '}"') + '\n'
        config_file.write(config)

    main(
        args.user,
        args.queue,
        args.results_path,
        default_args,
        test_args,
        default_env_vars,
        args.max_duration,
        args.exclusive,
        args.no_effect,
        args.with_nsight,
        args.nsight_args
    )
