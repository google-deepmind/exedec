# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Measure compositional generalization ability of LLMs.

From the `exedec` directory:

python spec_decomposition/run_llm_experiment.py \
    --model=favorite_llm --prompt_format=exedec
"""

import json
import multiprocessing.dummy
import os
import sys
import timeit
from typing import Any, Callable

from absl import app
from absl import flags
import tensorflow as tf
import tqdm

sys.path.append('../')
# pylint: disable=g-import-not-at-top

from exedec.spec_decomposition import cached_llm_access
from exedec.spec_decomposition import llm_utils

_MODEL = flags.DEFINE_string(
    'model', '',
    'Which model to use.')
_PROMPT_FORMAT = flags.DEFINE_enum(
    'prompt_format',
    'baseline',
    ['baseline', 'exedec', 'exedec_ablation'],
    'Format of the prompt to use.',
)
_TARGET_TASK = flags.DEFINE_enum(
    'task',
    None,
    [
        'NONE',
        'LENGTH_GENERALIZATION',
        'COMPOSE_DIFFERENT_CONCEPTS',
        'SWITCH_CONCEPT_ORDER',
        'COMPOSE_NEW_OP',
        'ADD_OP_FUNCTIONALITY',
    ],
    'If specified, only run one generalization task.',
)

_NUM_SAMPLES = flags.DEFINE_integer(
    'num_samples', 1,
    'Number of samples to draw for one problem.')
_TEMPERATURE = flags.DEFINE_float(
    'temperature', 0.0, 'Temperature for the LLM.'
)
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers', 72, 'Number of workers.',
)

_LLM_CACHE_DIR = flags.DEFINE_string(
    'llm_cache_dir', '~/llm_cache',
    'Directory for storing the LLM cache.')

# The ICLR'24 paper used version_robustfill=1, version_deepcoder=1, and
# version_deepcoder=4 (DeepCoder-Pythonic in the paper).
_VERSION_DEEPCODER = flags.DEFINE_integer(
    'version_deepcoder', 1, 'Version of Python programs and prompts to use.'
)
_VERSION_ROBUSTFILL = flags.DEFINE_integer(
    'version_robustfill', 1, 'Version of Python programs and prompts to use.'
)

DATA_FORMAT = 'data/llm_data/{dataset_type}/{generalization_task}.jsonl'
RESULTS_FORMAT = os.path.join(
    os.path.expanduser('~/exedec_results'),
    '{prompt_format}_{model}_{num_samples}-samples_{temperature}-temperature_deepcoder-v{version_deepcoder}_robustfill-v{version_robustfill}.json')
MAX_NUM_DEC_STEPS = 3

DatasetElement = llm_utils.DatasetElement
ExeDecTrajectory = llm_utils.ExeDecTrajectory


def _sample_length(dataset_type: str) -> int:
  """Returns the maximum number of decode steps for a given dataset type."""
  if _PROMPT_FORMAT.value == 'baseline':
    if dataset_type == 'deepcoder':
      return 150
    elif dataset_type == 'robustfill':
      return 400
    else:
      raise ValueError(f'Unhandled dataset type: {dataset_type}')
  elif _PROMPT_FORMAT.value == 'exedec_ablation':
    if dataset_type == 'deepcoder':
      return 80
    elif dataset_type == 'robustfill':
      return 200
    else:
      raise ValueError(f'Unhandled dataset type: {dataset_type}')
  elif _PROMPT_FORMAT.value == 'exedec':
    if dataset_type == 'deepcoder':
      return 200
    elif dataset_type == 'robustfill':
      return 400
    else:
      raise ValueError(f'Unhandled dataset type: {dataset_type}')
  else:
    raise ValueError(f'Unhandled prompt format: {_PROMPT_FORMAT.value}')


def query_llm(
    prompt: str,
    n: int,
    temperature: float,
    model: str,
    num_output_tokens: int) -> list[str]:
  """Queries an LLM with the given prompt, drawing n samples."""
  del prompt, n, temperature, model, num_output_tokens
  raise NotImplementedError('Call your favorite LLM here.')


def solve_problem_baseline(
    problem_index: int,
    few_shot_examples: list[DatasetElement],
    test_problem: DatasetElement,
    dataset_type: str,
    num_output_tokens: int,
    verbose: bool = False,
    ablation_style: bool = False,
) -> dict[str, Any]:
  """Solve a problem with baseline prompt."""
  del ablation_style
  start_time = timeit.default_timer()
  if dataset_type == 'robustfill':
    version = _VERSION_ROBUSTFILL.value
  elif dataset_type == 'deepcoder':
    version = _VERSION_DEEPCODER.value
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')
  prompt = llm_utils.few_shot_prompt(
      few_shot_examples,
      test_problem,
      dataset_type=dataset_type,
      version=version,
  )
  samples = cached_llm_access.query_llm(
      query_llm,
      prompt,
      n=_NUM_SAMPLES.value,
      temperature=_TEMPERATURE.value,
      model=_MODEL.value,
      num_output_tokens=num_output_tokens,
  )

  success = False
  for sample in samples:
    sample = llm_utils.cut_program_from_sample(sample)
    try:
      outputs = llm_utils.run_program(
          sample, test_problem.inputs, dataset_type=dataset_type
      )
    except Exception:  # pylint: disable=broad-exception-caught
      outputs = None
    if outputs == test_problem.outputs:
      success = True
      break

  elapsed_time = timeit.default_timer() - start_time
  result = {
      'index': problem_index,
      'test_problem': test_problem,
      'samples': samples,
      'success': success,
      'elapsed_time': elapsed_time,
  }
  if verbose:
    print(
        f'  Test problem #{problem_index}: '
        f'{"SUCCESS" if result["success"] else "fail"}',
        flush=True,
    )
  return result


def solve_problem_exedec(
    problem_index: int,
    few_shot_examples: list[DatasetElement],
    test_problem: DatasetElement,
    dataset_type: str,
    num_output_tokens: int,
    verbose: bool = False,
    ablation_style: bool = False,
) -> dict[str, Any]:
  """Solve a problem with ExeDec prompt."""
  start_time = timeit.default_timer()

  samples = []
  trajectories = []
  success = False
  if dataset_type == 'robustfill':
    version = _VERSION_ROBUSTFILL.value
  elif dataset_type == 'deepcoder':
    version = _VERSION_DEEPCODER.value
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')
  for _ in range(_NUM_SAMPLES.value):
    test_problem_wo_solution = DatasetElement(
        test_problem.inputs, test_problem.outputs, None, None
    )
    trajectory = []
    for i in range(MAX_NUM_DEC_STEPS):
      try:
        prompt = llm_utils.few_shot_exe_dec_prompt(
            few_shot_examples,
            test_problem_wo_solution,
            dataset_type=dataset_type,
            version=version,
            ablation_style=ablation_style,
        )
      except Exception:  # pylint: disable=broad-exception-caught
        # Throws error if the previous step does not match target string in
        # RobustFill or any other runtime error during program execution.
        # print(e)
        break
      if i > 0:
        prompt = prompt.rsplit('Putting the steps together', 1)[0]
        if ablation_style:
          prompt = prompt + f'Step {i + 1} code:\n'
        else:
          prompt = prompt + f'Step {i + 1} computes:\n'
      if verbose:
        print('===prompt')
        print(prompt.rsplit('[BEGIN PROBLEM]', 1)[-1])

      sample = cached_llm_access.query_llm(
          query_llm,
          prompt,
          n=1,  # For step-by-step, we generate one solution at a time
          temperature=_TEMPERATURE.value,
          model=_MODEL.value,
          num_output_tokens=num_output_tokens,
      )[0]
      program_step = llm_utils.cut_program_from_sample(sample)
      # Record the full prediction containing the LLM-predicted subgoals.
      # The actual outputs are not recorded but can be recomputed later.
      trajectory.append({
          'sample': sample,
          'program_step': program_step,
          'prompt': prompt,
      })
      if verbose:
        print('===program_step:')
        print(program_step)
      # We construct a executable program for the steps generated so far.
      program_prefix = test_problem_wo_solution.python_program
      if dataset_type == 'deepcoder':
        if program_prefix is None:
          # If this is the first step:
          # Borrow function signature from standard solution.
          program_prefix = test_problem.python_program.splitlines()[0] + '\n'
        else:
          program_prefix = program_prefix.rsplit('  return', 1)[0]
        new_var = program_step.split('=', 1)[0].strip()
        program_suffix = f'  return {new_var}'
      elif dataset_type == 'robustfill':
        if program_prefix is None:
          program_prefix = 'def program(x):\n  parts = [\n'
        else:
          program_prefix = program_prefix.rsplit('  ]', 1)[0]
        program_suffix = "  ]\n  return ''.join(parts)"
      else:
        raise ValueError(f'Unhandled dataset type: {dataset_type}')
      indent_spaces = '  ' if dataset_type == 'deepcoder' else '    '
      suffix_ = ',\n' if dataset_type == 'robustfill' else '\n'
      compose_program = (
          program_prefix
          + f'{indent_spaces}{program_step.strip()}{suffix_}'
          + program_suffix
      )
      # Update test problem with the new program
      test_problem_wo_solution = DatasetElement(
          test_problem.inputs, test_problem.outputs, None, compose_program
      )
      if verbose:
        print('===compose_program:')
        print(compose_program)
      try:
        outputs = llm_utils.run_program(
            compose_program, test_problem.inputs, dataset_type=dataset_type
        )
      except Exception:  # pylint: disable=broad-exception-caught
        outputs = None
      if outputs == test_problem.outputs:
        success = True
        break  # Stop at the first successful sample

    if verbose:
      print(test_problem_wo_solution.python_program)
    samples.append(test_problem_wo_solution.python_program)
    trajectories.append(trajectory)
    if success:
      break

  elapsed_time = timeit.default_timer() - start_time
  result = {
      'index': problem_index,
      'test_problem': test_problem,
      'samples': samples,
      'trajectories': trajectories,  # For ExeDec only
      'success': success,
      'elapsed_time': elapsed_time,
  }
  if verbose:
    print(
        f'  Test problem #{problem_index}: '
        f'{"SUCCESS" if result["success"] else "fail"}',
        flush=True,
    )
  return result


def solver_parallel_run(func: Callable, inputs, num_workers: int):  # pylint: disable=g-bare-generic
  """Run solvers in parallel with multithreading."""
  # Run the solvers and show metrics.
  total_cnt, pass_cnt = 0, 0
  with multiprocessing.dummy.Pool(processes=num_workers) as p:
    num_inputs = len(inputs)
    results: list[Any] = [{}] * num_inputs
    with tqdm.tqdm(total=num_inputs) as pbar:
      for res in p.starmap(func, inputs):
        # Update cummulated metrics.
        total_cnt += 1
        pass_cnt += int(res['success'])
        pass_rate = pass_cnt / total_cnt
        pbar.update()
        pbar.set_postfix({
            'pass_cnt': pass_cnt,
            'pass_rate': pass_rate,
            'total': total_cnt,
        })
        results[res['index']] = res
  return results


def run_experiment(
    dataset_type: str,
    generalization_task: str,
    verbose: bool = False,
    parallel: bool = True,
) -> list[dict[str, Any]]:
  """Runs the experiment for a generalization task."""
  print(f'Running experiment for {dataset_type} {generalization_task}...',
        flush=True)
  if dataset_type == 'robustfill':
    version = _VERSION_ROBUSTFILL.value
  elif dataset_type == 'deepcoder':
    version = _VERSION_DEEPCODER.value
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')
  dataset = llm_utils.load_jsonl_dataset(
      dataset_type=dataset_type,
      generalization_task=generalization_task,
      data_format=DATA_FORMAT,
      version=version,
  )
  num_test = len(dataset)
  print(f'Loaded {num_test} test problems', flush=True)

  results = []
  num_output_tokens = _sample_length(dataset_type)
  prompt_format = _PROMPT_FORMAT.value
  ablation_style = prompt_format == 'exedec_ablation'

  solver_inputs = []
  for problem_index in range(num_test):

    test_problem, few_shot_examples = dataset[problem_index]
    solver_inputs.append([
        problem_index,
        few_shot_examples,
        test_problem,
        dataset_type,
        num_output_tokens,
        verbose,
        ablation_style,
    ])
  if not parallel:
    for solver_input in solver_inputs:
      if prompt_format == 'baseline':
        problem_result = solve_problem_baseline(*solver_input)
      elif prompt_format == 'exedec':
        problem_result = solve_problem_exedec(*solver_input)
      elif prompt_format == 'exedec_ablation':
        problem_result = solve_problem_exedec(*solver_input)
      else:
        raise ValueError(f'Unhandled prompt format: {prompt_format}')

      results.append(problem_result)
  else:
    num_workers = _NUM_WORKERS.value
    if prompt_format == 'baseline':
      results = solver_parallel_run(
          solve_problem_baseline, solver_inputs, num_workers
      )
    elif prompt_format == 'exedec':
      results = solver_parallel_run(
          solve_problem_exedec, solver_inputs, num_workers
      )
    elif prompt_format == 'exedec_ablation':
      results = solver_parallel_run(
          solve_problem_exedec, solver_inputs, num_workers
      )
    else:
      raise ValueError(f'Unhandled prompt format: {prompt_format}')

  num_success = sum(r['success'] for r in results)
  print(f'  Solved {num_success} / {len(results)} problems', flush=True)
  return results


def run_entire_experiment() -> dict[str, dict[str, list[dict[str, Any]]]]:
  """Runs the experiment for all datasets and generalization tasks."""
  # Perform experiment.
  all_results = {}
  for dataset_type in ['deepcoder', 'robustfill']:
    all_results[dataset_type] = {}
    for generalization_task in [
        'NONE',
        'LENGTH_GENERALIZATION',
        'COMPOSE_DIFFERENT_CONCEPTS',
        'SWITCH_CONCEPT_ORDER',
        'COMPOSE_NEW_OP',
        'ADD_OP_FUNCTIONALITY',
    ]:
      if _TARGET_TASK.value is not None:
        if generalization_task != _TARGET_TASK.value:
          continue
      results = run_experiment(dataset_type, generalization_task, verbose=False)
      all_results[dataset_type][generalization_task] = results

  # Write actual results files.

  results_path = RESULTS_FORMAT.format(
      prompt_format=_PROMPT_FORMAT.value,
      model=_MODEL.value.split('/')[-1],
      num_samples=_NUM_SAMPLES.value,
      temperature=_TEMPERATURE.value,
      version_deepcoder=_VERSION_DEEPCODER.value,
      version_robustfill=_VERSION_ROBUSTFILL.value,
  )
  print(f'Writing results to {results_path}...', flush=True)
  with tf.io.gfile.GFile(results_path, 'w') as f:
    json.dump(all_results, f)
  print('Experiment done!')

  return all_results


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if _TEMPERATURE.value == 0.0:
    assert _NUM_SAMPLES.value == 1

  cache_dir = os.path.expanduser(_LLM_CACHE_DIR.value)
  model_name = _MODEL.value.split('/')[-1]
  cached_llm_access.init_cache(cache_dir, model_name)

  run_entire_experiment()


if __name__ == '__main__':
  app.run(main)
