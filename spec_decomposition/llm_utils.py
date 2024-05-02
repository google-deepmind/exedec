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

"""Utilities for measuring compositional generalization of LLMs."""

import ast
import collections
import copy
import json
import re
import types
from typing import Any

import tensorflow as tf

from exedec.tasks.deepcoder import deepcoder_dsl
from exedec.tasks.robust_fill import dsl as robustfill_dsl
from exedec.tasks.robust_fill import tokens as robustfill_tokens


# `inputs` is a dict for DeepCoder (name to list of values for each example), or
# a list for RobustFill (values for each example).
DatasetElement = collections.namedtuple(
    'DatasetElement',
    ['inputs', 'outputs', 'dsl_program', 'python_program'])

# `states` is similar to the `inputs` of `DatasetElement`, and it records the
# states of the newly created variable after executing the partial python
# program, `targets` is the `remains` string for RobustFill, and is None for
# DeepCoder.
StepData = collections.namedtuple(
    'StepData', ['states', 'targets', 'python_program_step']
)
ExeDecTrajectory = list[StepData]

ROBUSTFILL_ID_TOKEN_TABLE, _ = robustfill_tokens.build_token_tables()
ROBUSTFILL_EOS_ID = 2
ROBUSTFILL_FUNCTIONS = [
    'Const', 'SubStr', 'GetSpan', 'GetToken', 'ToCase', 'Replace', 'Trim',
    'GetUpto', 'GetFrom', 'GetFirst', 'GetAll', 'Substitute', 'SubstituteAll',
    'Remove', 'RemoveAll',
]
ROBUSTFILL_ENUMS = [
    robustfill_dsl.Type, robustfill_dsl.Case, robustfill_dsl.Boundary,
]

# Enables using the exact same datasets for any settings *up to* these numbers
# of examples, for more consistent comparisons between experiments that use
# different settings. The datasets will change if these numbers are changed.
MAX_NUM_FEW_SHOT_EXAMPLES = 10
MAX_NUM_TEST_PROBLEMS = 200

DEEPCODER_MAX_LIST_LENGTH = 5


def to_python_form(io: str) -> str:
  """Convert Deepcoder's "x1 = [ 1 2 ] | x2 = 3" into "x1 = [1, 2], x2 = 3"."""
  io = io.replace(' | ', ', ').replace('[ ', '[').replace(' ]', ']')
  io = re.sub(r'(?<=\d) (?=-|\d)', ', ', io)
  return io


def parse_dataset(dataset: tf.data.Dataset,
                  dataset_type: str,
                  version: int) -> list[DatasetElement]:
  """Parses the tf.data.Dataset into a list of DatasetElement."""
  data = []

  for element in dataset:
    inputs = [x.decode() for x in element['inputs'].numpy().tolist()]
    outputs = [x.decode() for x in element['outputs'].numpy().tolist()]
    program = element['program'].numpy().decode()

    if dataset_type == 'deepcoder':
      input_names = re.findall(r'x\d', inputs[0])
      inputs_dict = {name: [] for name in input_names}
      for s in inputs:
        for name in input_names:
          value_str = re.search(name + r' = ([\[\] \-0-9]+)($| \|)', s).group(1)
          value = ast.literal_eval(to_python_form(value_str))
          inputs_dict[name].append(value)
      inputs = inputs_dict
      outputs = [ast.literal_eval(to_python_form(o)) for o in outputs]
      program_object = deepcoder_dsl.Program.from_str(program)
    elif dataset_type == 'robustfill':
      program_tokens = [int(t) for t in program.replace('|', ' ').split()]
      program_tokens.append(ROBUSTFILL_EOS_ID)
      program_object = robustfill_dsl.decode_program(
          encoding=program_tokens, id_token_table=ROBUSTFILL_ID_TOKEN_TABLE)
    else:
      raise ValueError(f'Unhandled dataset type: {dataset_type}')

    python_program = program_object.to_python_program(version=version)

    d = DatasetElement(inputs, outputs, program, python_program)
    if dataset_type == 'deepcoder':
      d = canonicalize_deepcoder_variables(d)
    actual_outputs = run_program(d.python_program, d.inputs, dataset_type)
    if d.outputs != actual_outputs:
      raise ValueError(
          f'Program:\n'
          f'{d.python_program}\n'
          f'Inputs: {d.inputs}\n'
          f'Expected outputs: {d.outputs}\n'
          f'Actual outputs: {actual_outputs}\n'
      )
    data.append(d)
  return data


def create_dataset(file_pattern, num_examples):
  """Loads a DeepCoder or RobustFill dataset of entire programs.

  Args:
    file_pattern: A file pattern for the TFRecord files to read.
    num_examples: The number of examples in an I/O specification.

  Returns:
    A tf.data.Dataset.
  """
  filenames = sorted(tf.io.gfile.glob(file_pattern))
  raw_dataset = tf.data.TFRecordDataset(filenames)

  def _parse_fn(record):
    """Parses a record into a feature_dict."""
    empty_default = [''] * num_examples
    feature_values = tf.io.parse_single_example(
        serialized=record,
        features={
            'inputs':
                tf.io.FixedLenFeature([num_examples], tf.string,
                                      default_value=empty_default),
            'outputs':
                tf.io.FixedLenFeature([num_examples], tf.string,
                                      default_value=empty_default),
            'program':
                tf.io.FixedLenFeature([], tf.string, default_value=''),
        })
    return {
        'inputs': feature_values['inputs'],
        'outputs': feature_values['outputs'],
        'program': feature_values['program'],
    }

  dataset = raw_dataset.map(_parse_fn)
  return dataset


def json_to_dataset_element(json_dict: dict[str, Any],
                            dataset_type: str,
                            version: int) -> DatasetElement:
  """Converts a json dict to a DatasetElement."""
  dsl_program = json_dict['program']
  if dataset_type == 'deepcoder':
    program_object = deepcoder_dsl.Program.from_str(dsl_program)
    python_program = program_object.to_python_program(version=version)
  elif dataset_type == 'robustfill':
    program_tokens = [int(t) for t in dsl_program.replace('|', ' ').split()]
    program_tokens.append(ROBUSTFILL_EOS_ID)
    program_object = robustfill_dsl.decode_program(
        encoding=program_tokens, id_token_table=ROBUSTFILL_ID_TOKEN_TABLE)
    python_program = program_object.to_python_program(version=version)
  else:
    raise ValueError(f'Unknown dataset_type: {dataset_type}')

  return DatasetElement(
      inputs=json_dict['inputs'],
      outputs=json_dict['outputs'],
      dsl_program=dsl_program,
      python_program=python_program,
  )


def load_jsonl_dataset(
    dataset_type: str,
    generalization_task: str,
    data_format: str,
    version: int,
) -> list[tuple[DatasetElement, list[DatasetElement]]]:
  """Returns a list of tuples (test problem, few-shot examples for it)."""
  data_filename = data_format.format(
      dataset_type=dataset_type, generalization_task=generalization_task)
  with open(data_filename, 'r') as f:
    all_json_data = [json.loads(line) for line in f.readlines()]
  dataset = []
  for json_data in all_json_data:
    test_problem = json_to_dataset_element(
        json_data['test_problem'], dataset_type, version)
    few_shots = [json_to_dataset_element(few_shot_data, dataset_type, version)
                 for few_shot_data in json_data['few_shot_examples']]
    dataset.append((test_problem, few_shots))
  return dataset


def get_namespace(dataset_type: str) -> dict[str, Any]:
  """Gets a namespace with the dsl loaded."""
  dsl_object = types.SimpleNamespace()
  if dataset_type == 'deepcoder':
    for lambda_ in deepcoder_dsl.LAMBDAS:
      setattr(dsl_object, lambda_.name, lambda_.func)
    for op in deepcoder_dsl.OPERATIONS:
      setattr(dsl_object, op.token, op.func)
  elif dataset_type == 'robustfill':
    for function_name in ROBUSTFILL_FUNCTIONS:
      if function_name == 'Const':
        op_class = robustfill_dsl.ConstStr
        wrapper = lambda c, op_class=op_class: op_class(c)(None)
      else:
        op_class = getattr(robustfill_dsl, function_name)
        wrapper = lambda x, *args, op_class=op_class: op_class(*args)(x)
      setattr(dsl_object, function_name, wrapper)
    for enum_class in ROBUSTFILL_ENUMS:
      setattr(dsl_object, enum_class.__name__, enum_class)
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')
  return {'dsl': dsl_object}


def get_num_examples(inputs: list[str] | dict[str, Any],
                     dataset_type: str) -> int:
  """Returns the number of examples in the inputs."""
  if dataset_type == 'deepcoder':
    assert isinstance(inputs, dict)
    inputs_dict = inputs
    num_examples = len(list(inputs_dict.values())[0])
    assert all(len(v) == num_examples for v in inputs_dict.values())
  elif dataset_type == 'robustfill':
    assert isinstance(inputs, list), f'RobustFill inputs: {inputs}'
    num_examples = len(inputs)
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')
  return num_examples


def run_program(program_code: str,
                inputs: list[str] | dict[str, Any],
                dataset_type: str,
                program_name: str = 'program') -> list[Any] | None:
  """Runs a DeepCoder or RobustFill program."""
  # Set up code for calling the solution function with appropriate arguments.
  if dataset_type == 'deepcoder':
    assert isinstance(inputs, dict)
    call_code = f'{program_name}({", ".join(inputs.keys())})'
  elif dataset_type == 'robustfill':
    call_code = f'{program_name}(x)'
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')

  # Define the solution function.
  namespace = get_namespace(dataset_type)
  try:
    exec(program_code, namespace)  # pylint: disable=exec-used
  except:  # pylint: disable=bare-except
    return None

  # Run the solution function for each example.
  outputs = []
  for i in range(get_num_examples(inputs, dataset_type)):
    namespace_copy = namespace.copy()
    # Assign the argument values.
    if dataset_type == 'deepcoder':
      assert isinstance(inputs, dict)
      for input_name, input_values in inputs.items():
        namespace_copy[input_name] = input_values[i]
    elif dataset_type == 'robustfill':
      namespace_copy['x'] = inputs[i]
    else:
      raise ValueError(f'Unhandled dataset type: {dataset_type}')
    # Call the solution function.
    try:
      output = eval(call_code, namespace_copy)  # pylint: disable=eval-used
    except:  # pylint: disable=bare-except
      output = None
    outputs.append(output)

  return outputs


_DEEPCODER_FUNCTION_IMPLS = [
    '''
def Map(f, xs):
  return [f(x) for x in xs]
''',
    '''
def Filter(f, xs):
  return [x for x in xs if f(x)]
''',
    '''
def Count(f, xs):
  return len([x for x in xs if f(x)])
''',
    '''
def ZipWith(f, xs, ys):
  return [f(x, y) for (x, y) in zip(xs, ys)]
''',
    '''
def Scanl1(f, xs):
  ys = []
  for i, x in enumerate(xs):
    if i == 0:
      ys.append(x)
    else:
      ys.append(f(ys[-1], x))
  return ys
''',
]

_DEEPCODER_LAMBDA_IMPLS = '''
PLUS_ONE = lambda x: x + 1
MINUS_ONE = lambda x: x - 1
TIMES_TWO = lambda x: x * 2
DIV_TWO = lambda x: x // 2
NEGATE = lambda x: -x
SQUARE = lambda x: x ** 2
TIMES_THREE = lambda x: x * 3
DIV_THREE = lambda x: x // 3
TIMES_FOUR = lambda x: x * 4
DIV_FOUR = lambda x: x // 4
IS_POSITIVE = lambda x: x > 0
IS_NEGATIVE = lambda x: x < 0
IS_EVEN = lambda x: x % 2 == 0
IS_ODD = lambda x: x % 2 == 1
ADD = lambda x, y: x + y
SUBTRACT = lambda x, y: x - y
MULTIPLY = lambda x, y: x * y
MIN = lambda x, y: min(x, y)
MAX = lambda x, y: max(x, y)
'''.strip()


def dsl_description(dataset_type: str, version: int) -> str:
  """Gets a description of the DSL for prompting."""
  if dataset_type == 'deepcoder':
    dsl_purpose = 'manipulating lists of integers'
    if version == 1:
      function_details = ', '.join(
          [op.token for op in deepcoder_dsl.OPERATIONS])
      constant_details = ', '.join(
          [lambda_.name for lambda_ in deepcoder_dsl.LAMBDAS])
    elif version == 2:
      function_details = '\n\n'.join(
          [i.strip() for i in _DEEPCODER_FUNCTION_IMPLS])
      constant_details = _DEEPCODER_LAMBDA_IMPLS
    elif version == 3 or version == 5:
      function_details = _DEEPCODER_FUNCTION_IMPLS[-1].strip()
      constant_details = _DEEPCODER_LAMBDA_IMPLS
    elif version == 4:
      function_details = _DEEPCODER_FUNCTION_IMPLS[-1].strip()
      constant_details = None
    else:
      raise ValueError(f'Unhandled version: {version}')
  elif dataset_type == 'robustfill':
    if version == 1:
      dsl_purpose = 'manipulating strings'
      function_details = ', '.join(ROBUSTFILL_FUNCTIONS)
      constant_details = ', '.join(
          [robustfill_dsl.to_python(obj)  # pylint: disable=g-complex-comprehension
           for e in ROBUSTFILL_ENUMS for obj in e])
    else:
      raise ValueError(f'Unhandled version: {version}')
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')
  return (
      f'The `dsl` module is a custom library for {dsl_purpose}. It contains '
      'the following functions:\n\n'
      f'{function_details}\n\n'
      + (
          'Additionally, the module defines the following constants:\n\n'
          f'{constant_details}\n\n'
          if constant_details else '') +
      'Below are example programming problems using the `dsl` module, with'
      ' input-output test cases illustrating their behavior.\n\nImportant:'
      ' All programs begin with ```python and end with ``` alone.\n\n'
  )


def get_prompt_prefix(dataset_element: DatasetElement,
                      dataset_type: str) -> str:
  """Gets a prefix of the prompt describing one dataset element."""
  s = '[BEGIN PROBLEM]\n'
  s += 'Input-output test cases:\n'
  for i in range(get_num_examples(dataset_element.inputs, dataset_type)):
    s += f'  Case {i + 1}. '
    if dataset_type == 'deepcoder':
      sep = ''
      for name in dataset_element.inputs:
        s += f'{sep}{name} = {dataset_element.inputs[name][i]}'
        sep = ', '
      s += f' --> {dataset_element.outputs[i]}\n'
    elif dataset_type == 'robustfill':
      s += f'"{dataset_element.inputs[i]}" --> "{dataset_element.outputs[i]}"\n'
    else:
      raise ValueError(f'Unhandled dataset type: {dataset_type}')
  s += '\nProgram:\n```python\n'
  return s


def get_prompt_suffix(dataset_element: DatasetElement) -> str:
  return f'{dataset_element.python_program}\n```\n[END PROBLEM]\n\n'


def get_prompt(dataset_element: DatasetElement, dataset_type: str) -> str:
  return (get_prompt_prefix(dataset_element, dataset_type)
          + get_prompt_suffix(dataset_element))


def few_shot_prompt(few_shot_examples: list[DatasetElement],
                    test_problem: DatasetElement,
                    dataset_type: str,
                    version: int) -> str:
  prompt_parts = [dsl_description(dataset_type, version=version)]
  prompt_parts.extend(get_prompt(d, dataset_type) for d in few_shot_examples)
  prompt_parts.append(get_prompt_prefix(test_problem, dataset_type))
  return '\n'.join(prompt_parts)


def check_deepcoder_object_valid(s: Any) -> bool:
  """Check if the object is a valid DeepCoder object."""
  # For DeepCoder, every object is either an int or a list of ints
  if not (isinstance(s, int) or isinstance(s, list)):
    raise ValueError(f'Invalid DeepCoder object: {s}, type: {type(s)}')
  if isinstance(s, list):
    # Every list has length <= DEEPCODER_MAX_LIST_LENGTH
    if not len(s) <= DEEPCODER_MAX_LIST_LENGTH:
      raise ValueError(f'Invalid DeepCoder object: {s}, length: {len(s)}')
    for x in s:
      if not isinstance(x, int):
        raise ValueError(f'Invalid DeepCoder object: {s}, type: {type(x)}')
      # Every int is in the range [-50, 50] inclusive
      if not (-50 <= x <= 50):
        raise ValueError(f'Invalid DeepCoder object: {s}, int: {x}')
  else:
    if not (-50 <= s <= 50):
      raise ValueError(f'Invalid DeepCoder object: {s}, int: {s}')
  return True


def check_robustfill_object_valid(s: Any) -> bool:
  if not isinstance(s, str):
    raise ValueError(f'Invalid RobustFill object: {s}, type: {type(s)}')
  if len(s) > 20:
    raise ValueError(f'Invalid RobustFill object: {s}, length: {len(s)}')
  return True


def get_exe_dec_trajectory(
    dataset_element: DatasetElement, dataset_type: str
) -> ExeDecTrajectory:
  """Decompose the dataset element into a ExeDec trajectory."""
  trajectory: ExeDecTrajectory = []
  program_steps = dataset_element.python_program.splitlines()
  # The initial step.
  states = copy.deepcopy(dataset_element.inputs)
  targets = (
      None
      if dataset_type == 'deepcoder'
      else copy.deepcopy(dataset_element.outputs)
  )
  trajectory.append(StepData(states, targets, program_steps[0]))
  # The middle steps before return
  for j in range(1, len(program_steps) - 1):
    python_program_step = program_steps[j]
    if dataset_type == 'deepcoder':
      new_var = python_program_step.strip().split('=', 1)[0].strip()

      compose_program = '\n'.join(
          [x.python_program_step for x in trajectory]
          + [python_program_step, f'  return {new_var}']
      )

      actual_states = run_program(
          compose_program, dataset_element.inputs, dataset_type
      )
      for s in actual_states:
        if not check_deepcoder_object_valid(s):
          raise ValueError(f'Invalid DeepCoder object: {s}')
      actual_states = {new_var: actual_states}
      new_targets = None
    elif dataset_type == 'robustfill':
      if python_program_step.strip() in ['parts = [', ']']:
        continue
      python_program_step = python_program_step.strip()
      if python_program_step.endswith(','):
        python_program_step = python_program_step[:-1]
      compose_program = (
          trajectory[0].python_program_step
          + '\n'
          + f'  return {python_program_step}'
      )
      actual_states = run_program(
          compose_program, dataset_element.inputs, dataset_type
      )
      previous_targets = trajectory[-1].targets
      new_targets = []
      num_examples = get_num_examples(dataset_element.inputs, dataset_type)

      for s in actual_states:
        if not check_robustfill_object_valid(s):
          raise ValueError(f'Invalid RobustFill object: {s}')
      for i in range(num_examples):
        if (not isinstance(actual_states[i], str)) or (
            not previous_targets[i].startswith(actual_states[i])
        ):
          raise ValueError(
              f'Case {i + 1}: {previous_targets[i]} does not match the prefix'
              f' of {actual_states[i]}'
          )
        new_targets.append(previous_targets[i][len(actual_states[i]) :])
    else:
      raise ValueError(f'Unhandled dataset type: {dataset_type}')

    trajectory.append(StepData(actual_states, new_targets, python_program_step))
  return trajectory


def get_exe_dec_prompt_prefix(
    dataset_element: DatasetElement,
    dataset_type: str,
    ablation_style: bool = False,
) -> str:
  """Gets a prefix of the ExeDec prompt describing one dataset element."""
  s = '[BEGIN PROBLEM]\n'
  s += 'Input-output test cases:\n'
  num_examples = get_num_examples(dataset_element.inputs, dataset_type)
  if dataset_type == 'deepcoder':
    for i in range(num_examples):
      s += f'  Case {i+1}. '
      sep = ''
      for name in dataset_element.inputs:
        s += f'{sep}{name} = {dataset_element.inputs[name][i]}'
        sep = ', '
      s += f' --> {dataset_element.outputs[i]}\n'
  elif dataset_type == 'robustfill':
    for i in range(num_examples):
      s += f'  Case {i+1}. x = '
      s += f'"{dataset_element.inputs[i]}" --> "{dataset_element.outputs[i]}"\n'
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')

  s += '\nWe solve this problem step-by-step.\n\n'
  if ablation_style:
    if dataset_element.python_program is None:
      s += 'Step 1 code:\n'
      return s
  else:
    if dataset_element.python_program is None:
      s += 'Step 1 computes:\n'
      return s
  trajectory = get_exe_dec_trajectory(dataset_element, dataset_type)
  for j in range(1, len(trajectory)):
    subgoals = f'Step {j} computes:\n'
    if dataset_type == 'deepcoder':
      for i in range(get_num_examples(dataset_element.inputs, dataset_type)):
        subgoals += f'  Case {i+1}. '
        sep = ''
        for name in trajectory[j].states:
          subgoals += f'{sep}{name} = {trajectory[j].states[name][i]}'
          sep = ', '
        subgoals += '\n'
    elif dataset_type == 'robustfill':
      for i in range(num_examples):
        subgoals += f'  Case {i+1}. '
        subgoals += (
            f'"{trajectory[j].states[i]}" so "{trajectory[j].targets[i]}"'
            ' remains\n'
        )
    else:
      raise ValueError(f'Unhandled dataset type: {dataset_type}')
    step_code = f'Step {j} code:\n'
    step_code += (
        f'```python\n{trajectory[j].python_program_step.strip()}\n```\n'
    )
    if ablation_style:
      s += step_code + '\n' + subgoals + '\n'
    else:
      s += subgoals + '\n' + step_code + '\n'
  s += (
      'Putting the steps together, the problem is solved with the'
      ' program:\n```python\n'
  )
  return s


def get_exe_dec_prompt_suffix(dataset_element: DatasetElement) -> str:
  return f'{dataset_element.python_program}\n```\n[END PROBLEM]\n\n'


def get_exe_dec_prompt(
    dataset_element: DatasetElement,
    dataset_type: str,
    ablation_style: bool = False,
) -> str:
  return get_exe_dec_prompt_prefix(
      dataset_element, dataset_type, ablation_style=ablation_style
  ) + get_exe_dec_prompt_suffix(dataset_element)


def few_shot_exe_dec_prompt(
    few_shot_examples: list[DatasetElement],
    test_problem: DatasetElement,
    dataset_type: str,
    version: int,
    ablation_style: bool = False,
) -> str:
  """Generate the ExeDec few-shot prompt."""
  prompt_parts = [
      dsl_description(dataset_type, version=version).replace(
          'illustrating their behavior.',
          'illustrating the program behavior step-by-step.',
      )
  ]
  prompt_parts.extend(
      get_exe_dec_prompt(d, dataset_type, ablation_style=ablation_style)
      for d in few_shot_examples
  )
  prompt_parts.append(
      get_exe_dec_prompt_prefix(
          test_problem, dataset_type, ablation_style=ablation_style
      )
  )
  return '\n'.join(prompt_parts)


def canonicalize_deepcoder_variables(
    dataset_element: DatasetElement,
) -> DatasetElement:
  """Canonicalizes the variable names in deepcoder programs and inputs."""
  program: str = dataset_element.python_program
  input_mapping_dict = {}
  input_occurences = re.findall(r'x\d', program)
  x_index = 0
  for input_name in input_occurences:
    if input_name not in input_mapping_dict:
      input_mapping_dict[input_name] = f'x{x_index}'
      x_index += 1

  # canonicalize python program and dsl program
  # To avoid collision, (x7 --> x0, x0 ---> x2) is decomposed to 2 steps
  # (x7 --> y0, x0 --> y2), and (y0 --> x0, y2 --> x2).
  dsl_program = dataset_element.dsl_program
  for input_name, new_name in input_mapping_dict.items():
    program = program.replace(input_name, new_name.replace('x', 'y'))
    dsl_program = dsl_program.replace(input_name, new_name.replace('x', 'y'))
  for new_name in input_mapping_dict.values():
    program = program.replace(new_name.replace('x', 'y'), new_name)
    dsl_program = dsl_program.replace(new_name.replace('x', 'y'), new_name)

  # canonicalize inputs
  inputs = {}
  for input_name, value in dataset_element.inputs.items():
    inputs[input_mapping_dict[input_name]] = copy.deepcopy(value)

  return DatasetElement(inputs, dataset_element.outputs, dsl_program, program)


def cut_program_from_sample(sample: str) -> str:
  if '```python\n' in sample:
    sample = sample.partition('```python\n')[-1]
  if '```' in sample:
    sample = sample.partition('```')[0]
  return sample
