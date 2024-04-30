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

"""Utils for using the datasets."""

import types
from typing import Any

from exedec.tasks import experiment
from exedec.tasks.deepcoder import deepcoder_dsl
from exedec.tasks.robust_fill import dsl as robustfill_dsl
from exedec.tasks.robust_fill import tokens as robustfill_tokens

# For convenience.
Experiment = experiment.Experiment
# Same for DeepCoder and RobustFill.
BOS_ID = 1
EOS_ID = 2


################################################################################
# Utils for the LLM data in exedec/data/llm_data/*.jsonl
#
# These data files contain the actual test problems and few-shot examples used
# in the LLM experiments in ExeDec's ICLR'24 paper. There is one jsonl file for
# each kind of compositional generalization for both domains (DeepCoder and
# RobustFill). Each jsonl file contains 200 PBE problems from the test
# distribution, and for each test problem, the 4 PBE problems from the train
# distribution that were used as examples for few-shot prompting.
#
# This data is formatted for use with LLMs that can write Python programs. Each
# element is a dict with keys:
#
#   index: The index within that dataset file, as an int from 0 to 199.
#   test_problem: A PBE problem from the test distribution, as a dict with keys:
#     inputs: A dict (DeepCoder) or list (RobustFill) containing the program
#       inputs, one per I/O example. Can be passed to `run_python_program`.
#     outputs: A list of output objects, one per I/O example.
#     program: A DSL program that solves the problem.
#   few_shot_examples: A list of four dicts, each describing a PBE problem from
#     the train distribution, in the same format as test_problem.

_ROBUSTFILL_PROGRAM_ID_TO_TOKEN, _ = robustfill_tokens.build_token_tables()
_ROBUSTFILL_FUNCTIONS = [
    'Const', 'SubStr', 'GetSpan', 'GetToken', 'ToCase', 'Replace', 'Trim',
    'GetUpto', 'GetFrom', 'GetFirst', 'GetAll', 'Substitute', 'SubstituteAll',
    'Remove', 'RemoveAll',
]
_ROBUSTFILL_ENUMS = [
    robustfill_dsl.Type, robustfill_dsl.Case, robustfill_dsl.Boundary,
]


def dsl_program_to_python(
    dsl_program: str, dataset_type: str, pythonic: bool = False) -> str:
  """Converts a DSL program into a Python program.

  Args:
    dsl_program: A program in DSL form. A DeepCoder DSL program looks like
      "x0 = INPUT | x1 = Map (+1) x0 | x2 = Scanl1 (+) x1", while a RobustFill
      DSL program looks like "5 216 218|4 83|3 8 111 7 103 217". The *.jsonl
      dataset files have programs in this form.
    dataset_type: The kind of dataset, either 'deepcoder' or 'robustfill'.
    pythonic: For DeepCoder, whether to use the Pythonic style referenced in the
      ICLR'24 paper. Must be False for RobustFill.

  Returns:
    The program as a valid Python function that calls a hypothetical `dsl`
    library. Such a program may be given to `run_python_program`.
  """
  if dataset_type == 'deepcoder':
    program_object = deepcoder_dsl.Program.from_str(dsl_program)
    return program_object.to_python_program(version=4 if pythonic else 1)
  elif dataset_type == 'robustfill':
    if pythonic:
      raise ValueError('There is no Pythonic version for RobustFill.')
    program_tokens = [int(t) for t in dsl_program.replace('|', ' ').split()]
    program_tokens.append(EOS_ID)
    program_object = robustfill_dsl.decode_program(
        encoding=program_tokens, id_token_table=_ROBUSTFILL_PROGRAM_ID_TO_TOKEN)
    return program_object.to_python_program(version=1)
  else:
    raise ValueError(f'Unknown dataset_type: {dataset_type}')


def _get_namespace(dataset_type: str) -> dict[str, Any]:
  """Gets a namespace with the dsl loaded."""
  dsl_object = types.SimpleNamespace()
  if dataset_type == 'deepcoder':
    for lambda_ in deepcoder_dsl.LAMBDAS:
      setattr(dsl_object, lambda_.name, lambda_.func)
    for op in deepcoder_dsl.OPERATIONS:
      setattr(dsl_object, op.token, op.func)
  elif dataset_type == 'robustfill':
    for function_name in _ROBUSTFILL_FUNCTIONS:
      if function_name == 'Const':
        op_class = robustfill_dsl.ConstStr
        wrapper = lambda c, op_class=op_class: op_class(c)(None)
      else:
        op_class = getattr(robustfill_dsl, function_name)
        wrapper = lambda x, *args, op_class=op_class: op_class(*args)(x)
      setattr(dsl_object, function_name, wrapper)
    for enum_class in _ROBUSTFILL_ENUMS:
      setattr(dsl_object, enum_class.__name__, enum_class)
  else:
    raise ValueError(f'Unhandled dataset_type: {dataset_type}')
  return {'dsl': dsl_object}


def _get_num_examples(inputs: list[str] | dict[str, Any],
                      dataset_type: str) -> int:
  """Returns the number of examples in the inputs."""
  if dataset_type == 'deepcoder':
    assert isinstance(inputs, dict)
    inputs_dict = inputs
    num_examples = len(list(inputs_dict.values())[0])
    assert all(len(v) == num_examples for v in inputs_dict.values())
  elif dataset_type == 'robustfill':
    assert isinstance(inputs, list)
    num_examples = len(inputs)
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')
  return num_examples


def run_python_program(python_program: str,
                       inputs: dict[str, Any] | list[str],
                       dataset_type: str,
                       program_name: str = 'program') -> list[Any] | None:
  """Runs a Python-style DeepCoder or RobustFill program.

  Warning: This will call exec() on `python_program` and eval() on `inputs`,
  which may be unsafe. Take care when calling this function on untrusted inputs.
  Consider using security measures such as a Docker container, VM, or similar.

  Args:
    python_program: A Python function representing the program, such as one
      returned by `dsl_program_to_python`.
    inputs: For DeepCoder, a dict mapping each input variable name (e.g., 'x0')
      to a list of its values for each I/O example. For RobustFill, a list
      containing the input string for each I/O example.
    dataset_type: The kind of dataset, either 'deepcoder' or 'robustfill'.
    program_name: The name of the function to call.

  Returns:
    A list of program outputs (one per I/O example), or None if there was an
    error.
  """
  # Set up code for calling the solution function with appropriate arguments.
  if dataset_type == 'deepcoder':
    assert isinstance(inputs, dict)
    call_code = f'{program_name}({", ".join(inputs.keys())})'
  elif dataset_type == 'robustfill':
    call_code = f'{program_name}(x)'
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')

  # Define the solution function.
  namespace = _get_namespace(dataset_type)
  try:
    exec(python_program, namespace)  # pylint: disable=exec-used
  except:  # pylint: disable=bare-except
    return None

  # Run the solution function for each example.
  outputs = []
  for i in range(_get_num_examples(inputs, dataset_type)):
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


################################################################################
# Utils for the test datasets in exedec/data/test_data/*.jsonl
#
# These data files contain the actual test problems used in the small
# Transformer experiments in ExeDec's ICLR'24 paper. There is one jsonl file for
# each kind of compositional generalization for both domains (DeepCoder and
# RobustFill). Each jsonl file contains 1000 PBE problems from the test
# distribution.
#
# This data is formatted for use with neural models trained from scratch, where
# we control the vocabulary and tokenization of programs and I/O example
# specifications. For RobustFill, we used different vocabularies for I/O
# specifications and for programs, but for DeepCoder, these are the same. Each
# data element is one PBE problem as a dict with keys:
#
#   index: The index within that dataset file, as an int from 0 to 999.
#   inputs: A list[str] containing the problem inputs, one per I/O example.
#   outputs: A list[str] containing the problem outputs, one per I/O example.
#   program: A DSL program that solves the problem.

SEPARATOR_TOKEN = '|'


def spec_vocab_tables(
    dataset_type: str) -> tuple[dict[int, Any], dict[Any, int]]:
  """Returns id-to-token and token-to-id vocab mappings for inputs/outputs."""
  if dataset_type == 'deepcoder':
    # Same as DeepCoder's program vocab.
    return deepcoder_dsl.vocab_tables()
  elif dataset_type == 'robustfill':
    spec_vocab = robustfill_dsl.CHARACTER + SEPARATOR_TOKEN
    spec_id_to_token = {i + 3: token for i, token in enumerate(spec_vocab)}
    spec_id_to_token[0] = None  # Padding.
    spec_id_to_token[BOS_ID] = robustfill_dsl.BOS
    spec_id_to_token[EOS_ID] = robustfill_dsl.EOS
    spec_token_to_id = {
        token: id for id, token in spec_id_to_token.items()
    }
    return spec_id_to_token, spec_token_to_id
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')


def program_vocab_tables(
    dataset_type: str) -> tuple[dict[int, Any], dict[Any, int]]:
  """Returns id-to-token and token-to-id vocab mappings for programs."""
  if dataset_type == 'deepcoder':
    # Same as DeepCoder's specification vocab.
    return deepcoder_dsl.vocab_tables()
  elif dataset_type == 'robustfill':
    return robustfill_tokens.build_token_tables()
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')


def spec_str_to_ids(spec_str: str,
                    dataset_type: str,
                    spec_token_to_id: dict[Any, int]) -> list[int]:
  """Converts an input or output string into a list of vocab IDs."""
  if dataset_type == 'deepcoder':
    return [spec_token_to_id[t] for t in spec_str.split(' ')]
  elif dataset_type == 'robustfill':
    return [spec_token_to_id[t] for t in spec_str]
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')


def spec_ids_to_str(spec_ids: list[int],
                    dataset_type: str,
                    spec_id_to_token: dict[int, Any]) -> str:
  """Converts vocab IDs for an input or output into a string."""
  if dataset_type == 'deepcoder':
    separator = ' '
  elif dataset_type == 'robustfill':
    separator = ''
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')
  return separator.join(spec_id_to_token[i] for i in spec_ids
                        if i > 0 and i != BOS_ID and i != EOS_ID)


def program_str_to_ids(program_str: str,
                       dataset_type: str,
                       program_token_to_id: dict[Any, int]) -> list[int]:
  """Converts a program string into a list of vocab IDs."""
  if dataset_type == 'deepcoder':
    return [program_token_to_id[t] for t in program_str.split(' ')] + [EOS_ID]
  elif dataset_type == 'robustfill':
    split_program = program_str.replace(SEPARATOR_TOKEN, ' ').split(' ')
    return [int(i) for i in split_program] + [EOS_ID]
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')


def ids_to_program(
    program_ids: list[int],
    dataset_type: str,
    program_id_to_token: dict[int, Any],
) -> deepcoder_dsl.Program | robustfill_dsl.Program | None:
  """Parses program vocab IDs into a program object, or None if malformed."""
  if dataset_type == 'deepcoder':
    program_tokens = [program_id_to_token[p_id]
                      for p_id in program_ids
                      if p_id > 0 and p_id != EOS_ID]
    try:
      return deepcoder_dsl.Program.from_tokens(program_tokens)
    except Exception:  # pylint: disable=broad-except
      return None
  elif dataset_type == 'robustfill':
    try:
      processed_ids = [p_id for p_id in program_ids
                       if p_id > 0 and p_id != BOS_ID and p_id != EOS_ID]
      return robustfill_dsl.decode_program(processed_ids + [EOS_ID],
                                           program_id_to_token)
    except Exception:  # pylint: disable=broad-except
      return None
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')


def run_program(program: deepcoder_dsl.Program | robustfill_dsl.Program,
                inputs: list[str],
                dataset_type: str) -> list[Any]:
  """Runs a program on inputs and returns the outputs (None on error)."""
  outputs = []
  for i in inputs:
    if dataset_type == 'deepcoder':
      try:
        assert isinstance(program, deepcoder_dsl.Program)
        initial_state = deepcoder_dsl.ProgramState.from_str(i)
        final_state = program.run(initial_state.state)
        output = (deepcoder_dsl.result_to_str(final_state.get_output())
                  if final_state else None)
      except Exception:  # pylint: disable=broad-except
        output = None
    elif dataset_type == 'robustfill':
      try:
        assert isinstance(program, robustfill_dsl.Program)
        output = program(i)
      except Exception:  # pylint: disable=broad-except
        output = None
    else:
      raise ValueError(f'Unhandled dataset type: {dataset_type}')
    outputs.append(output)
  return outputs
