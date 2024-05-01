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

"""XM Launcher for exedec/spec_decomposition:end_to_end_predict."""

import collections
import itertools
import os
from typing import Any

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_local

# Flags for experiment setup.
_EXP_TITLE = flags.DEFINE_string(
    'exp_title', 'end_to_end_predict',
    'Title of the experiment.')
_SAVE_DIR = flags.DEFINE_string(
    'save_dir', None,
    'CNS directory for writing.')
_ACCELERATOR = flags.DEFINE_string(
    'accelerator', 'v100=1',
    'Accelerator specification. Format: <GPU>=<count> or <TPU>=<topology>.')
_BUILD_TARGET = flags.DEFINE_string(
    'build_target',
    '//third_party/deepmind/exedec/spec_decomposition:end_to_end_predict',
    'The build target to run.')

# Flags for dataset info.
_DATASET_TYPE = flags.DEFINE_enum(
    'dataset_type', 'robustfill',
    ['robustfill', 'deepcoder'],
    'The kind of dataset to use.')
_EXPERIMENTS = flags.DEFINE_string(
    'experiments', '',
    'A comma-separated list of experiment.Experiment names to use.')
_TEST_DATASET_FORMAT = flags.DEFINE_string(
    'test_dataset_format', None,
    'Filepattern for TFRecord test dataset.')
_NUM_TEST_BATCHES = flags.DEFINE_integer(
    'num_test_batches', 1000,
    'Number of test batches.')
_NUM_EXAMPLES = flags.DEFINE_integer(
    'num_examples', 4,
    'Number of examples per task.')
_MAX_IO_LENGTH = flags.DEFINE_integer(
    'max_io_length', 200,
    'Maximum number of characters in input/output strings.')
_MAX_PROGRAM_LENGTH = flags.DEFINE_integer(
    'max_program_length', 100,
    'Maximum number of characters in a program part.')
_MAX_SPEC_PART_LENGTH = flags.DEFINE_integer(
    'max_spec_part_length', 85,
    'Maximum number of characters in a spec part.')

# Flags for model hyperparameters.
_SPEC_DECOMPOSER_PATH_FORMAT = flags.DEFINE_string(
    'spec_decomposer_path_format', None,
    'Directory with saved weights for the SpecDecomposerModel.')
_SYNTHESIZER_PATH_FORMAT = flags.DEFINE_string(
    'synthesizer_path_format', None,
    'Directory with saved weights for the SynthesizerModel.')
_JOINT_PATH_FORMAT = flags.DEFINE_string(
    'joint_path_format', None,
    'Directory with saved weights for the JointModel.')
_SEED = flags.DEFINE_multi_integer(
    'seed', [10, 20, 30, 40, 50],
    'Seeds used for training.')
_EMBEDDING_DIM = flags.DEFINE_integer(
    'embedding_dim', 512,
    'Embedding dimension.')
_HIDDEN_DIM = flags.DEFINE_integer(
    'hidden_dim', 1024,
    'Hidden dimension.')
_SPEC_DECOMPOSER_NUM_POSITION_BUCKETS = flags.DEFINE_integer(
    'spec_decomposer_num_position_buckets', 32,
    'Number of relative attention position buckets in SpecDecomposer.')
_SYNTHESIZER_NUM_POSITION_BUCKETS = flags.DEFINE_integer(
    'synthesizer_num_position_buckets', 32,
    'Number of relative attention position buckets in Synthesizer.')
_SPEC_DECOMPOSER_MAX_DISTANCE = flags.DEFINE_integer(
    'spec_decomposer_max_distance', 128,
    'Max distance for relative attention positions in SpecDecomposer.')
_SYNTHESIZER_MAX_DISTANCE = flags.DEFINE_integer(
    'synthesizer_max_distance', 20,
    'Max distance for relative attention positions in Synthesizer.')
_SPEC_DECOMPOSER_MAX_PROGRAM_CROSS_EMBED_DISTANCE = flags.DEFINE_integer(
    'spec_decomposer_max_program_cross_embed_distance', 800,
    'Max distance for relative attention positions in SpecDecomposer.')
_SYNTHESIZER_MAX_PROGRAM_CROSS_EMBED_DISTANCE = flags.DEFINE_integer(
    'synthesizer_max_program_cross_embed_distance', 100,
    'Max distance for relative attention positions in Synthesizer.')
_USE_RELATIVE_ATTENTION = flags.DEFINE_bool(
    'use_relative_attention', True,
    'Whether to use relative positonal embeddings.')
_ALIGNED_RELATIVE_ATTENTION = flags.DEFINE_enum(
    'aligned_relative_attention', 'true',
    ['true', 'false', 'both'],
    'Whether to align relative attention positions between targets and encoded '
    'I/O examples.',
    case_sensitive=False)
_CORRUPTION_RATE = flags.DEFINE_multi_float(
    'corruption_rate', [0.0],
    'Next part corruption rate for the SynthesizerModel.')

# Flags for end-to-end prediction settings.
_BEAM_SIZE = flags.DEFINE_multi_integer(
    'beam_size', [10],
    'Beam search size.')
_PREDICTION_TYPE = flags.DEFINE_enum(
    'prediction_type', 'separate',
    ['separate', 'joint'],
    'Whether to use separate models (SpecDecomposerModel and then '
    'SynthesizerModel) or one joint prediction model.')
_DETECT_INVALID = flags.DEFINE_enum(
    'detect_invalid', 'true',
    ['true', 'false', 'both'],
    'Whether to detect invalid beam elements and mark them as finished.',
    case_sensitive=False)
_USE_EXECUTION = flags.DEFINE_enum(
    'use_execution', 'true',
    ['true', 'false', 'both'],
    'Whether to guide beam search with program execution results.',
    case_sensitive=False)
_DISCARD_REPEAT_FUNCTIONALITY = flags.DEFINE_enum(
    'discard_repeat_functionality', 'true',
    ['true', 'false', 'both'],
    'Whether to mark duplicate program functionality in a beam as invalid.',
    case_sensitive=False)

# DeepCoder settings.
_DEEPCODER_MAX_LIST_LENGTH = flags.DEFINE_integer(
    'deepcoder_max_list_length', 5,
    'The maximum length of a DeepCoder list input.')
_DEEPCODER_MAX_INT = flags.DEFINE_integer(
    'deepcoder_max_int', 50,
    'The maximum value of a DeepCoder int.')


def create_job(experiment, save_dir):
  """Creates a job."""
  requirements = xm.JobRequirements(
      service_tier=xm.ServiceTier.PROD,
      **_ACCELERATOR.value,
  )

  # Static arguments that don't change in the hyperparameter sweep.
  args = collections.OrderedDict([
      # Experiment setup.
      ('save_dir', save_dir),
      # Dataset info.
      ('dataset_type', _DATASET_TYPE.value),
      ('test_dataset_format', _TEST_DATASET_FORMAT.value),
      ('num_test_batches', _NUM_TEST_BATCHES.value),
      ('num_examples', _NUM_EXAMPLES.value),
      ('max_io_length', _MAX_IO_LENGTH.value),
      ('max_program_length', _MAX_PROGRAM_LENGTH.value),
      ('max_spec_part_length', _MAX_SPEC_PART_LENGTH.value),
      # Model hyperparameters.
      ('spec_decomposer_path_format', _SPEC_DECOMPOSER_PATH_FORMAT.value),
      ('synthesizer_path_format',
       _SYNTHESIZER_PATH_FORMAT.value if _PREDICTION_TYPE.value == 'separate'
       else _JOINT_PATH_FORMAT.value),
      ('embedding_dim', _EMBEDDING_DIM.value),
      ('hidden_dim', _HIDDEN_DIM.value),
      ('num_heads', 4),
      ('num_layers', 3),
      ('spec_decomposer_num_position_buckets',
       _SPEC_DECOMPOSER_NUM_POSITION_BUCKETS.value),
      ('synthesizer_num_position_buckets',
       _SYNTHESIZER_NUM_POSITION_BUCKETS.value),
      ('spec_decomposer_max_distance', _SPEC_DECOMPOSER_MAX_DISTANCE.value),
      ('synthesizer_max_distance', _SYNTHESIZER_MAX_DISTANCE.value),
      ('spec_decomposer_max_program_cross_embed_distance',
       _SPEC_DECOMPOSER_MAX_PROGRAM_CROSS_EMBED_DISTANCE.value),
      ('synthesizer_max_program_cross_embed_distance',
       _SYNTHESIZER_MAX_PROGRAM_CROSS_EMBED_DISTANCE.value),
      ('use_relative_attention', _USE_RELATIVE_ATTENTION.value),
      ('aligned_relative_attention', _ALIGNED_RELATIVE_ATTENTION.value),
      # DeepCoder settings.
      ('deepcoder_max_list_length', _DEEPCODER_MAX_LIST_LENGTH.value),
      ('deepcoder_max_int', _DEEPCODER_MAX_INT.value),
  ])

  env_vars = collections.OrderedDict([
      ('XLA_PYTHON_CLIENT_MEM_FRACTION', '.5'),
  ])

  [executable] = experiment.package([
      xm.bazel_binary(
          label=_BUILD_TARGET.value,
          executor_spec=xm_local.Local.Spec(),
          args=args,
          env_vars=env_vars,
      )])

  executor = xm_local.Local(requirements=requirements)

  return xm.Job(executable, executor)


def get_boolean_options(flag):
  flag_value = flag.value.lower()
  if flag_value == 'true':
    return [True]
  elif flag_value == 'false':
    return [False]
  elif flag_value == 'both':
    return [True, False]
  else:
    raise ValueError(f'Unhandled boolean option: {flag_value}')


def product_sweep(
    all_name_and_values: list[tuple[str, list[Any]]],
) -> dict[str, dict[str, Any]]:
  names = []
  value_lists = []
  for name, value_list in all_name_and_values:
    names.append(name)
    value_lists.append(value_list)
  return [{'args': dict(zip(names, choices))}
          for choices in itertools.product(*value_lists)]


def get_sweep():
  """Returns a hyperparameter sweep."""
  sweep = product_sweep([
      ('seed', _SEED.value),
      ('experiment', _EXPERIMENTS.value.upper().split(',')),
      ('beam_size', _BEAM_SIZE.value),
      ('prediction_type', [_PREDICTION_TYPE.value]),
      ('detect_invalid', get_boolean_options(_DETECT_INVALID)),
      ('use_execution', get_boolean_options(_USE_EXECUTION)),
      ('discard_repeat_functionality',
       get_boolean_options(_DISCARD_REPEAT_FUNCTIONALITY)),
      ('aligned_relative_attention',
       get_boolean_options(_ALIGNED_RELATIVE_ATTENTION)
       if _PREDICTION_TYPE.value == 'separate'
       else [False]),
      ('corruption_rate',
       _CORRUPTION_RATE.value
       if _PREDICTION_TYPE.value == 'separate'
       else [0.0]),
  ])

  # Some arg settings are incompatible with each other. See
  # end_to_end_predict.py for reasonings.
  filtered_sweep = []
  for sweep_element in sweep:
    args = sweep_element['args']
    if args['prediction_type'] == 'joint' and not args['use_execution']:
      continue
    if not args['use_execution'] and args['discard_repeat_functionality']:
      continue
    if not args['detect_invalid'] and args['discard_repeat_functionality']:
      continue
    filtered_sweep.append(sweep_element)
  return filtered_sweep


def main(_):
  """Launch the experiment."""

  with xm_local.create_experiment(
      experiment_title=_EXP_TITLE.value) as experiment:

    save_dir = os.path.join(_SAVE_DIR.value, _EXP_TITLE.value)
    job = create_job(experiment, save_dir)
    for sweep_args in get_sweep():
      experiment.add(job, args=sweep_args)


if __name__ == '__main__':
  app.run(main)
