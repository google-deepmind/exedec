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

"""Launches potentially multiple runs of exedec/spec_decomposition/train.py."""

import collections
import itertools
import os
import subprocess
from typing import Any

from absl import app
from absl import flags

# Flags for experiment setup.
_SEED = flags.DEFINE_multi_integer(
    'seed', [10, 20, 30, 40, 50],
    'Seeds used for training.')
_SAVE_DIR = flags.DEFINE_string(
    'save_dir', None,
    'Directory for checkpoints and TensorBoard files.')
_PREDICT_ONLY = flags.DEFINE_bool(
    'predict_only', False,
    'Whether to only do beam search prediction without training.')

# Flags for dataset info.
_DATASET_TYPE = flags.DEFINE_enum(
    'dataset_type', 'robustfill',
    ['robustfill', 'deepcoder'],
    'The kind of dataset to use.')
_EXPERIMENTS = flags.DEFINE_string(
    'experiments', '',
    'A comma-separated list of experiment.Experiment names train on.')
_DATASET_DIR = flags.DEFINE_string(
    'dataset_dir', None,
    'Directory to find TFRecord datasets for train and test.')
_NUM_EXAMPLES = flags.DEFINE_integer(
    'num_examples', 4,
    'Number of examples per task.')
_MAX_INPUT_LENGTH = flags.DEFINE_integer(
    'max_input_length', 120,
    'Maximum number of characters in the model input.')
_PREDICT_MAX_INPUT_LENGTH = flags.DEFINE_integer(
    'predict_max_input_length', 200,
    'Maximum number of characters in the model input for prediction.')
_MAX_TARGET_LENGTH = flags.DEFINE_integer(
    'max_target_length', 100,
    'Maximum number of tokens in the prediction target.')

# Used to compute input/target lengths and relative attention distances for
# DeepCoder.
_MAX_PROGRAM_ARITY = flags.DEFINE_integer(
    'max_program_arity', 2, 'Maximum number of inputs.')
_MAX_NUM_STATEMENTS = flags.DEFINE_integer(
    'max_num_statements', 5, 'Maximum number of statements in a program.')
_DEEPCODER_MAX_LIST_LENGTH = flags.DEFINE_integer(
    'deepcoder_max_list_length', 5,
    'The maximum length of a DeepCoder list input.')
_DEEPCODER_MAX_INT = flags.DEFINE_integer(
    'deepcoder_max_int', 50,
    'The maximum value of a DeepCoder int.')

# Flags for training settings.
_NUM_TRAIN_STEPS = flags.DEFINE_integer(
    'num_train_steps', 500_000,
    'Number of training steps.')
_NUM_EVAL_STEPS = flags.DEFINE_integer(
    'num_eval_steps', 10,
    'Number of evaluation steps.')
_LOG_FREQ = flags.DEFINE_integer(
    'log_freq', 2000,
    'Number of steps between training logs.')
_EVAL_FREQ = flags.DEFINE_integer(
    'eval_freq', 10000,
    'Number of steps between eval.')
_PREDICT_FREQ = flags.DEFINE_integer(
    'predict_freq', 50000,
    'Number of steps between prediction (beam search).')
_CHECKPOINT_FREQ = flags.DEFINE_integer(
    'checkpoint_freq', 50000,
    'Number of steps between checkpoint saves.')

# Flags for model hyperparameters.
_MODEL_TYPE = flags.DEFINE_enum(
    'model_type', 'spec_decomposer_model',
    ['spec_decomposer_model', 'synthesizer_model', 'joint_model',
     'baseline_model'],
    'Which model to train.')
_LR = flags.DEFINE_multi_float(
    'lr', [2e-4],
    'Options for learning rate.')
_PER_DEVICE_BATCH_SIZE = flags.DEFINE_integer(
    'per_device_batch_size', 16,
    'Number of program tasks in a batch.')
_EMBEDDING_DIM = flags.DEFINE_multi_integer(
    'embedding_dim', [512],
    'Options for embedding dimension.')
_HIDDEN_DIM = flags.DEFINE_multi_integer(
    'hidden_dim', [1024],
    'Options for hidden dimension.')
_DROPOUT_RATE = flags.DEFINE_multi_float(
    'dropout_rate', [0.1],
    'Options for dropout rate.')
_ATTENTION_DROPOUT_RATE = flags.DEFINE_multi_float(
    'attention_dropout_rate', [0.1],
    'Options for attention dropout rate.')
_NUM_POSITION_BUCKETS = flags.DEFINE_multi_integer(
    'num_position_buckets', [32],
    'Options for number of relative attention position buckets.')
_MAX_DISTANCE = flags.DEFINE_multi_integer(
    'max_distance', [128],
    'Options for max relative attention distance.')
_MAX_PROGRAM_CROSS_EMBED_DISTANCE = flags.DEFINE_multi_integer(
    'max_program_cross_embed_distance', [128],
    'Options for max relative attention distance.')
_ALIGNED_RELATIVE_ATTENTION = flags.DEFINE_multi_integer(
    'aligned_relative_attention', [1],
    'Options for whether to align relative attention positions between targets '
    'and encoded I/O examples, 0 for False and 1 for True.')

_SYNTHESIZER_CORRUPTED_NEXT_PART_RATE = flags.DEFINE_multi_float(
    'synthesizer_corrupted_next_part_rate', [0.0],
    'The fraction of examples that use the corrupted next part. Ignored if not '
    'training the SynthesizerModel.')


def compute_lengths():
  """Computes lengths and distances based on dataset, model, and other flags."""
  if _DATASET_TYPE.value == 'deepcoder':
    # One object in a ProgramState tokenizes to a variable, equals, at most 2
    # list brackets, separator token, and <= `deepcoder_max_list_length`
    # list elements.
    object_token_length = _DEEPCODER_MAX_LIST_LENGTH.value + 5
    # A ProgramState input can have (max_program_arity + max_num_statements - 1)
    # objects, as the input spec for the last step of synthesis for the longest
    # program.
    max_input_objects = _MAX_PROGRAM_ARITY.value + _MAX_NUM_STATEMENTS.value - 1
    # The max number of tokens in a ProgramState input spec. Note that "input"
    # to the model includes both the ProgramState and the desired output. The
    # desired output, being only 1 object per example, will always fit in
    # `max_input_length` tokens.
    max_input_length = max_input_objects * object_token_length

    # Use the same input length for prediction. Programs in the train and test
    # splits have the same maximum number of statements (5 for the NONE
    # generalization task).
    predict_max_input_length = max_input_length

    # The output of the SpecDecomposerModel is one output object per example.
    max_output_prediction_length = _NUM_EXAMPLES.value * object_token_length

    # The output of the SynthesizerModel or JointModel is the RHS of a
    # statement, which has at most 1 operation, 1 lambda, and 2 variables. We
    # leave extra tokens for BOS and EOS (might not be truly necessary but
    # doesn't hurt).
    max_program_length = 6

    if _MODEL_TYPE.value == 'spec_decomposer_model':
      max_target_length = max_output_prediction_length
    elif _MODEL_TYPE.value in ['synthesizer_model', 'joint_model']:
      max_target_length = max_program_length
    else:
      assert _MODEL_TYPE.value == 'baseline_model'
      max_program_parts = _MAX_PROGRAM_ARITY.value + _MAX_NUM_STATEMENTS.value
      # A program part has a LHS variable, =, RHS (4 tokens), and separator.
      max_program_part_length = 7
      max_target_length = max_program_part_length * max_program_parts

    # Maximum distance for relative attention (self-attention and cross
    # attention of inputs and outputs one example at a time, and self-attention
    # of the predicted target).
    max_distance = max(max_input_length, max_target_length)
    # Maximum distance for relative cross attention from the predicted target to
    # the I/O embedding, which has all examples concatenated.
    max_program_cross_embed_distance = max(
        max_input_length * _NUM_EXAMPLES.value, max_target_length)

    max_distance_options = [max_distance]
    max_program_cross_embed_distance_options = [
        max_program_cross_embed_distance]

  elif _DATASET_TYPE.value == 'robustfill':
    # Just use the values set via flags.
    max_input_length = _MAX_INPUT_LENGTH.value
    predict_max_input_length = _PREDICT_MAX_INPUT_LENGTH.value
    max_target_length = _MAX_TARGET_LENGTH.value
    # These are multi integer flags, so they are already int lists.
    max_distance_options = _MAX_DISTANCE.value
    max_program_cross_embed_distance_options = (
        _MAX_PROGRAM_CROSS_EMBED_DISTANCE.value)

  else:
    raise ValueError(f'Unsupported dataset type: {_DATASET_TYPE.value}')

  print(f'For dataset {_DATASET_TYPE.value} and model {_MODEL_TYPE.value}:')
  print(f'  max_input_length = {max_input_length}')
  print(f'  predict_max_input_length = {predict_max_input_length}')
  print(f'  max_target_length = {max_target_length}')
  print(f'  max_distance_options = {max_distance_options}')
  print(f'  max_program_cross_embed_distance_options = '
        f'{max_program_cross_embed_distance_options}')

  return (max_input_length, predict_max_input_length, max_target_length,
          max_distance_options, max_program_cross_embed_distance_options)


def product_sweep(
    all_name_and_values: list[tuple[str, list[Any]]],
) -> dict[str, dict[str, Any]]:
  names = []
  value_lists = []
  for name, value_list in all_name_and_values:
    names.append(name)
    value_lists.append(value_list)
  return [dict(zip(names, choices))
          for choices in itertools.product(*value_lists)]


def get_sweep(max_distance_options, max_program_cross_embed_distance_options):
  """Returns a hyperparameter sweep."""
  return product_sweep([
      ('experiment', _EXPERIMENTS.value.upper().split(',')),
      ('seed', _SEED.value),
      ('lr', _LR.value),
      ('embedding_dim', _EMBEDDING_DIM.value),
      ('hidden_dim', _HIDDEN_DIM.value),
      ('use_relative_attention', [True]),
      ('dropout_rate', _DROPOUT_RATE.value),
      ('attention_dropout_rate', _ATTENTION_DROPOUT_RATE.value),
      ('num_position_buckets', _NUM_POSITION_BUCKETS.value),
      ('max_distance', max_distance_options),
      ('max_program_cross_embed_distance',
       max_program_cross_embed_distance_options),
      ('aligned_relative_attention',
       [bool(x) for x in _ALIGNED_RELATIVE_ATTENTION.value]
       if _MODEL_TYPE.value == 'spec_decomposer_model'
       else [False]),
      ('synthesizer_corrupted_next_part_rate',
       _SYNTHESIZER_CORRUPTED_NEXT_PART_RATE.value
       if _MODEL_TYPE.value == 'synthesizer_model'
       else [0.0]),
  ])


def run_job(job_args):
  subprocess.run(
      args=(['python', '-m', 'spec_decomposition.train']
            + [f'--{k}={v}' for k, v in job_args.items()]),
      check=True,
  )


def main(_):
  """Launch the experiment."""

  save_dir = os.path.join(_SAVE_DIR.value, _MODEL_TYPE.value)

  (max_input_length,
   predict_max_input_length,
   max_target_length,
   max_distance_options,
   max_program_cross_embed_distance_options) = compute_lengths()

  # Static arguments that don't change in the hyperparameter sweep.
  static_args = collections.OrderedDict([
      # Experiment setup.
      ('save_dir', save_dir),
      ('predict_only', _PREDICT_ONLY.value),
      # Dataset info.
      ('dataset_type', _DATASET_TYPE.value),
      ('dataset_dir', _DATASET_DIR.value),
      ('num_examples', _NUM_EXAMPLES.value),
      ('max_input_length', max_input_length),
      ('predict_max_input_length', predict_max_input_length),
      ('max_target_length', max_target_length),
      # Training settings.
      ('num_train_steps', _NUM_TRAIN_STEPS.value),
      ('num_eval_steps', _NUM_EVAL_STEPS.value),
      ('log_freq', _LOG_FREQ.value),
      ('eval_freq', _EVAL_FREQ.value),
      ('predict_freq', _PREDICT_FREQ.value),
      ('checkpoint_freq', _CHECKPOINT_FREQ.value),
      # Model hyperparameters.
      ('model_type', _MODEL_TYPE.value),
      ('per_device_batch_size', _PER_DEVICE_BATCH_SIZE.value),
      ('num_heads', 4),
      ('num_layers', 3),
      # DeepCoder-specific settings.
      ('deepcoder_max_list_length', _DEEPCODER_MAX_LIST_LENGTH.value),
      ('deepcoder_max_int', _DEEPCODER_MAX_INT.value),
  ])

  # Run training jobs in sequence.
  for sweep_args in get_sweep(max_distance_options,
                              max_program_cross_embed_distance_options):
    job_args = static_args | sweep_args
    run_job(job_args)


if __name__ == '__main__':
  app.run(main)
