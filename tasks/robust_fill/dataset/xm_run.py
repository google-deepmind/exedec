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

"""XM Launcher for exedec/tasks/robust_fill/dataset:write_data."""

import collections
import itertools

from absl import app
from absl import flags

from xmanager import xm
from xmanager import xm_local

# Flags for experiment setup.
_EXP_TITLE = flags.DEFINE_string(
    'exp_title', 'robustfill_dataset', 'Title of the experiment.')
_SEED = flags.DEFINE_integer(
    'seed', 0, 'Starting seed.')
_SAVE_DIR = flags.DEFINE_string(
    'save_dir', None, 'Directory to save dataset in.')

# Flags for dataset info.
_EXPERIMENTS = flags.DEFINE_string(
    'experiments', '',
    'A comma-separated list of experiment.Experiment names to '
    'generate data for.')
_SPLITS = flags.DEFINE_string(
    'splits', 'train,valid,test',
    'A comma-separated list of dataset splits to generate.')
_NUM_TRAIN_SHARDS = flags.DEFINE_integer(
    'num_train_shards', 128,
    'Number of shards (jobs) in the train split. Other splits use 1 shard.')
_NUM_TRAIN_PROGRAMS_PER_SHARD = flags.DEFINE_integer(
    'num_train_programs_per_shard', 1000000,
    'Number of programs to generate per shard of the train split.')
_NUM_TEST_PROGRAMS = flags.DEFINE_integer(
    'num_test_programs', 10000,
    'Number of programs in the test/valid splits (in 1 shard).')
_NUM_EXAMPLES = flags.DEFINE_integer(
    'num_examples', 4, 'Number of input/output examples per task.')
_MAX_INPUT_LENGTH = flags.DEFINE_integer(
    'max_input_length', 20,
    'Maximum number of characters in input strings.')

BUILD_TARGET = '//third_party/deepmind/exedec/tasks/robust_fill/dataset:write_data'


def create_job(xm_experiment):
  """Creates a job."""
  requirements = xm.JobRequirements()

  # Static arguments that don't change per worker.
  args = collections.OrderedDict([
      # Experiment setup.
      ('seed', _SEED.value),
      ('save_dir', _SAVE_DIR.value),
      # Dataset info.
      ('num_examples', _NUM_EXAMPLES.value),
      ('max_input_length', _MAX_INPUT_LENGTH.value),
  ])

  [executable] = xm_experiment.package([
      xm.bazel_binary(
          label=BUILD_TARGET,
          executor_spec=xm_local.LocalSpec(),
          args=args,
      )])

  executor = xm_local.Local(requirements=requirements)
  return xm.Job(executable, executor)


def main(_):
  """Launch the experiment."""

  experiments = _EXPERIMENTS.value.upper().split(',')
  splits = _SPLITS.value.lower().split(',')
  assert all(split in {'train', 'valid', 'test'} for split in splits)

  with xm_local.create_experiment(
      experiment_title=_EXP_TITLE.value) as xm_experiment:
    annotations = xm_experiment.context.annotations
    annotations.add_tags('spec_decomposition',
                         'dataset_generation',
                         'robustfill')

    job = create_job(xm_experiment)

    for experiment, split in itertools.product(experiments, splits):
      if split == 'train':
        num_shards = _NUM_TRAIN_SHARDS.value
        num_programs = _NUM_TRAIN_PROGRAMS_PER_SHARD.value
      else:
        num_shards = 1
        num_programs = _NUM_TEST_PROGRAMS.value

      for shard_id in range(num_shards):
        xm_experiment.add(job, args={'args': {
            'num_shards': num_shards,
            'shard_id': shard_id,
            'experiment': experiment,
            'split': split,
            'num_programs': num_programs,
        }})


if __name__ == '__main__':
  app.run(main)
