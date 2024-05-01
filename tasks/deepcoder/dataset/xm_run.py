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

"""XM Launcher for exedec/tasks/deepcoder/dataset:write_data.

Use from the `run_data_generation.sh` script.
"""

import collections
import itertools

from absl import app
from absl import flags

from xmanager import xm
from xmanager import xm_local


# Flags for experiment setup.
_EXP_TITLE = flags.DEFINE_string(
    'exp_title', 'deepcoder_dataset', 'Title of the experiment.')
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
_MAX_PROGRAM_ARITY = flags.DEFINE_integer(
    'max_program_arity', 2, 'Maximum number of inputs.')
_NUM_EXAMPLES = flags.DEFINE_integer(
    'num_examples', 3, 'Number of input/output examples per task.')
_DEEPCODER_MAX_LIST_LENGTH = flags.DEFINE_integer(
    'deepcoder_max_list_length', 5,
    'The maximum length of a DeepCoder list input.')
_DEEPCODER_MAX_INT = flags.DEFINE_integer(
    'deepcoder_max_int', 50,
    'The maximum value of a DeepCoder int.')

# Flags for dataset size.
_NUM_SHARDS_TRAIN = flags.DEFINE_integer(
    'num_shards_train', 1000,
    'Number of shards (jobs) in the train split.')
_NUM_PROGRAMS_PER_SEARCH_TRAIN = flags.DEFINE_integer(
    'num_programs_per_search_train', 1000,
    'Number of programs to generate per train search.')
_NUM_SEARCHES_TRAIN = flags.DEFINE_integer(
    'num_searches_train', 128,
    'Number of searches to perform per shard of the train split.')
_NUM_SHARDS_TEST = flags.DEFINE_integer(
    'num_shards_test', 100,
    'Number of shards (jobs) in the test or valid splits.')
_NUM_PROGRAMS_PER_SEARCH_TEST = flags.DEFINE_integer(
    'num_programs_per_search_test', 10,
    'Number of programs to generate per test or valid search.')
_NUM_SEARCHES_TEST = flags.DEFINE_integer(
    'num_searches_test', 10,
    'Number of searches to perform per shard of the test or valid split.')


BUILD_TARGET = '//third_party/deepmind/exedec/tasks/deepcoder/dataset:write_data'


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
      ('max_program_arity', _MAX_PROGRAM_ARITY.value),
      ('deepcoder_max_list_length', _DEEPCODER_MAX_LIST_LENGTH.value),
      ('deepcoder_max_int', _DEEPCODER_MAX_INT.value),
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

    job = create_job(xm_experiment)

    for experiment, split in itertools.product(experiments, splits):
      if split == 'train':
        num_shards = _NUM_SHARDS_TRAIN.value
        num_programs_per_search = _NUM_PROGRAMS_PER_SEARCH_TRAIN.value
        num_searches = _NUM_SEARCHES_TRAIN.value
      else:
        num_shards = _NUM_SHARDS_TEST.value
        num_programs_per_search = _NUM_PROGRAMS_PER_SEARCH_TEST.value
        num_searches = _NUM_SEARCHES_TEST.value

      for shard_id in range(num_shards):
        xm_experiment.add(job, args={'args': {
            'num_shards': num_shards,
            'shard_id': shard_id,
            'experiment': experiment,
            'split': split,
            'num_programs_per_search': num_programs_per_search,
            'num_searches': num_searches,
        }})


if __name__ == '__main__':
  app.run(main)
