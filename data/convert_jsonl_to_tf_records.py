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

"""Converts test_data/*/*.jsonl files to TFRecords files.

Run directly with python3. Needs TensorFlow and absl-py dependencies.
"""

from collections.abc import Sequence
import json
import os
from typing import Any

from absl import app
import tensorflow as tf

TEST_DATA_DIR = 'test_data/'
RESULT_DIR = os.path.expanduser('~/exedec_data/test_data_tfrecords')


def _bytes_feature(strs: list[str | bytes]):
  """Returns a bytes_list Feature from a list of strings."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(
      value=[s if isinstance(s, bytes) else str.encode(s) for s in strs]))


def serialize_task(task: dict[str, Any]):
  """Creates a tf.Example message for a PBE task."""
  feature = {
      'inputs': _bytes_feature(task['inputs']),
      'outputs': _bytes_feature(task['outputs']),
      'program': _bytes_feature([task['program']]),
  }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for dataset in sorted(os.listdir(TEST_DATA_DIR)):
    print(f'Processing {dataset=}')
    for filename in sorted(os.listdir(os.path.join(TEST_DATA_DIR, dataset))):
      print(f'  Processing {filename=}')
      with open(os.path.join(TEST_DATA_DIR, dataset, filename)) as f:
        data = [json.loads(line) for line in f.readlines()]
      generalization_task = filename.removesuffix('.jsonl')
      this_result_dir = os.path.join(
          RESULT_DIR, dataset, generalization_task + '_data')
      os.makedirs(this_result_dir)
      out_filename = os.path.join(
          this_result_dir, 'entire_programs_test.tf_records-00000-of-00001')
      with tf.io.TFRecordWriter(out_filename) as writer:
        for task in data:
          writer.write(serialize_task(task))
      print(f'    Wrote to {out_filename=}')


if __name__ == '__main__':
  app.run(main)
