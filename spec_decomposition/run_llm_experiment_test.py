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

"""Tests for run_llm_experiment.py."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver

from exedec.spec_decomposition import cached_llm_access
from exedec.spec_decomposition import llm_utils
from exedec.spec_decomposition import run_llm_experiment


DEEPCODER_1 = llm_utils.DatasetElement(
    inputs={'x0': [[], [1, 0, 6, 9, 1], [3, 7, 1, 4]],
            'x1': [[0], [9], [-3, -1]]},
    outputs=[[], [5, 14, 15], [4, 5, 9]],
    dsl_program=('x0 = INPUT | x1 = INPUT | x2 = Scanl1 (-) x0 | '
                 'x3 = Map (*(-1)) x2 | x4 = Filter (>0) x3'),
    python_program='''
def program(x0, x1):
  x2 = dsl.Scanl1(dsl.SUBTRACT, x0)
  x3 = dsl.Map(dsl.NEGATE, x2)
  x4 = dsl.Filter(dsl.IS_POSITIVE, x3)
  return x4
'''.strip(),
)
DEEPCODER_2 = llm_utils.DatasetElement(
    inputs={'x3': [[1, 2, 3], [10, -10], [45]]},
    outputs=[6, 0, 45],
    dsl_program=None,  # Unused.
    python_program='''
def program(x3):
  x7 = dsl.Sum(x3)
  return x7
'''.strip(),
)
DEEPCODER_3 = llm_utils.DatasetElement(
    inputs={'x0': [5, 1, 2], 'x1': [[1, 2, 3], [6, 5, 4], [9, 7, 8]]},
    outputs=[[1, 2, 3], [6], [7, 9]],
    dsl_program=None,  # Unused.
    python_program='''
def program(x0, x1):
  x2 = dsl.Take(x0, x1)
  x3 = dsl.Sort(x2)
  return x3
'''.strip(),
)

ROBUSTFILL_1 = llm_utils.DatasetElement(
    inputs=['#My##:Gxbo[Ned[Er%', '#%$Ua.Qaeq?Opa%Kcr#',
            "%{Eos#(Mdjt#'Yi{Oclf", '%##Tq@Fh#Xza#?Fdlu'],
    outputs=['k[MY##:GXBO[NED[ER%8y##:Gxbo[Ned[',
             'kK%$UA.QAEQ?OPA%KCR#8aUa.Qaeq?Opa%',
             "kO{EOS#(MDJT#'YI{OCLF8osos#(Mdjt#'Yi",
             'kF##TQ@FH#XZA#?FDLU8qTq@Fh#Xza#?F'],
    dsl_program=('4 29|7 109 211|3 8 111 17 109 216|'
                 '3 15 109 216 79 7 106 216|5 219 230'),
    python_program='''
def program(x):
  parts = [
      dsl.Const('k'),
      dsl.GetToken(x, dsl.Type.CHAR, -4),
      dsl.ToCase(dsl.Remove(x, dsl.Type.CHAR, 1), dsl.Case.ALL_CAPS),
      dsl.Substitute(dsl.GetToken(x, dsl.Type.PROP_CASE, 1), dsl.Type.CHAR, 1, '8'),
      dsl.SubStr(x, 4, 15),
  ]
  return ''.join(parts)
'''.strip(),
)
ROBUSTFILL_2 = llm_utils.DatasetElement(
    inputs=['apple', 'banana', 'clementine', 'durian'],
    outputs=['Apple!', 'Banana!', 'Clementine!', 'Durian!'],
    dsl_program=None,  # Unused.
    python_program='''
def program(x):
  parts = [
      dsl.ToCase(x, dsl.Case.PROPER),
      dsl.Const('!'),
  ]
  return ''.join(parts)
'''.strip(),
)
ROBUSTFILL_3 = llm_utils.DatasetElement(
    inputs=['x y', 'a b', '1 2', '! ?'],
    outputs=['y.x', 'b.a', '2.1', '?.!'],
    dsl_program=None,  # Unused.
    python_program='''
def program(x):
  parts = [
      dsl.GetToken(x, dsl.Type.CHAR, -1),
      dsl.Const('.'),
      dsl.GetToken(x, dsl.Type.CHAR, 1),
  ]
  return ''.join(parts)
'''.strip(),
)

BAD_RESPONSE_1 = '''
  dsl.NotAnOperation()
  return None
```

More text...
'''

BAD_RESPONSE_2 = '''
  bad indentation
    invalid python code
```

More text...
'''


def _response(dataset_element: llm_utils.DatasetElement) -> str:
  return f'{dataset_element.python_program}\n```\n\nMore text...'


class RunLlmExperimentTest(absltest.TestCase):

  @mock.patch.object(run_llm_experiment, 'query_llm')
  @mock.patch.object(llm_utils, 'load_jsonl_dataset')
  def test_run_entire_experiment(self, mock_load_jsonl_dataset, mock_query_llm):
    # Don't actually write results to disk.
    saved_results_format = run_llm_experiment.RESULTS_FORMAT
    run_llm_experiment.RESULTS_FORMAT = os.path.join(
        self.create_tempdir().full_path, '{model}_{num_samples}-samples.json')

    model = 'mock_model'

    # Use a temporary cache location.
    cache_dir = self.create_tempdir().full_path
    model_name = model.split('/')[-1]
    cached_llm_access.init_cache(cache_dir, model_name)

    # Check that the examples actually work.
    for e in [DEEPCODER_1, DEEPCODER_2, DEEPCODER_3]:
      self.assertEqual(
          llm_utils.run_program(e.python_program, e.inputs, 'deepcoder'),
          e.outputs)
    for e in [ROBUSTFILL_1, ROBUSTFILL_2, ROBUSTFILL_3]:
      self.assertEqual(
          llm_utils.run_program(e.python_program, e.inputs, 'robustfill'),
          e.outputs)

    # load_datasets is called 6 times for the DeepCoder generalization tasks and
    # then 6 more times for RobustFill. If we re-use a prompt, the LLM caching
    # will lead to duplicate samples, so ensure that the prompts are different.
    mock_load_jsonl_dataset.side_effect = [
        [(DEEPCODER_2, [DEEPCODER_1])],
        [(DEEPCODER_3, [DEEPCODER_1])],
        [(DEEPCODER_1, [DEEPCODER_2])],
        [(DEEPCODER_3, [DEEPCODER_2])],
        [(DEEPCODER_1, [DEEPCODER_3])],
        [(DEEPCODER_2, [DEEPCODER_3])],

        [(ROBUSTFILL_2, [ROBUSTFILL_1])],
        [(ROBUSTFILL_3, [ROBUSTFILL_1])],
        [(ROBUSTFILL_1, [ROBUSTFILL_2])],
        [(ROBUSTFILL_3, [ROBUSTFILL_2])],
        [(ROBUSTFILL_1, [ROBUSTFILL_3])],
        [(ROBUSTFILL_2, [ROBUSTFILL_3])],
    ] * 2  # *2 because we'll repeat the experiment to test caching.
    # The corresponding 2 samples per prompt (since there is only 1 test problem
    # per generalization task).
    model_samples = [
        [BAD_RESPONSE_1, BAD_RESPONSE_2],  # Failure.
        [BAD_RESPONSE_1, _response(DEEPCODER_2)],  # Failure.
        [_response(DEEPCODER_1), BAD_RESPONSE_1],  # Success.
        [BAD_RESPONSE_2, _response(DEEPCODER_3)],  # Success.
        [_response(DEEPCODER_2), BAD_RESPONSE_1],  # Failure.
        [_response(DEEPCODER_2), BAD_RESPONSE_1],  # Success.

        [_response(ROBUSTFILL_1), BAD_RESPONSE_1],  # Failure.
        [BAD_RESPONSE_1, _response(ROBUSTFILL_3)],  # Success.
        [BAD_RESPONSE_1, _response(ROBUSTFILL_3)],  # Failure.
        [_response(ROBUSTFILL_2), BAD_RESPONSE_2],  # Failure.
        [_response(ROBUSTFILL_1), BAD_RESPONSE_1],  # Success.
        [_response(ROBUSTFILL_2), BAD_RESPONSE_2],  # Success.
    ]
    mock_query_llm.side_effect = model_samples
    # Which tasks we expect to solve.
    expected_successes = {
        'deepcoder': ['COMPOSE_DIFFERENT_CONCEPTS',
                      'SWITCH_CONCEPT_ORDER',
                      'ADD_OP_FUNCTIONALITY'],
        'robustfill': ['LENGTH_GENERALIZATION',
                       'COMPOSE_NEW_OP',
                       'ADD_OP_FUNCTIONALITY'],
    }

    # Run the experiment.
    with flagsaver.flagsaver(model=model, num_samples=2):
      all_results = run_llm_experiment.run_entire_experiment()

    # Check that we solved exactly the tasks we expected to solve.
    for dataset_type in all_results:
      for generalization_task in all_results[dataset_type]:
        should_succeed = generalization_task in expected_successes[dataset_type]
        self.assertEqual(
            all_results[dataset_type][generalization_task][0]['success'],
            should_succeed)

    # Check that query_llm was called.
    self.assertEqual(mock_query_llm.call_count, 12)

    # Check the caching by redoing the experiment but with only bad samples.
    mock_query_llm.reset_mock()
    mock_query_llm.return_value = [BAD_RESPONSE_1, BAD_RESPONSE_2]
    with flagsaver.flagsaver(model=model, num_samples=2):
      all_results = run_llm_experiment.run_entire_experiment()
    for dataset_type in all_results:
      for generalization_task in all_results[dataset_type]:
        should_succeed = generalization_task in expected_successes[dataset_type]
        self.assertEqual(
            all_results[dataset_type][generalization_task][0]['success'],
            should_succeed)
    self.assertEqual(mock_query_llm.call_count, 0)

    run_llm_experiment.RESULTS_FORMAT = saved_results_format


if __name__ == '__main__':
  absltest.main()
