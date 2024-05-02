#!/bin/bash
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


# Choose whichever generalization tasks and splits you want to generate.
declare -a experiments_array=(
  "NONE"
  "LENGTH_GENERALIZATION"
  "COMPOSE_DIFFERENT_CONCEPTS"
  "SWITCH_CONCEPT_ORDER"
  "COMPOSE_NEW_OP"
  "ADD_OP_FUNCTIONALITY"
)
declare -a splits_array=(
  "train"
  "valid"
  "test"
)

# Change these options as desired.
seed=0  # Base seed that affects each worker differently.
max_program_arity=2
num_examples=3
deepcoder_max_list_length=5
deepcoder_max_int=50

dataset_name=deepcoder_data
base_save_dir=~/exedec_data/

# Whether to generate a full dataset or just a small one for testing purposes.
GENERATE_FULL_DATA=false
if ${GENERATE_FULL_DATA}; then
  echo 'Generating full dataset'
  # We train for 500K steps on 8 devices with a per-device batch size of 16, or
  # 64M examples total. These settings will generate that many programs. (Also
  # note that the decomposed data will have multiple data elements per program,
  # but we still need this many programs for training a no-decomposition
  # baseline, if we control for the number of training steps.)
  num_shards_train=1000
  num_programs_per_search_train=1000
  num_searches_train=64

  # 10K test examples, although the experiments only use the first 1000 of them.
  # If you are generating completely new test datasets, it may be better to set
  # num_programs_per_search_test=1 to get more variety in program inputs (which
  # will be the same for all programs obtained from one search).
  num_shards_test=100
  num_programs_per_search_test=10
  num_searches_test=10
else
  echo 'Generating small dataset'
  num_shards_train=2
  num_programs_per_search_train=100
  num_searches_train=1

  num_shards_test=2
  num_programs_per_search_test=10
  num_searches_test=1
fi

# Generate comma-separated strings to pass as an argument.
experiments=$(printf ",%s" "${experiments_array[@]}")
experiments=${experiments:1}
splits=$(printf ",%s" "${splits_array[@]}")
splits=${splits:1}

# Launch the experiment.
xmanager launch tasks/deepcoder/dataset/xm_run.py -- \
  --exp_title=generate_${dataset_name} \
  --seed=${seed} \
  --save_dir=${base_save_dir}/${dataset_name} \
  --experiments=${experiments} \
  --splits=${splits} \
  --num_shards_train=${num_shards_train} \
  --num_programs_per_search_train=${num_programs_per_search_train} \
  --num_searches_train=${num_searches_train} \
  --num_shards_test=${num_shards_test} \
  --num_programs_per_search_test=${num_programs_per_search_test} \
  --num_searches_test=${num_searches_test} \
  --max_program_arity=${max_program_arity} \
  --num_examples=${num_examples} \
  --deepcoder_max_list_length=${deepcoder_max_list_length} \
  --deepcoder_max_int=${deepcoder_max_int} \
