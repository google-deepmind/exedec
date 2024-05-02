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


declare -a experiments_array=(
    "NONE"
    "LENGTH_GENERALIZATION"
    "COMPOSE_DIFFERENT_CONCEPTS"
    "SWITCH_CONCEPT_ORDER"
    "COMPOSE_NEW_OP"
    "ADD_OP_FUNCTIONALITY"
)

num_examples=4

embedding_dim=512
hidden_dim=1024

base_data_dir=gs://exedec/test_data_tf_records/robustfill
train_run=neurips23final
base_model_dir=gs://exedec/trained_models/robustfill
num_test=1000
eval_run=e2e_predict
save_dir=~/exedec_results/evaluation/robustfill_${eval_run}

test_dataset_format=${base_data_dir}/{experiment}_data/entire_programs_test.tf_records*
spec_decomposer_path_format=${base_model_dir}/spec_decomposer_model/checkpoints/adr=0.1,ara={aligned_relative_attention},dr=0.1,e={experiment},ed=${embedding_dim},hd=${hidden_dim},l=0.0002,md=200,mpced=800,npb=32,s={seed},scnpr=0.0,ura=True/
synthesizer_path_format=${base_model_dir}/synthesizer_model/checkpoints/adr=0.1,ara=False,dr=0.1,e={experiment},ed=${embedding_dim},hd=${hidden_dim},l=0.0002,md=20,mpced=80,npb=32,s={seed},scnpr={corruption_rate},ura=True/
joint_path_format=${base_model_dir}/joint_model/checkpoints/adr=0.1,ara=False,dr=0.1,e={experiment},ed=${embedding_dim},hd=${hidden_dim},l=0.0002,md=200,mpced=800,npb=32,s={seed},scnpr=0.0,ura=True/

# Generate comma-separated strings to pass as an argument.
experiments=$(printf ",%s" "${experiments_array[@]}")
experiments=${experiments:1}

for prediction_type in separate joint; do

  if [[ "${prediction_type}" == "separate" ]]; then
    max_io_length=200
    max_program_length=100
    max_spec_part_length=85
    spec_decomposer_max_distance=200
    synthesizer_max_distance=20
    spec_decomposer_max_program_cross_embed_distance=800
    synthesizer_max_program_cross_embed_distance=80
  elif [[ "${prediction_type}" == "joint" ]]; then
    max_io_length=200
    max_program_length=100
    max_spec_part_length=-1  # Unused.
    spec_decomposer_max_distance=-1  # Unused.
    synthesizer_max_distance=200
    spec_decomposer_max_program_cross_embed_distance=-1  # Unused.
    synthesizer_max_program_cross_embed_distance=800
  else
    echo "Unhandled model ${prediction_type}"
    exit 1
  fi

  xmanager launch end_to_end_predict_xm_run.py -- \
  --exp_title=end_to_end_predict-robustfill-run-${eval_run}-${prediction_type} \
  --save_dir=${save_dir} \
  --dataset_type=robustfill \
  --experiments=${experiments} \
  --test_dataset_format=${test_dataset_format} \
  --num_test_batches=${num_test} \
  --num_examples=${num_examples} \
  --max_io_length=${max_io_length} \
  --max_program_length=${max_program_length} \
  --max_spec_part_length=${max_spec_part_length} \
  --spec_decomposer_path_format=${spec_decomposer_path_format} \
  --synthesizer_path_format=${synthesizer_path_format} \
  --joint_path_format=${joint_path_format} \
  --embedding_dim=${embedding_dim} \
  --hidden_dim=${hidden_dim} \
  --spec_decomposer_num_position_buckets=32 \
  --synthesizer_num_position_buckets=32 \
  --spec_decomposer_max_distance=${spec_decomposer_max_distance} \
  --synthesizer_max_distance=${synthesizer_max_distance} \
  --spec_decomposer_max_program_cross_embed_distance=${spec_decomposer_max_program_cross_embed_distance}  \
  --synthesizer_max_program_cross_embed_distance=${synthesizer_max_program_cross_embed_distance} \
  --use_relative_attention=True \
  --beam_size=10 \
  --prediction_type=${prediction_type} \
  --detect_invalid=true \
  --use_execution=true \
  --discard_repeat_functionality=true \
  --aligned_relative_attention=true \
  --corruption_rate=0.0 \
  --seed=10 \
  --seed=20 \
  --seed=30 \
  --seed=40 \
  --seed=50 \

done
