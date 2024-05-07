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

num_examples=3
deepcoder_max_list_length=5
deepcoder_max_int=50
max_program_arity=2
max_num_statements=5

embedding_dim=512
hidden_dim=1024

# Reimplement the length and distance computation from launch_train.py.
# It is important that these distances are exactly as used in training.
object_token_length=$((deepcoder_max_list_length + 5))
max_input_objects=$((max_program_arity + max_num_statements - 1))
max_input_length=$((max_input_objects * object_token_length))
max_output_prediction_length=$((num_examples * object_token_length))
max_program_part_length=6
spec_decomposer_max_distance=$((max_input_length > max_output_prediction_length ? max_input_length : max_output_prediction_length))
synthesizer_max_distance=$((max_input_length > max_program_part_length ? max_input_length : max_program_part_length))
spec_decomposer_max_program_cross_embed_distance=$((max_input_length * num_examples > max_output_prediction_length ? max_input_length * num_examples : max_output_prediction_length))
synthesizer_max_program_cross_embed_distance=$((max_input_length * num_examples > max_program_part_length ? max_input_length * num_examples : max_program_length))

echo "spec_decomposer_max_distance=${spec_decomposer_max_distance}"
echo "synthesizer_max_distance=${synthesizer_max_distance}"
echo "spec_decomposer_max_program_cross_embed_distance=${spec_decomposer_max_program_cross_embed_distance}"
echo "synthesizer_max_program_cross_embed_distance=${synthesizer_max_program_cross_embed_distance}"

# Compute lengths. These don't have to be exact, only long enough.
max_num_variables=10
max_io_length=$((max_num_variables * object_token_length))
max_num_program_parts=7
max_program_length=$((max_program_part_length * max_num_program_parts))
max_spec_part_length=30

# To use test data in the GCS bucket:
base_data_dir=gs://exedec/test_data_tf_records/deepcoder
# To use test data generated locally:
# base_data_dir=~/exedec_data/deepcoder_data

# To use trained models in the GCS bucket:
base_model_dir=gs://exedec/trained_models/deepcoder
# To use models trained locally:
# train_run=1
# base_model_dir=~/exedec_results/exedec_train_deepcoder_run-${train_run}

num_test=1000
eval_run=e2e_predict_1
save_dir=~/exedec_results/evaluation/deepcoder_${eval_run}

test_dataset_format=${base_data_dir}/{experiment}_data/entire_programs_test.tf_records*
spec_decomposer_path_format=${base_model_dir}/spec_decomposer_model/checkpoints/adr=0.1,ara={aligned_relative_attention},dr=0.1,e={experiment},ed=${embedding_dim},hd=${hidden_dim},l=0.0002,md=60,mpced=180,npb=32,s={seed},scnpr=0.0,ura=True/
synthesizer_path_format=${base_model_dir}/synthesizer_model/checkpoints/adr=0.1,ara=False,dr=0.1,e={experiment},ed=${embedding_dim},hd=${hidden_dim},l=0.0002,md=60,mpced=180,npb=32,s={seed},scnpr={corruption_rate},ura=True/
joint_path_format=${base_model_dir}/joint_model/checkpoints/adr=0.1,ara=False,dr=0.1,e={experiment},ed=${embedding_dim},hd=${hidden_dim},l=0.0002,md=60,mpced=180,npb=32,s={seed},scnpr=0.0,ura=True/

# Generate comma-separated strings to pass as an argument.
experiments=$(printf ",%s" "${experiments_array[@]}")
experiments=${experiments:1}

for prediction_type in separate joint; do

  python -m spec_decomposition.launch_end_to_end_predict \
  --exp_title=end_to_end_predict-deepcoder-run-${eval_run}-${prediction_type} \
  --save_dir=${save_dir} \
  --dataset_type=deepcoder \
  --experiments=${experiments} \
  --deepcoder_max_list_length=${deepcoder_max_list_length} \
  --deepcoder_max_int=${deepcoder_max_int} \
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
