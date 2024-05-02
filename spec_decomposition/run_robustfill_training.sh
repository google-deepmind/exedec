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


# Which generalization tasks to train on.
declare -a experiments_array=(
  "NONE"
  "LENGTH_GENERALIZATION"
  "COMPOSE_DIFFERENT_CONCEPTS"
  "SWITCH_CONCEPT_ORDER"
  "COMPOSE_NEW_OP"
  "ADD_OP_FUNCTIONALITY"
)

# Which individual model types to train.
declare -a models_array=(
  "spec_decomposer_model"
  "synthesizer_model"
  "joint_model"
  "baseline_model"
)

# Which dataset to train on.
examples=4  # Number of I/O examples in specifications.
data_dir=~/exedec_data/robustfill_data

# This training run.
run=1
base_save_dir=~/exedec_results
title_without_model_type=exedec_train_robustfill_run-${run}
save_dir=${base_save_dir}/${title_without_model_type}

# Generate comma-separated strings to pass as an argument.
experiments=$(printf ",%s" "${experiments_array[@]}")
experiments=${experiments:1}

# Each model type will be a separate XM experiment, with a separate TensorBoard.
# It wouldn't make much sense to compare models trained on different prediction
# tasks.
for model_type in "${models_array[@]}"; do

  if [[ "${model_type}" == "spec_decomposer_model" ]]; then
    max_input_length=200
    max_target_length=85
    max_distance=200
    max_program_cross_embed_distance=800
  elif [[ "${model_type}" == "synthesizer_model" ]]; then
    max_input_length=20
    max_target_length=15
    max_distance=20
    max_program_cross_embed_distance=80
  elif [[ "${model_type}" == "joint_model" ]]; then
    max_input_length=200
    max_target_length=15
    max_distance=200
    max_program_cross_embed_distance=800
  elif [[ "${model_type}" == "baseline_model" ]]; then
    max_input_length=200
    # A full program has <= 10 parts, each part has <= 12 tokens plus separator.
    # Also make room for BOS and EOS.
    max_target_length=132
    max_distance=200
    max_program_cross_embed_distance=800
  else
    echo "Unhandled model ${model_type}"
    exit 1
  fi

  xmanager launch train_xm_run.py -- \
    --exp_title=${title_without_model_type}_${model_type} \
    --save_dir=${save_dir} \
    --dataset_type=robustfill \
    --experiments=${experiments} \
    --dataset_dir=${data_dir} \
    --num_examples=${examples} \
    --max_input_length=${max_input_length} \
    --predict_max_input_length=${max_input_length} \
    --max_target_length=${max_target_length} \
    --max_distance=${max_distance} \
    --max_program_cross_embed_distance=${max_program_cross_embed_distance} \
    --num_train_steps=500000 \
    --num_eval_steps=10 \
    --model_type=${model_type} \
    --per_device_batch_size=16 \
    --lr=2e-4 \
    --embedding_dim=512 \
    --hidden_dim=1024 \
    --dropout_rate=0.1 \
    --attention_dropout_rate=0.1 \
    --num_position_buckets=32 \
    --aligned_relative_attention=1 \
    --synthesizer_corrupted_next_part_rate=0.0 \
    --seed=10 \
    --seed=20 \
    --seed=30 \
    --seed=40 \
    --seed=50 \
    --eval_freq=10000 \
    --predict_freq=50000 \
    --checkpoint_freq=50000 \

    # Add these for the prediction-only job.
    # --exp_title=${title_without_model_type}_${model_type}_predict-only \
    # --predict_only=True \
    # --predict_freq=1 \

    # Use these for quick iteration.
    # --log_freq=500 \
    # --eval_freq=1000 \
    # --predict_freq=1000 \

done
