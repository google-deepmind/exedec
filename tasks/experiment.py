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

"""The Experiment enum describing different datasets."""

import enum


@enum.unique
class Experiment(enum.Enum):
  NONE = 0
  LENGTH_GENERALIZATION = 1
  COMPOSE_DIFFERENT_CONCEPTS = 2
  SWITCH_CONCEPT_ORDER = 3
  COMPOSE_NEW_OP = 4
  ADD_OP_FUNCTIONALITY = 5
