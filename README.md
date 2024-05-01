# ExeDec: Execution Decomposition for Compositional Generalization in Neural Program Synthesis

This repository provides code and datasets associated with our research paper
published at ICLR'24 ([OpenReview](https://openreview.net/forum?id=oTRwljRgiv)).

In this paper, we describe different forms of compositional generalization that
are desirable in program synthesis and introduce datasets for measuring them. We
also present ExeDec, a decomposition-based approach to program synthesis
achieving higher compositional generalization on two domains compared to prior
approaches, when applied to LLMs and small Transformers trained from scratch.

## Installation

No installation is needed to use our test datasets which are provided as
`*.jsonl` files.

To generate training data or more test data, you will need to install
[XManager](https://github.com/google-deepmind/xmanager),
[NumPy](https://numpy.org/install/),
[TensorFlow](https://www.tensorflow.org/install), and
[Abseil Python](https://abseil.io/docs/python/quickstart):

```
pip install xmanager numpy tensorflow absl-py
```

To run our trained model checkpoints or train new models, you will additionally
need [JAX](https://jax.readthedocs.io/en/latest/installation.html) and
[Flax](https://flax.readthedocs.io/en/latest/#installation):

```
pip install "jax[cpu]" flax
```

Note, you may need to alter your JAX installation depending on your desired
platform.


TODO: check all of the above

## Datasets

The ICLR'24 paper includes experiments in two domains, DeepCoder and RobustFill.
Within each domain, there are 6 different train/test splits, including 5 forms
of compositional generalization relevant to programming
(`LENGTH_GENERALIZATION`, `COMPOSE_DIFFERENT_CONCEPTS`, `SWITCH_CONCEPT_ORDER`,
`COMPOSE_NEW_OP`, and `ADD_OP_FUNCTIONALITY`), and a scenario where no
generalization is required (`NONE`).

For convenience and reproducibility, we provide the **test datasets** used in
the ICLR'24 experiments:

* `data/test_data/` contains test problems for the experiments on small
  Transformers trained from scratch, in JSON Lines format. This same data is
  also provided in TFRecords format in a Google Cloud Storage
  [bucket](https://console.developers.google.com/storage/browser/exedec),
  because the TFRecords format is more convenient for use with our scripts for
  evaluating these small Transformers.

* `data/llm_data/` contains test problems and few-shot examples for the LLM
  experiments, also in JSON Lines format.

`data/data_utils.py` contains details about the data contents and provides
helpful utility functions for parsing and evaluating programs in the dataset.

#### Generating new data

We additionally provide scripts to generate new train and test data (in
TFRecords format) for the DeepCoder and RobustFill DSLs, according to the
different compositional generalization splits:

```
bash tasks/deepcoder/dataset/run_data_generation.sh
bash tasks/robust_fill/dataset/run_data_generation.sh
```

By default the scripts generate small datasets locally just for testing
purposes, but you can edit the scripts to generate larger datasets using a cloud
computing platform.

These scripts use [XManager](https://github.com/google-deepmind/xmanager) to
manage parallelization across CPU workers. You may instead use a different
mechanism to coordinate workers so that each CPU worker directly calls
`tasks/deepcoder/dataset/write_data.py` or
`tasks/robust_fill/dataset/write_data.py` with appropriate command-line flags.

## Trained model checkpoints

We provide checkpoints for the Transformer models trained from scratch in a
Google Cloud Storage bucket,
[gs://exedec](https://console.developers.google.com/storage/browser/exedec). We
include checkpoints for the ExeDec, Ablation, and Transformer Baseline
approaches as described in the paper, trained on each dataset with 5 random
initializations.

In the `spec_decomposition/` directory, `run_deepcoder_end_to_end_predict.sh`
and `run_robustfill_end_to_end_predict.sh` demonstrate how to evaluate the
checkpoints on our datasets. However, you will likely need to edit the scripts
to use ML accelerators with your preferred cloud computing platform. We mainly
provide the scripts to demonstrate how `end_to_end_predict.py` should be invoked
with command-line flags.

## Training new models

TODO

## Other notes

It may help to know the following, when navigating and interpreting the code:

* The paper's "Execution Decomposition" technique is instead called
  "spec_decomposition" in the code.
* The paper's "SubgoalModel" is instead called "SpecDecomposerModel" in the
  code.
* The paper's "CombinedModel" is instead called "JointModel" in the code.

## Citing this work

If you use our datasets or code, please cite our ICLR'24 paper:

```latex
@inproceedings{shi2024exedec,
      title={{ExeDec}: Execution Decomposition for Compositional Generalization in Neural Program Synthesis},
      author={Kensen Shi and Joey Hong and Yinlin Deng and Pengcheng Yin and Manzil Zaheer and Charles Sutton},
      booktitle={The Twelfth International Conference on Learning Representations},
      year={2024},
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
