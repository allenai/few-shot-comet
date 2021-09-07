## Section 4: Learning Few-Shot Commonsense Knowledge Models

This repo contains code for training few-shot COMETs.

## Setup
For setup, install `requirements.txt` in the `infra/` directory.

T5-11B (the backbone model) requires running on TPU. For instructions on how to run on TPU, please visit [Google's offical TPU blog](https://cloud.google.com/tpu/docs) or the [TPU start guide](https://cloud.google.com/tpu/docs/beginners-guide). We provide model files hosted on Google Cloud Platform: `https://console.cloud.google.com/storage/browser/ai2-mosaic-public/projects/few-shot-comet`

## `infra/`
We provide code for running T5-11B on TPU. Please note that this code must be run on a Google Cloud VM -- it does not work locally on GPU or without a TPU.

## `atomic2020_splits/`
We also provide training data for retraining T5-11B models. Each training data split is stored in `atomic2020_splits/`, in the paper, we average across 5 splits, and we provide each of the 5 corrasponding splits in the data repo here.