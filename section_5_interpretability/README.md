## Section 5: Understanding Few-Shot Knowledge Models

This repo contains code for creating the heatmaps in the paper.

## Setup

Setup your env by running `requirements.txt` (ideally in a `conda` enviroment on python3)

## `heatmaps/`

To generate heatmaps, first download the model file and comparasion model file (likely T5-11B without additional fine-tuning.). We provide a script for downloading the model in `heatmaps/download.py`. The model files are located here: `https://console.cloud.google.com/storage/browser/ai2-mosaic-public/projects/few-shot-comet`. Then, calculate the weight values via `calculate_differences.py`. The heatmap itself is produced with `heatmaps.py`.

### Credit
`calculate_differences.py` is adapted from `https://github.com/VITA-Group/BERT-Tickets`.