# SatImageSegmentation

This project contains infrastructure for training multiple image segmentation models (e.g., water, vegetation) using SAM on large TIFF images.

## Structure
- `data/`: Raw, processed, and external data
- `models/`: Model checkpoints and logs for each segmentation type
- `src/`: Source code, organized by model and shared utilities
- `notebooks/`: Jupyter notebooks for exploration and experiments

## Getting Started
1. Place your data in the `data/raw/` directory.
2. Install dependencies from `requirements.txt`.
3. Use the scripts in `src/` to train and evaluate models.

## Requirements
See `requirements.txt` for dependencies.
