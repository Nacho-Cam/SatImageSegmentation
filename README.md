# SatImageSegmentation

This project provides a modular pipeline for satellite image segmentation, focusing on water body detection using the Segment Anything Model (SAM) and supporting grouped (multi-pipeline) preprocessing. The architecture is designed for extensibility, supporting additional segmentation tasks (e.g., vegetation) and large TIFF images.

## Project Architecture

### 1. Data
- **data/raw/**: Contains all raw datasets.
  - **water_bodies/Water Bodies Dataset/**: Main water segmentation dataset (Images & Masks).
  - **NEW2-AerialImageDataset/AerialImageDataset/**: Additional aerial imagery for testing and validation.
  - **large_water_images/**: For large TIFF images.
- **data/processed/**: For processed/augmented data (user-generated).
- **data/external/**: For external datasets.

### 2. Models
- **models/water/**: All water segmentation code, checkpoints, and outputs.
  - **checkpoints/SAM-water-hf/**: HuggingFace-style checkpoints for the fine-tuned SAM model (with subfolders for each checkpoint).
  - **logs/**: Training logs.
  - **outputs/**: Inference outputs and visualizations.
  - **finetune_sam.py**: Fine-tuning script for SAM (supports grouped 15-channel input, checkpoint resume, and robust training).
  - **main_infer_tiff.py**: Inference and visualization script (runs on 10 random images from each dataset, shows bounding boxes, supports grouped preprocessing and checkpoint resume).
  - **batch_predict_preprocessed.py**: Batch inference on preprocessed images.
  - **preprocess_and_infer_single.py**: Preprocessing and inference for a single image.
  - **test_sam_water_metrics.py**: Evaluation script for metrics (IoU, Dice, etc.).

- **models/vegetation/**: Placeholder for vegetation segmentation (mirrors water structure).

### 3. Source Code
- **src/common/**: Shared utilities, config, and dataset code.
- **src/water/**: Water-specific model, training, and evaluation code.
- **src/vegetation/**: Vegetation-specific code.

### 4. Notebooks
- **notebooks/experiments.ipynb**: Experiments and demonstrations (Keras/Kaggle SAM, data loading, fine-tuning, etc.).
- **notebooks/data_exploration.ipynb**: Data exploration and visualization.

### 5. Utilities
- **convert_tiff_to_jpg_and_preproc.py**: Preprocessing utilities for TIFF/JPG conversion and grouped pipelines (imported dynamically by scripts).

## Key Features
- **Grouped Preprocessing**: 15-channel input combining multiple preprocessing pipelines (original, CLAHE, grayscale, HSV, normalized, etc.).
- **Flexible Checkpointing**: Fine-tuning supports resuming from latest or user-specified checkpoint. (Full optimizer/epoch resume in progress.)
- **Inference & Visualization**: Inference script displays bounding boxes around detected water regions, merging close regions and using thick outlines.
- **Extensible**: Easily add new segmentation tasks (e.g., vegetation) by mirroring the water model structure.

## Getting Started
1. Place your data in the `data/raw/` directory as described above.
2. Install dependencies from `requirements.txt` (see below).
3. Use `models/water/finetune_sam.py` to fine-tune the SAM model on water segmentation.
4. Use `main_infer_tiff.py` for inference and visualization on random samples.
5. See notebooks for experiments and data exploration.

## Requirements
See `requirements.txt` for dependencies. Main libraries include:
- torch
- transformers
- numpy
- opencv-python
- pillow
- matplotlib
- tqdm
- scikit-learn
- scikit-image
- tifffile
- kagglehub (for Keras/Kaggle experiments)

## Notes
- Model checkpoints are saved in HuggingFace format for compatibility.
- Preprocessing pipelines are modular and can be extended.
- For large TIFF images, ensure `imagecodecs` is installed for LZW support.
- For full training state resumption (optimizer, epoch, batch), see comments in `finetune_sam.py`.
