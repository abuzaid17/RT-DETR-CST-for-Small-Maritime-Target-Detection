# RT-DETR-CST for Small Maritime Target Detection 

This repository contains the source code, dataset manifest, environment specification, and training artifacts supporting the paper *"[Paper Title Here]"*. It provides the materials required for independent verification of the results reported in the paper, including the architecture implementation, the exact per-epoch training record, and the dataset partitioning used for training and evaluation.

## Repository Structure

```
.
├── environment/
│   ├── environment.lock          # Full, exact environment snapshot (pip freeze)
│   └── requirements-core.txt     # Curated list of core dependencies only
├── model_source/
│   ├── IR_TRAINNING.ipynb        # Full training notebook (architecture assembly + training loop)
│   └── rt_detr_cst_modules.py    # TCN, SWN, and CFAN module definitions
├── splits/
│   ├── train_split.txt           # Filenames used for training
│   ├── val_split.txt             # Filenames used for validation
│   └── test_split.txt            # Filenames used for testing
├── checkpoints.txt               # Link to trained model weights (see Checkpoints section below)
├── dataset_manifest.csv          # Per-image SHA-256 hash, object count, and split assignment
├── training_log.csv              # Raw per-epoch training metrics (loss, precision, recall, mAP, fitness)
└── README.md
```

## Requirements and Environment

Two environment files are provided:

- **`environment/environment.lock`** — the complete, exact snapshot of the training environment (all installed packages and versions). Use this file for full reproducibility.
- **`environment/requirements-core.txt`** — a shorter list of only the core dependencies (PyTorch, Ultralytics, timm, OpenCV, etc.) for quick reference.

To recreate the full environment:

```bash
pip install -r environment/environment.lock
```

To install only the core dependencies:

```bash
pip install -r environment/requirements-core.txt
```

The detection framework used is Ultralytics `8.4.146`. The exact PyTorch, CUDA, and cuDNN versions used during training are recorded in `environment.lock`.

## Dataset

Training and evaluation were performed on the ISDD infrared ship-detection dataset (see the paper's dataset section and primary source citation). The dataset itself is not redistributed in this repository; `dataset_manifest.csv` and the files in `splits/` document exactly which images were used and how they were partitioned, so that the original dataset can be matched against this record.

### `dataset_manifest.csv`

Each row corresponds to one image and contains:

| Column | Description |
|---|---|
| `filename` | Image filename |
| `split` | `train`, `val`, or `test` |
| `sha256` | SHA-256 hash of the image file, for integrity verification |
| `num_objects` | Number of labeled objects in the corresponding YOLO-format label file |

### `splits/`

`train_split.txt`, `val_split.txt`, and `test_split.txt` list the exact filenames assigned to each partition, generated directly from the dataset directory structure used during training (`images/train`, `images/val`, `images/test`).

## Model Source

- **`rt_detr_cst_modules.py`** defines the three custom modules introduced in this work: the Temporal Convolutional Network (TCN) branch, the SynapticWeave Network (SWN) block, and the Channel-and-Feature Attention Network (CFAN) module.
- **`IR_TRAINNING.ipynb`** contains the complete model assembly (how the custom modules are integrated into the RT-DETR backbone/neck/head) and the training procedure used to produce the results reported in the paper.

## Training Log

`training_log.csv` contains the raw, per-epoch record (loss, precision, recall, mAP, and fitness score) produced during training. Every value reported in the paper's results table is directly extracted from this file, including the checkpoint-selection comparison between epoch 27 and epoch 29 discussed in the paper.

## Checkpoints

Trained model weights (`.pt` files) are hosted externally due to file size. `checkpoints.txt` contains the download link to the checkpoint files, including the epoch 27 and epoch 29 checkpoints referenced in the paper's checkpoint-selection analysis.

## Reproducing the Results

1. Install dependencies from `environment/environment.lock`.
2. Download the dataset and arrange it according to the structure referenced in `dataset_manifest.csv` and `splits/`.
3. Download the checkpoints listed in `checkpoints.txt` if only evaluating a pretrained model, or run the training procedure in `model_source/IR_TRAINNING.ipynb` to retrain from scratch.
4. Compare output metrics against `training_log.csv`.

## Citation

If you use this code or the accompanying paper, please cite:

```
[Citation to be added upon publication]
```

## License

[License to be added]

## Contact

For questions regarding this repository or the paper, please open an issue in this repository.
