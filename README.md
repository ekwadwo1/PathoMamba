# PathoMamba

Stiffness-modulated deformable registration for longitudinal glioma MRI,
built on the BraTS-Reg benchmark. PathoMamba conditions the registration
dynamics on a tumor signed-distance prior, aiming for plastic deformation
inside pathology while preserving topology in healthy parenchyma.

## Overview

PathoMamba parameterizes the deformation via a Stationary Velocity Field
integrated by scaling-and-squaring, and modulates feature dynamics using a
signed-distance-function (SDF) prior derived from the tumor segmentation. 
We optimize the network using \textbf{AdamW} ($\text{lr} = 2e^{-4}$) with a 
cosine annealing schedule for 300 epochs.
To manage the memory footprint of 3D state-space modeling, we utilize the gradient 
Checkpointing and Automatic Mixed Precision (AMP).

## Repository structure

    PathoMamba/
    ├── README.md
    ├── requirements.txt
    ├── environment.yml
    ├── .gitignore
    ├── configs/
    │   └── default.yaml
    ├── pathomamba/
    │   ├── preprocessing.py
    │   ├── dataset.py
    │   ├── model.py
    │   ├── losses.py
    │   ├── trainer.py
    │   ├── metrics.py
    │   └── utils.py
    └── scripts/
        ├── 01_preprocess.py
        ├── 02_train.py
        ├── 03_evaluate.py
        └── 04_diagnose.py

## Requirements

- Google Colab, NVIDIA A100 (40GB)} GPU using PyTorch and MONAI.
- PyTorch with CUDA
- nibabel, scipy, numpy, pandas, tqdm, matplotlib

See `requirements.txt` or `environment.yml`.

## Usage

    # 1. Preprocess BraTS-Reg volumes into .pt tensors
    python scripts/01_preprocess.py --config configs/default.yaml

    # 2. Train
    python scripts/02_train.py --config configs/default.yaml

Paths (dataset, processed-data, checkpoint directories) are set in
`configs/default.yaml`.

## Data

This project uses the BraTS-Reg challenge dataset, which must be obtained
separately under its own license. No patient data is included in this repo.
