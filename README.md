# MIF - Medical Image Fusion

This repository contains implementations and experiments for medical image fusion methods.

## Overview

Medical image fusion combines information from multiple imaging modalities to create more informative images for diagnosis and treatment.

## Supported Modalities

- CT-MRI
- PET-MRI
- SPECT-MRI

## Dataset Structure

The dataset is organized in `DatasetBMP_update/` directory with the following structure:

```
DatasetBMP_update/
├── CT-MRI/
│   ├── train/
│   │   ├── CT/          # 184 training CT images
│   │   └── MRI/         # 184 training MRI images
│   └── test/
│       ├── CT/          # 24 test CT images
│       └── MRI/         # 24 test MRI images
├── PET-MRI/
│   ├── train/
│   │   ├── PET/         # 251 training PET images
│   │   └── MRI/         # 251 training MRI images
│   └── test/
│       ├── PET/         # Test PET images
│       └── MRI/         # Test MRI images
├── SPECT-MRI/
│   ├── train/
│   │   ├── SPECT/       # 370 training SPECT images
│   │   └── MRI/         # 370 training MRI images
│   └── test/
│       ├── SPECT/       # Test SPECT images
│       └── MRI/         # Test MRI images
└── convert_gray.py      # Utility script for image conversion
```

**Image Format:** All images are in `.bmp` format (24-bit bitmap)

**Paired Images:** Each modality pair (e.g., CT and MRI) contains the same number of corresponding images with matching filenames for proper alignment during fusion.

## Project Structure

```
MIF/
├── datasets/          # Dataset directory (ignored by git)
├── logs/             # Training and experiment logs (ignored by git)
├── result/           # Fusion results (ignored by git)
├── .gitignore        # Git ignore configuration
└── README.md         # This file
```

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- CUDA (for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone git@github.com:longcris04/MIF.git
cd MIF
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

Coming soon...

## License

TBD

## Contact

For questions and feedback, please open an issue on GitHub.
