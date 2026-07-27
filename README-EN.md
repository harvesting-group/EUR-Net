# ExposureMoE

ExposureMoE is a strawberry instance segmentation and ripeness estimation project built on a customized local branch of Ultralytics 8.3.0. It is designed to improve ripeness estimation robustness under mixed and uneven illumination using an illumination-aware Mixture-of-Experts module.

## Overview

The project extends Ultralytics with:

- A custom `ripe` task for instance segmentation and ripeness regression
- `LightMoEStem` for illumination-aware feature modeling
- Training, validation, prediction, error analysis, and visualization scripts

Main project files:

```text
ExposureMoE/
├── ultralytics/                         # Customized Ultralytics source
├── ultralytics/cfg/models/11/          # YOLO11 ripe / LightMoE configs
├── ultralytics/datasets/               # Dataset configs
├── train_exposuremoe.py                # Training
├── val_exposuremoe.py                  # Validation
├── predict_exposuremoe.py                      # Prediction
└── fps_exposuremoe.py                  # Inference speed evaluation
```

## Model

Main model configurations:

```text
ultralytics/cfg/models/11/yolo11-ripe-lightmoe.yaml
```

`yolo11-ripe-lightmoe.yaml` adds `LightMoEStem` before the backbone to model illumination-related features. The custom `Ripe` head predicts:

- Strawberry instances
- Instance masks
- Ripeness values
- uncertainty / variance branches

## Dataset

The default dataset configuration is:

Current class definition:

```text
0: strawberry
```

See `ultralytics/datasets/MixExposure/README-EN.md` for additional dataset information.

## Installation

Create an isolated Python environment:

```bash
conda create -n exposuremoe python=3.10 -y
conda activate exposuremoe
```

Install PyTorch according to your CUDA version. For example, CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the main dependencies:

```bash
pip install numpy opencv-python pillow matplotlib pandas openpyxl pyyaml scipy tqdm seaborn psutil
```

## Usage

### Training

```bash
python train_exposuremoe.py
```

Typical LightMoE auxiliary-loss settings include:

```text
light_corr=0.2
light_route=1.0
light_identity=0.5
light_smooth=0.3
light_balance=0.3
light_diverse=0.3
```

> Before training, verify that the model path in `train_exposuremoe.py` matches the actual YAML filename in `ultralytics/cfg/models/11/`.

### Validation

```bash
python val_exposuremoe.py
```

The default checkpoint is:

```text
runs/ripe/train_exposuremoe/weights/best.pt
```

### Prediction

```bash
python predict_moe.py
```

The default prediction source is:

```text
ultralytics/datasets/MixExposure/images/test
```
