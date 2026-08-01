<div align="center">

# REMIND: Retrieval-Augmented Reconstruction With Dual Memories for Modality-Missing Object Re-Identification

### IEEE Transactions on Image Processing (TIP), 2026

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Xplore-blue)](https://doi.org/10.1109/TIP.2026.3715092)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTIP.2026.3715092-red)](https://doi.org/10.1109/TIP.2026.3715092)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/skye-1201/REMIND)

**Zhendong Xu, Zi Wang, Aihua Zheng, Chenglong Li, Jin Tang**

</div>

## News

- **July 2026:** REMIND is published in *IEEE Transactions on Image Processing*, Vol. 35, pp. 8107–8120.
- **July 2026:** The official implementation is publicly available.

## Overview

REMIND addresses modality-missing object re-identification by recovering discriminative information from absent modalities. Existing reconstruction-based methods often overlook modality-specific cues and rely on a single reconstruction path, resulting in incomplete representations and limited cross-modal modeling capability.

REMIND contains two key components. The **Dual Memory Construction (DMC)** module disentangles modality-specific and modality-common representations and stores them in dedicated memory banks as structured semantic priors. The **Retrieval-Augmented Missing Reconstruction (RMR)** module first uses modality-common features to retrieve identity-consistent memory entries and then performs perturbation-aware multi-path reconstruction with adaptive fusion. This design improves both the semantic completeness and robustness of reconstructed features.

Experiments are conducted on four multi-modal object Re-ID benchmarks: **RGBNT100**, **MSVR310**, **WMVEID863**, and **RGBNT201**, covering fixed modality-missing, randomly missing, and complete-modality settings.

<!-- ==================== Insert Figure here ==================== -->
<!--
<img width="1572" alt="The overall REMIND framework" src="<img width="2145" height="1010" alt="Fig2_01" src="https://github.com/user-attachments/assets/96d6b59e-53ea-4fd1-a18b-cceac9d7001a" />
" />
-->

<p align="center"><em>Figure . The overall REMIND framework, including Dual Memory Construction and Retrieval-Augmented Missing Reconstruction.</em></p>


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/skye-1201/REMIND.git
cd REMIND
```

### 2. Create the environment

A Linux machine with an NVIDIA GPU is recommended. The experiments reported in the paper were conducted on a single NVIDIA RTX 4090 GPU, while an RTX 3090 or RTX 4090 is recommended for training and evaluation.

```bash
conda create -n remind python=3.8 -y
conda activate remind
```

Install the required packages:

```bash
pip install -r requirements.txt
```

If the command above does not work properly, use:

```bash
python -m pip install -r requirements.txt
```

## Data and Pretrained Models

### Datasets

REMIND is evaluated on four RGB–NIR–TIR object Re-ID datasets. Please download the datasets from their official sources and follow their licenses and access policies.

A typical local organization is:

```text
/path/to/datasets/
├── RGBNT100/
├── MSVR310/
├── WMVEID863/
└── RGBNT201/
```

The corresponding configuration files are:

```text
RGBNT100:   configs/RGBNT100/REMIND.yml
MSVR310:    configs/MSVR310/REMIND.yml
WMVEID863:  configs/WMVEID863/REMIND.yml
RGBNT201:   configs/RGBNT201/REMIND.yml
```

Before training or evaluation, update the dataset paths in the corresponding configuration files according to your local environment.

### Pretrained Model

- **CLIP pretrained model:** [Baidu Pan](https://pan.baidu.com/s/1YPhaL0YgpI-TQ_pSzXHRKw)  
  Extraction code: `52fu`

After downloading the pretrained model, place it in your preferred directory and update the corresponding model path in the code or configuration file.

## Training

Activate the environment and move to the project directory before running the commands below.

### Example training command on MSVR310

```bash
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --config_file configs/MSVR310/REMIND.yml
```

## Evaluation

Replace `/path/to/output_or_checkpoint_dir` with the directory containing the model checkpoint to be evaluated.

### Example testing command on MSVR310:

```bash
CUDA_VISIBLE_DEVICES=0 python test_net.py \
  --config_file configs/MSVR310/REMIND.yml \
  OUTPUT_DIR /path/to/output_or_checkpoint_dir
```

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{Xu2026REMIND,
  author  = {Xu, Zhendong and Wang, Zi and Zheng, Aihua and Li, Chenglong and Tang, Jin},
  title   = {REMIND: Retrieval-Augmented Reconstruction With Dual Memories for Modality-Missing Object Re-Identification},
  journal = {IEEE Transactions on Image Processing},
  year    = {2026},
  volume  = {35},
  pages   = {8107--8120}
}
```

## Acknowledgements

This codebase is built upon [DeMo](https://github.com/924973292/DeMo). We thank the authors and maintainers of the datasets and open-source projects used in this work.

## Contact

For questions about the paper or code, please contact:

```text
zhendongxu1201@foxmail.com
```
