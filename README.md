<p align="center">
  <h1 align="center">REMIND: Retrieval-Augmented Reconstruction With Dual Memories for Modality-Missing Object Re-Identification</h1>
</p>

---

## Introduction

This repository provides the official implementation of **REMIND: Retrieval-Augmented Reconstruction With Dual Memories for Modality-Missing Object Re-Identification**.

---

## Recommended Hardware

We recommend using an NVIDIA **RTX 3090** or **RTX 4090** GPU for training and evaluation.

---

## Environment Setup

We recommend using `conda` to create a new environment.

```bash
conda create -n remind python=3.8 -y
conda activate remind
```

Install the required packages:

```bash
pip install -r requirements.txt
```

If the `pip` command does not work properly, please use:

```bash
python -m pip install -r requirements.txt
```

---

## Pretrained Models

* **CLIP**: [Baidu Pan](https://pan.baidu.com/s/1YPhaL0YgpI-TQ_pSzXHRKw)  
  Extraction code: `52fu`

---

## Configuration

The configuration files for different datasets are listed as follows:

```text
RGBNT100:   configs/RGBNT100/DeMo.yml
MSVR310:    configs/MSVR310/DeMo.yml
WMVEID863:  configs/WMVEID863/DeMo.yml
RGBNT201:   configs/RGBNT201/DeMo.yml
```

---

## Training

Example training command on MSVR310:

```bash
conda activate remind
cd /path/to/REMIND

python train_net.py \
  --config_file configs/MSVR310/DeMo.yml
```

---

## Testing

Example testing command on MSVR310:

```bash
conda activate remind
cd /path/to/REMIND

python test_net.py \
  --config_file configs/MSVR310/DeMo.yml \
  OUTPUT_DIR /path/to/output_or_checkpoint_dir 

```


---

## Notes

This codebase is built upon [DeMo](https://github.com/924973292/DeMo).

---

## Contact

If you have any questions, please feel free to contact us by email:

```text
zhendongxu1201@foxmail.com
```