# REMIND

Code for **REMIND: Retrieval-Augmented Reconstruction With Dual Memories for Modality-Missing Object Re-Identification**.

## Main files

```text
REMIND_refactored/
├── train_net.py
├── test_net.py
├── config/
├── configs/
│   ├── RGBNT100/REMIND.yml
│   ├── RGBNT201/REMIND.yml
│   ├── MSVR310/REMIND.yml
│   └── WMVEID863/REMIND.yml
├── data/
├── engine/
│   ├── processor.py
│   └── remind_losses.py
├── layers/
├── modeling/
│   ├── make_model.py
│   ├── meta_arch.py
│   ├── MemoryBank/
│   ├── clip/
│   └── moe/
├── solver/
└── utils/
```

## Training

```bash
python train_net.py --config_file configs/RGBNT201/REMIND.yml
python train_net.py --config_file configs/RGBNT100/REMIND.yml
python train_net.py --config_file configs/MSVR310/REMIND.yml
python train_net.py --config_file configs/WMVEID863/REMIND.yml
```

You can also run the dataset scripts:

```bash
bash RGBNT201.sh
bash RGBNT100.sh
bash MSVR310.sh
bash WMVEID863.sh
```

## Testing a missing-modality case

Set `TEST.MISS` by command line. Supported values are `r`, `n`, `t`, `rn`, `rt`, `nt`, and empty string for the complete-modality setting.

```bash
python test_net.py \
  --config_file configs/RGBNT201/REMIND.yml \
  TEST.MISS r
```
