# REMIND

Official-style cleaned code for **REMIND: Retrieval-Augmented Reconstruction With Dual Memories for Modality-Missing Object Re-Identification**.

This version keeps the code aligned with the paper and removes unrelated debugging, visualization, cache, and legacy files. The main pipeline contains two core components:

- **Dual Memory Construction (DMC)**: extracts modality-specific and modality-common features and stores them in dedicated memory banks.
- **Retrieval-augmented Missing Reconstruction (RMR)**: retrieves identity-aware memory priors and reconstructs missing modality features through multi-path reconstruction.

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

`TEST.WEIGHT_PATH`, `TEST.MEMORY_PATH`, `TEST.MEMORY_N`, and latency-related settings are defined in the yml file. Command-line arguments are only kept as optional overrides.

## Notes

- `engine/processor.py` now only contains the training and inference loops.
- Epoch-dependent loss composition has been moved to `engine/remind_losses.py`.
- Configs are named `REMIND.yml` and old experiment/debug names are removed.
- All tunable REMIND hyperparameters are centralized under the `REMIND` section in each yml file, including memory warm-up/momentum, DMC attention settings, RMR retrieval and perturbation settings, and loss weights/schedule.
