#!/bin/bash
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train_net.py --config_file configs/WMVEID863/REMIND.yml
