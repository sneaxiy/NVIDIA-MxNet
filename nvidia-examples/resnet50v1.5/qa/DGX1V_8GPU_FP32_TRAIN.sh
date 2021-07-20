#!/bin/bash
../runner -n 8 -b 96 -e 5 --dtype float32 --seed 0 --dllogger-log report.json
python check_curves.py curve_baselines/dgx1v16g_8gpu_fp32.json report.json --epochs 5
