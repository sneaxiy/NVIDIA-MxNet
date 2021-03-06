#!/bin/bash
python ../benchmark.py --executable ../runner -n 1 -b 1,2,4,8,16,64,128,192,256 -i 800 -e 2 -w 1 --num-examples 38400 --mode val -o report.json
python check_perf.py benchmark_baselines/dgx1v16g_infer_fp16_dali.json report.json --metrics val.ips
