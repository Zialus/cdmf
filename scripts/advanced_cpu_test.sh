#!/bin/bash
set -x

./cdmf -k 40 -t 20 -T 1 -l 0.05 -nThreadsPerBlock 32 -n 16 -p 1 -V 2 -P 0 -d 1 -r 1 -q 1 ../../DATASETS/jester/
