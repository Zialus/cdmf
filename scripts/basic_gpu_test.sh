#!/bin/bash
set -x

./cdmf -k 40 -t 20 -T 1 -l 0.05 -nThreadsPerBlock 128 -p 1 -V 2 -P 1 -d 0 ../../DATASETS/jester/
