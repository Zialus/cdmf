#!/bin/bash
set -x

../exec/cdmf -k 40 -t 20 -T 1 -l 0.05 -nThreadsPerBlock 128 -p 1 -V 2 -P 1 -d 0 -r 1 -q 1 ../../DATASETS/jester/
