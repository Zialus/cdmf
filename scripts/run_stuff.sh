#!/bin/bash

set -x

SIZES=(4 8 16 32 64 128 256 512 1024 2048)

for size in "${SIZES[@]}"; do
    ../exec/cdmf -k 40 -t 20 -T 1 -P 1 -d 0 -l 0.05 -p 1 -V 2 -nThreadsPerBlock "${size}" "$1"
done
