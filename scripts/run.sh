#!/bin/bash
set -x

../exec/cdmf -k 20 -t 10 -T 1 -q 1 -P 1 -d 0 -l 0.05 -p 1 -V 2 -nThreadsPerBlock 64  "$1"
