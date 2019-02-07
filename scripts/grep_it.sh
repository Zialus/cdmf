#!/bin/bash

rawA="$1.rmse"

grep "OCL Training time" "$1" | cut -d':' -f2 | cut -d' ' -f2 >> "${rawA}"
grep "Test RMSE" "$1" | cut -d'=' -f2 | cut -d' ' -f2 >> "${rawA}"
