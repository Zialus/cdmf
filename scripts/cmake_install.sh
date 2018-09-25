#!/bin/bash

wget https://cmake.org/files/v3.12/cmake-3.12.2-Linux-x86_64.sh
sudo sh cmake-3.12.2-Linux-x86_64.sh --prefix="$HOME"/.local/ --exclude-subdir
