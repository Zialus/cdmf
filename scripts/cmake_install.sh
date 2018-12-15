#!/bin/bash

wget https://github.com/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2-Linux-x86_64.sh
sudo sh cmake-3.13.2-Linux-x86_64.sh --prefix="$HOME"/.local/ --exclude-subdir
