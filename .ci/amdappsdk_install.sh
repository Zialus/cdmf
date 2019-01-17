#!/bin/bash

set -exu

wget -q https://github.com/Microsoft/LightGBM/releases/download/v2.0.12/AMD-APP-SDKInstaller-v3.0.130.136-GA-linux64.tar.bz2
tar -xjf AMD-APP-SDK*.tar.bz2

export AMDAPPSDKROOT=$HOME/AMDAPPSDK
export OPENCL_VENDOR_PATH=${AMDAPPSDKROOT}/etc/OpenCL/vendors
export LD_LIBRARY_PATH=${AMDAPPSDKROOT}/lib/x86_64:${LD_LIBRARY_PATH-}

mkdir -p "$OPENCL_VENDOR_PATH"
mkdir -p "$AMDAPPSDKROOT"

sh AMD-APP-SDK*.sh --tar -xf -C "$AMDAPPSDKROOT"
mv "$AMDAPPSDKROOT"/lib/x86_64/sdk/* "$AMDAPPSDKROOT"/lib/x86_64/
echo libamdocl64.so > "$OPENCL_VENDOR_PATH"/amdocl64.icd

set +exu
