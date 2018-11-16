#!/bin/bash

set -exu

###########################
# Get Intel OpenCL Runtime
###########################

# https://software.intel.com/en-us/articles/opencl-drivers#cpu-section
PACKAGE_URL=http://registrationcenter-download.intel.com/akdlm/irc_nas/vcp/13793/l_opencl_p_18.1.0.013.tgz
PACKAGE_NAME=l_opencl_p_18.1.0.013

wget -q ${PACKAGE_URL} -O /tmp/opencl_runtime.tgz
tar -xzf /tmp/opencl_runtime.tgz -C /tmp
sed 's/decline/accept/g' -i /tmp/${PACKAGE_NAME}/silent.cfg
sudo /tmp/${PACKAGE_NAME}/install.sh -s /tmp/${PACKAGE_NAME}/silent.cfg

set +exu
