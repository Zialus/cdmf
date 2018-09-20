# CDMF Useful Info

AMD
```
cmake .. -DOpenCL_LIBRARY=/opt/rocm/opencl/lib/x86_64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/opt/rocm/opencl/include -DCMAKE_BUILD_TYPE=Release
```

NVIDIA
```
cmake .. -DCUDA_PATH=/usr/local/cuda-9.2/ -DCMAKE_BUILD_TYPE=Release
```
