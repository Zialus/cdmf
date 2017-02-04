#compilers

#GLOBAL_PARAMETERS
VALUE_TYPE = double
NUM_RUN = 100

#ENVIRONMENT_PARAMETERS

#CUDA_PARAMETERS
NVCC_FLAGS = -O3  -w -m64 -arch=sm_35
#NVCC_FLAGS = -O3  -w -m64 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52
CUDA_INSTALL_PATH = /usr/local/cuda
CUDA_INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/samples/common/inc
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart

cuda:
	nvcc $(NVCC_FLAGS) spmv.cu util.cpp -o spmv $(CUDA_INCLUDES) $(CUDA_LIBS) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
