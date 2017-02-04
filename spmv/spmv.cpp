#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "meta.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

typedef double VALUE_TYPE;


	template <typename T>
inline std::string to_string(T value)
{
	std::ostringstream os ;
	os << value ;
	return os.str() ;
}

#define CL_CHECK(API) if((err=API)!=CL_SUCCESS){\
	printf("[err] id: %d, msg: %s, line: %d\n", err, get_error_string(err), __LINE__);\
	exit(-1); \
}

#define CHECK_ERROR(err) 	if(err != CL_SUCCESS){ \
	printf ("[err] %s\n", get_error_string (err)); \
	exit(-1); \
}

void csrmv_adaptive(unsigned int m, unsigned int n, unsigned int nnz, unsigned int *ptr, unsigned *idx, \
		VALUE_TYPE *value, VALUE_TYPE *x, VALUE_TYPE *y, VALUE_TYPE alpha)
{
	const cl_uint group_size = 256;
	std::string params = std::string( )
		+ " -DROWBITS=" + to_string( ROW_BITS )
		+ " -DWGBITS=" + to_string( WG_BITS )
		+ " -DVALUE_TYPE=" + "float"
		+ " -DWG_SIZE=" + to_string( group_size )
		+ " -DBLOCKSIZE=" + to_string( BLKSIZE )
		+ " -DBLOCK_MULTIPLIER=" + to_string( BLOCK_MULTIPLIER )
		+ " -DROWS_FOR_VECTOR=" + to_string( ROWS_FOR_VECTOR )
		+ " -DINDEX_TYPE=" + "unsigned";

	if(0)
	{
		params += " -DEXTENDED_PRECISION";
	}

	char filename[1024] = {"./kcode/csrmv_adaptive.cl"};
	// create context and build the kernel code
	cl_int status;
	cl_uint NumDevice;
	cl_platform_id platform;
	int platform_id = 1;
	char device_type[4] = {'g', 'p', 'u', '\0'};
	if(platform_id==0){
		device_type[0] = 'c';
		device_type[1] = 'p';
		device_type[2] = 'u';
	}
	else{
		device_type[0] = 'g';
		device_type[1] = 'p';
		device_type[2] = 'u';
	}
	getPlatform (platform, platform_id);
	printf("[info] the selected platform: %d, device type: %s\n", platform_id, device_type);
	cl_device_id * devices = getCl_device_id (platform, device_type);
	cl_context context = clCreateContext (NULL, 1, devices, NULL, NULL, NULL);
	status = clGetContextInfo (context, CL_CONTEXT_NUM_DEVICES, sizeof (cl_uint),
			&NumDevice, NULL);
	printf("[info] %d devices found!\n", NumDevice);
	cl_command_queue commandQueue = clCreateCommandQueue (context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);

	string sourceStr;
	status = convertToString (filename, sourceStr);
	printf("[info] The kernel to be compiled: %s\n", filename);
	const char *source = sourceStr.c_str ();
	size_t sourceSize[] = { strlen(source)};
	cl_program program = clCreateProgramWithSource (context, 1, &source, sourceSize, NULL);
	status = clBuildProgram (program, 1, devices, params.c_str(), NULL, NULL);
	if (status != CL_SUCCESS) {
		size_t length;
		clGetProgramBuildInfo (program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &length);
		char *buffer = (char *) malloc (length + 1);
		clGetProgramBuildInfo (program, devices[0], CL_PROGRAM_BUILD_LOG, length, buffer, NULL);
		printf ("build info: %s\n", buffer);
		if(buffer!= NULL) free(buffer);
	}

	// compute rowBlock buffer
	unsigned int rowBlockSize = ComputeRowBlocksSize(ptr, m, BLKSIZE, \
			BLOCK_MULTIPLIER, ROWS_FOR_VECTOR);
	printf("[info] row blocks: %d\n", rowBlockSize);
	unsigned long *rowblocks = (unsigned long *)malloc(rowBlockSize * sizeof(unsigned long));
	for(unsigned int i=0;i<rowBlockSize;i++){
		rowblocks[i] = 0;
	}
	ComputeRowBlocks<unsigned long>(rowblocks, rowBlockSize, ptr, m, BLKSIZE, \
			BLOCK_MULTIPLIER, ROWS_FOR_VECTOR, true);

	// create device buffers and move data to the device
	VALUE_TYPE *p_alpha = &alpha;	
	VALUE_TYPE beta = 0.0;
	VALUE_TYPE *p_beta = &beta;
	cl_int err;
	cl_mem	alphaBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,1*sizeof(VALUE_TYPE), (void *)p_alpha,&err);
	CHECK_ERROR(err);
	cl_mem	ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, (m+1)*sizeof(unsigned int),(void *)ptr, &err);
	CHECK_ERROR(err);
	cl_mem	idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,nnz*sizeof(unsigned int), (void *)idx, &err);
	CHECK_ERROR(err);
	cl_mem	valueBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,nnz*sizeof(VALUE_TYPE), (void *)value, &err);	
	CHECK_ERROR(err);
	cl_mem	xBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,n*sizeof(VALUE_TYPE), (void *)x, &err);
	CHECK_ERROR(err);
	cl_mem	betaBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,1*sizeof(VALUE_TYPE), (void *)p_beta, &err);
	CHECK_ERROR(err);
	cl_mem	yBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,m*sizeof(VALUE_TYPE), (void *)y, &err);
	CHECK_ERROR(err);
	cl_mem	rbBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,rowBlockSize*sizeof(unsigned long), (void *)rowblocks, &err);
	CHECK_ERROR(err);

	// creating and building kernels	
	cl_kernel csrmv_adaptive = clCreateKernel(program, "csrmv_adaptive", &err);
	CHECK_ERROR(err);

	CL_CHECK(clSetKernelArg(csrmv_adaptive, 0, sizeof(cl_mem), (void*)&valueBuffer));
	CL_CHECK(clSetKernelArg(csrmv_adaptive, 1, sizeof(cl_mem), (void*)&idxBuffer));
	CL_CHECK(clSetKernelArg(csrmv_adaptive, 2, sizeof(cl_mem), (void*)&ptrBuffer));
	CL_CHECK(clSetKernelArg(csrmv_adaptive, 3, sizeof(cl_mem), (void*)&xBuffer));
	CL_CHECK(clSetKernelArg(csrmv_adaptive, 4, sizeof(cl_mem), (void*)&yBuffer));
	CL_CHECK(clSetKernelArg(csrmv_adaptive, 5, sizeof(cl_mem), (void*)&rbBuffer));
	CL_CHECK(clSetKernelArg(csrmv_adaptive, 6, sizeof(cl_mem), (void*)&alphaBuffer));
	CL_CHECK(clSetKernelArg(csrmv_adaptive, 7, sizeof(cl_mem), (void*)&betaBuffer));

	// if NVIDIA is used it does not allow to run the group size
	// which is not a multiplication of group_size. Don't know if that
	// have an impact on performance
	// Setting global work size to half the row block size because we are only
	// using half the row blocks buffer for actual work.
	// The other half is used for the extended precision reduction.
	size_t global_work_size[1] = {((rowBlockSize/2)-1) * group_size};
	size_t local_work_size[1] = {group_size};
	global_work_size[0] = global_work_size[0] > local_work_size[0] ? global_work_size[0] : local_work_size[0];
	printf("[info] global work size: %ld, local work size: %ld\n", global_work_size[0], local_work_size[0]);

	cl_ulong t_spmv = 0;
	cl_ulong t_start = 0;
	cl_ulong t_end = 0;
	cl_event eventPoint;
	for(int i=0; i<NUM_RUN; i++){
		CL_CHECK(clEnqueueNDRangeKernel(commandQueue, csrmv_adaptive, 1, 
					NULL, global_work_size,
					local_work_size, 0, NULL,
					&eventPoint));
		CL_CHECK(clWaitForEvents(1, &eventPoint));
		CL_CHECK(clGetEventProfilingInfo(eventPoint, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, NULL));
		CL_CHECK(clGetEventProfilingInfo(eventPoint, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, NULL));
		t_spmv += t_end - t_start;			  
	}
	CL_CHECK(clReleaseEvent(eventPoint));
	printf("[info] spmv time: %lf\n", (double)t_spmv/(double)NUM_RUN*1e-6);

	CL_CHECK(clEnqueueReadBuffer(commandQueue, yBuffer, CL_TRUE, 0, m*sizeof(VALUE_TYPE), y, 0, NULL, NULL));	

	CL_CHECK(clReleaseMemObject(ptrBuffer));	//Release mem object.
	CL_CHECK(clReleaseMemObject(idxBuffer));	//Release mem object.
	CL_CHECK(clReleaseMemObject(valueBuffer));	//Release mem object.
	CL_CHECK(clReleaseMemObject(xBuffer));
	CL_CHECK(clReleaseMemObject(yBuffer));
	CL_CHECK(clReleaseMemObject(rbBuffer));
	CL_CHECK(clReleaseMemObject(alphaBuffer));
	CL_CHECK(clReleaseMemObject(betaBuffer));
	CL_CHECK(clReleaseCommandQueue(commandQueue));
	CL_CHECK(clReleaseKernel(csrmv_adaptive));
	CL_CHECK(clReleaseProgram(program));	//Release the program object.
	CL_CHECK(clReleaseContext(context));
	if(devices!=NULL) free(devices);
	if(rowblocks!=NULL) free(rowblocks);

	return ;
}	

void csrmv_vector(unsigned int m, unsigned int n, unsigned int nnz, unsigned int *ptr, unsigned *idx, \
		VALUE_TYPE *value, VALUE_TYPE *x, VALUE_TYPE *y, VALUE_TYPE alpha)
{
	int nnz_per_row = nnz/m;	// average nnz per row
	int wave_size = 32;	// 32 for NV hardware, 64 for AMD hardware
	int group_size = 256;    // 256 gives best performance!
	int subwave_size = wave_size;

	// adjust subwave_size according to nnz_per_row;
	// each wavefron will be assigned to the row of the csr matrix
	if(wave_size > 32)
	{
		//this apply only for devices with wavefront > 32 like AMD(64)
		if (nnz_per_row < 64) {  subwave_size = 32;  }
	}
	if (nnz_per_row < 32) {  subwave_size = 16;  }
	if (nnz_per_row < 16) {  subwave_size = 8;  }
	if (nnz_per_row < 8)  {  subwave_size = 4;  }
	if (nnz_per_row < 4)  {  subwave_size = 2;  }

	std::string params = std::string() +
		+ " -DVALUE_TYPE=" + "float"
		+ " -DSIZE_TYPE=" + "unsigned"
		+ " -DWG_SIZE=" + to_string(group_size)
		+ " -DWAVE_SIZE=" + to_string(wave_size)
		+ " -DSUBWAVE_SIZE=" + to_string(subwave_size)
		+ " -DINDEX_TYPE=" + "unsigned";
	if(0)
	{
		params += " -DEXTENDED_PRECISION";
	}

	char filename[1024] = {"./kcode/csrmv_general.cl"};
	// create context and build the kernel code
	cl_int status;
	cl_uint NumDevice;
	cl_platform_id platform;
	int platform_id = 1;
	char device_type[4] = {'g', 'p', 'u', '\0'};
	if(platform_id==0){
		device_type[0] = 'c';
		device_type[1] = 'p';
		device_type[2] = 'u';
	}
	else{
		device_type[0] = 'g';
		device_type[1] = 'p';
		device_type[2] = 'u';
	}
	getPlatform (platform, platform_id);
	printf("[info] the selected platform: %d, device type: %s\n", platform_id, device_type);
	cl_device_id * devices = getCl_device_id (platform, device_type);
	cl_context context = clCreateContext (NULL, 1, devices, NULL, NULL, NULL);
	status = clGetContextInfo (context, CL_CONTEXT_NUM_DEVICES, sizeof (cl_uint),
			&NumDevice, NULL);
	printf("[info] %d devices found!\n", NumDevice);
	cl_command_queue commandQueue = clCreateCommandQueue (context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);

	string sourceStr;
	status = convertToString (filename, sourceStr);
	printf("[info] The kernel to be compiled: %s\n", filename);
	const char *source = sourceStr.c_str ();
	size_t sourceSize[] = { strlen(source)};
	cl_program program = clCreateProgramWithSource (context, 1, &source, sourceSize, NULL);
	status = clBuildProgram (program, 1, devices, params.c_str(), NULL, NULL);
	if (status != CL_SUCCESS) {
		size_t length;
		clGetProgramBuildInfo (program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &length);
		char *buffer = (char *) malloc (length + 1);
		clGetProgramBuildInfo (program, devices[0], CL_PROGRAM_BUILD_LOG, length, buffer, NULL);
		printf ("build info: %s\n", buffer);
		if(buffer!= NULL) free(buffer);
	}

	VALUE_TYPE *p_alpha = &alpha;	
	unsigned off_alpha = 0;
	VALUE_TYPE beta = 0.0;
	VALUE_TYPE *p_beta = &beta;
	unsigned off_beta = 0;
	unsigned off_x = 0;
	unsigned off_y = 0;
	cl_int err;
	cl_mem	alphaBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,1*sizeof(VALUE_TYPE), (void *)p_alpha,&err);
	CHECK_ERROR(err);
	cl_mem	ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, (m+1)*sizeof(unsigned int),(void *)ptr, &err);
	CHECK_ERROR(err);
	cl_mem	idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,nnz*sizeof(unsigned int), (void *)idx, &err);
	CHECK_ERROR(err);
	cl_mem	valueBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,nnz*sizeof(VALUE_TYPE), (void *)value, &err);	
	CHECK_ERROR(err);
	cl_mem	xBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,n*sizeof(VALUE_TYPE), (void *)x, &err);
	CHECK_ERROR(err);
	cl_mem	betaBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,1*sizeof(VALUE_TYPE), (void *)p_beta, &err);
	CHECK_ERROR(err);
	cl_mem	yBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,m*sizeof(VALUE_TYPE), (void *)y, &err);
	CHECK_ERROR(err);

	// creating and building kernels	
	cl_kernel csrmv_vector = clCreateKernel(program, "csrmv_general", &err);
	CHECK_ERROR(err);

	CL_CHECK(clSetKernelArg(csrmv_vector, 0, sizeof(unsigned), &m));
	CL_CHECK(clSetKernelArg(csrmv_vector, 1, sizeof(cl_mem), (void*)&alphaBuffer));
	CL_CHECK(clSetKernelArg(csrmv_vector, 2, sizeof(unsigned), &off_alpha));
	CL_CHECK(clSetKernelArg(csrmv_vector, 3, sizeof(cl_mem), (void*)&ptrBuffer));
	CL_CHECK(clSetKernelArg(csrmv_vector, 4, sizeof(cl_mem), (void*)&idxBuffer));
	CL_CHECK(clSetKernelArg(csrmv_vector, 5, sizeof(cl_mem), (void*)&valueBuffer));
	CL_CHECK(clSetKernelArg(csrmv_vector, 6, sizeof(cl_mem), (void*)&xBuffer));
	CL_CHECK(clSetKernelArg(csrmv_vector, 7, sizeof(unsigned), &off_x));
	CL_CHECK(clSetKernelArg(csrmv_vector, 8, sizeof(cl_mem), (void*)&betaBuffer));
	CL_CHECK(clSetKernelArg(csrmv_vector, 9, sizeof(unsigned), &off_beta));
	CL_CHECK(clSetKernelArg(csrmv_vector, 10, sizeof(cl_mem), (void*)&yBuffer));
	CL_CHECK(clSetKernelArg(csrmv_vector, 11, sizeof(unsigned), &off_y));

	// subwave takes care of each row in matrix;
	// predicted number of subwaves to be executed;
	int predicted = subwave_size * m;

	// if NVIDIA is used it does not allow to run the group size
	// which is not a multiplication of group_size. Don't know if that
	// have an impact on performance
	size_t global_work_size[1] = {group_size* ((predicted + group_size - 1 ) / group_size)};
	size_t local_work_size[1] = {group_size};
	global_work_size[0] = global_work_size[0] > local_work_size[0] ? global_work_size[0] : local_work_size[0];
	//std::cout<<subwave_size<<", "<<m<<", "<<global_work_size[0]<<", "<<local_work_size[0]<<std::endl;

	cl_ulong t_spmv = 0;
	cl_ulong t_start = 0;
	cl_ulong t_end = 0;
	cl_event eventPoint;
	for(int i=0; i<NUM_RUN; i++){
		CL_CHECK(clEnqueueNDRangeKernel(commandQueue, csrmv_vector, 1, 
					NULL, global_work_size,
					local_work_size, 0, NULL,
					&eventPoint));
		CL_CHECK(clWaitForEvents(1, &eventPoint));
		clGetEventProfilingInfo(eventPoint, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, NULL);
		clGetEventProfilingInfo(eventPoint, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, NULL);
		t_spmv += t_end - t_start;			  
	}
	CL_CHECK(clReleaseEvent(eventPoint));
	printf("[info] spmv time: %lf\n", (double)t_spmv/(double)NUM_RUN*1e-6);

	CL_CHECK(clEnqueueReadBuffer(commandQueue, yBuffer, CL_TRUE, 0, m*sizeof(VALUE_TYPE), y, 0, NULL, NULL));	

	CL_CHECK(clReleaseMemObject(ptrBuffer));	//Release mem object.
	CL_CHECK(clReleaseMemObject(idxBuffer));	//Release mem object.
	CL_CHECK(clReleaseMemObject(valueBuffer));	//Release mem object.
	CL_CHECK(clReleaseMemObject(xBuffer));
	CL_CHECK(clReleaseMemObject(yBuffer));
	CL_CHECK(clReleaseMemObject(alphaBuffer));
	CL_CHECK(clReleaseMemObject(betaBuffer));
	CL_CHECK(clReleaseCommandQueue(commandQueue));
	CL_CHECK(clReleaseKernel(csrmv_vector));
	CL_CHECK(clReleaseProgram(program));	//Release the program object.
	CL_CHECK(clReleaseContext(context));
	if(devices!=NULL) free(devices);

	return ;
}

void csrmv_tb(unsigned int m, unsigned int n, unsigned int nnz, unsigned int *ptr, unsigned *idx, \
		VALUE_TYPE *value, VALUE_TYPE *x, VALUE_TYPE *y, VALUE_TYPE alpha, \
		parameter &cmdParam)
{
	int group_size = cmdParam.nThreadsPerBlock;    // 256 gives best performance!
	printf("[info] block size: %d\n", group_size);
	std::string params = std::string() +
		+ " -DVALUE_TYPE=" + "double"
		+ " -DWG_SIZE=" + to_string(group_size);

	char filename[1024] = {"./kcode/csrmv_tb.cl"};
	// create context and build the kernel code
	cl_int status;
	cl_uint NumDevice;
	cl_platform_id platform;
	int platform_id = cmdParam.platform_id;
	char device_type[4] = {'g', 'p', 'u', '\0'};
	if(platform_id==0){
		device_type[0] = 'c';
		device_type[1] = 'p';
		device_type[2] = 'u';
	}
	else if(platform_id==1){
		device_type[0] = 'g';
		device_type[1] = 'p';
		device_type[2] = 'u';
	}
	else if(platform_id==2){
		device_type[0] = 'm';
		device_type[1] = 'i';
		device_type[2] = 'c';
	}
	else{
		printf("[info] invalidate platform id!\n");
		exit(-1);
	}
	getPlatform (platform, platform_id);
	printf("[info] the selected platform: %d, device type: %s\n", platform_id, device_type);
	cl_device_id * devices = getCl_device_id (platform, device_type);
	cl_context context = clCreateContext (NULL, 1, devices, NULL, NULL, NULL);
	status = clGetContextInfo (context, CL_CONTEXT_NUM_DEVICES, sizeof (cl_uint),
			&NumDevice, NULL);
	printf("[info] %d devices found!\n", NumDevice);
	cl_command_queue commandQueue = clCreateCommandQueue (context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);

	string sourceStr;
	status = convertToString (filename, sourceStr);
	printf("[info] The kernel to be compiled: %s\n", filename);
	const char *source = sourceStr.c_str ();
	size_t sourceSize[] = { strlen(source)};
	cl_program program = clCreateProgramWithSource (context, 1, &source, sourceSize, NULL);
	status = clBuildProgram (program, 1, devices, params.c_str(), NULL, NULL);
	if (status != CL_SUCCESS) {
		size_t length;
		clGetProgramBuildInfo (program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &length);
		char *buffer = (char *) malloc (length + 1);
		clGetProgramBuildInfo (program, devices[0], CL_PROGRAM_BUILD_LOG, length, buffer, NULL);
		printf ("build info: %s\n", buffer);
		if(buffer!= NULL) free(buffer);
	}

	cl_int err;
	cl_mem	ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, (m+1)*sizeof(unsigned int),(void *)ptr, &err);
	CHECK_ERROR(err);
	cl_mem	idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,nnz*sizeof(unsigned int), (void *)idx, &err);
	CHECK_ERROR(err);
	cl_mem	valueBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,nnz*sizeof(VALUE_TYPE), (void *)value, &err);	
	CHECK_ERROR(err);
	cl_mem	xBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,n*sizeof(VALUE_TYPE), (void *)x, &err);
	CHECK_ERROR(err);
	cl_mem	yBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,m*sizeof(VALUE_TYPE), (void *)y, &err);
	CHECK_ERROR(err);

	// creating and building kernels	
	cl_kernel csrmv_tb = clCreateKernel(program, "csrmv_tb", &err);
	CHECK_ERROR(err);

	CL_CHECK(clSetKernelArg(csrmv_tb, 0, sizeof(cl_mem), (void*)&ptrBuffer));
	CL_CHECK(clSetKernelArg(csrmv_tb, 1, sizeof(cl_mem), (void*)&idxBuffer));
	CL_CHECK(clSetKernelArg(csrmv_tb, 2, sizeof(cl_mem), (void*)&valueBuffer));
	CL_CHECK(clSetKernelArg(csrmv_tb, 3, sizeof(cl_mem), (void*)&xBuffer));
	CL_CHECK(clSetKernelArg(csrmv_tb, 4, sizeof(cl_mem), (void*)&yBuffer));

	//CL_CHECK(clSetKernelArg(csrmv_tb, 0, sizeof(unsigned), &m));
	//CL_CHECK(clSetKernelArg(csrmv_tb, 1, sizeof(cl_mem), (void*)&alphaBuffer));
	//CL_CHECK(clSetKernelArg(csrmv_tb, 2, sizeof(unsigned), &off_alpha));
	//CL_CHECK(clSetKernelArg(csrmv_tb, 7, sizeof(unsigned), &off_x));
	//CL_CHECK(clSetKernelArg(csrmv_tb, 8, sizeof(cl_mem), (void*)&betaBuffer));
	//CL_CHECK(clSetKernelArg(csrmv_tb, 9, sizeof(unsigned), &off_beta));	
	//CL_CHECK(clSetKernelArg(csrmv_tb, 11, sizeof(unsigned), &off_y));

	// if NVIDIA is used it does not allow to run the group size
	// which is not a multiplication of group_size. Don't know if that
	// have an impact on performance
	size_t global_work_size[1] = {m * group_size};
	size_t local_work_size[1] = {group_size};
	//global_work_size[0] = global_work_size[0] > local_work_size[0] ? global_work_size[0] : local_work_size[0];

	cl_ulong t_spmv = 0;
	cl_ulong t_start = 0;
	cl_ulong t_end = 0;
	cl_event eventPoint;
	for(int i=0; i<NUM_RUN; i++){
		CL_CHECK(clEnqueueNDRangeKernel(commandQueue, csrmv_tb, 1, 
					NULL, global_work_size,
					local_work_size, 0, NULL,
					&eventPoint));
		CL_CHECK(clWaitForEvents(1, &eventPoint));
		clGetEventProfilingInfo(eventPoint, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, NULL);
		clGetEventProfilingInfo(eventPoint, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, NULL);
		t_spmv += t_end - t_start;			  
	}
	CL_CHECK(clReleaseEvent(eventPoint));
	printf("[info] spmv time: %lf\n", (double)t_spmv/(double)NUM_RUN*1e-6);

	CL_CHECK(clEnqueueReadBuffer(commandQueue, yBuffer, CL_TRUE, 0, m*sizeof(VALUE_TYPE), y, 0, NULL, NULL));	

	CL_CHECK(clReleaseMemObject(ptrBuffer));	//Release mem object.
	CL_CHECK(clReleaseMemObject(idxBuffer));	//Release mem object.
	CL_CHECK(clReleaseMemObject(valueBuffer));	//Release mem object.
	CL_CHECK(clReleaseMemObject(xBuffer));
	CL_CHECK(clReleaseMemObject(yBuffer));
	CL_CHECK(clReleaseCommandQueue(commandQueue));
	CL_CHECK(clReleaseKernel(csrmv_tb));
	CL_CHECK(clReleaseProgram(program));	//Release the program object.
	CL_CHECK(clReleaseContext(context));
	if(devices!=NULL) free(devices);

	return ;
}

int main(int argc, char** argv){
	char input_file_name[1024];
	char filename[1024] = {"./kcode/ccd01.cl"};
	parameter param = parse_command_line(argc, argv, input_file_name, NULL, filename);
	// reading rating matrix
	smat_t R;	// val: csc, val_t: csr
	load(input_file_name, R, false, false);
	unsigned int m = R.rows;
	unsigned int n = R.cols;
	unsigned int *ptr = R.row_ptr;
	unsigned int *idx = R.col_idx;
	unsigned int nnz = R.nnz;
	VALUE_TYPE * value = R.val_t;
	cout << " ( " << m << ", " << n << " ) nnz = " << nnz << endl;

	// native spmv
	VALUE_TYPE *x = (VALUE_TYPE *)malloc(n * sizeof(VALUE_TYPE));
	for(unsigned int i = 0; i < n; i++)
		x[i] = (VALUE_TYPE)i * 0.1;
	VALUE_TYPE *y = (VALUE_TYPE *)malloc(m * sizeof(VALUE_TYPE));
	VALUE_TYPE *y_ref = (VALUE_TYPE *)malloc(m * sizeof(VALUE_TYPE));

	// for debugging
	srand(time(NULL));
	for(unsigned int i=0; i<nnz; i++)
	{
		value[i] = rand() % 10;
	}

	// compute reference results on a cpu core
	VALUE_TYPE alpha = 1.0;
	double gb = getB<int, VALUE_TYPE>(m, nnz);
	double gflop = getFLOP<int>(nnz);
	double t = gettime();
	int ref_iter = 1;
	for (int iter = 0; iter < ref_iter; iter++){
		for (int i = 0; i < m; i++){
			VALUE_TYPE sum = 0;
			for (int j = ptr[i]; j < ptr[i+1]; j++)
				sum += x[idx[j]] * value[j] * alpha;
			y_ref[i] = sum;
		}
	}

	double ref_time = (gettime() - t) / (double)ref_iter;
	cout << "cpu sequential time = " << ref_time
		<< " ms. Bandwidth = " << gb/(1.0e+6 * ref_time)
		<< " GB/s. GFlops = " << gflop/(1.0e+6 * ref_time)  << " GFlops." << endl << endl;
	// compute spmv on the ocl device
	//csrmv_vector(m, n, nnz, ptr, idx, value, x, y, alpha);
	//csrmv_adaptive(m, n, nnz, ptr, idx, value, x, y, alpha);
	csrmv_tb(m, n, nnz, ptr, idx, value, x, y, alpha, param);

	// compare reference and anonymouslib results
	int error_count = 0;
	for (int i = 0; i < m; i++)
		if (abs(y_ref[i] - y[i]) > 0.01 * abs(y_ref[i]))
		{
			error_count++;
			cout << y_ref[i] << "," << y[i] << "\t";
		}
	if (error_count == 0)
		cout << "Check... PASS!" << endl;
	else
		cout << "Check... NO PASS! #Error = " << error_count << " out of " << m << " entries." << endl;
	cout << "------------------------------------------------------" << endl;

	if(x!=NULL) free(x);
	if(y!=NULL) free(y);
	if(y_ref!=NULL) free(y_ref);
	return 0;
}


