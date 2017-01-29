#include "util.h"
#include "anonymouslib_opencl.h"

int call_anonymouslib(int m, int n, int nnzA,
		int *csrRowPtrA, int *csrColIdxA, VALUE_TYPE *csrValA,
		VALUE_TYPE *x, VALUE_TYPE *y, VALUE_TYPE alpha)
{
	int err = 0;

	// set device
	BasicCL basicCL;

	char platformVendor[CL_STRING_LENGTH];
	char platformVersion[CL_STRING_LENGTH];

	char gpuDeviceName[CL_STRING_LENGTH];
	char gpuDeviceVersion[CL_STRING_LENGTH];
	int  gpuDeviceComputeUnits;
	cl_ulong  gpuDeviceGlobalMem;
	cl_ulong  gpuDeviceLocalMem;

	cl_uint             numPlatforms;           // OpenCL platform
	cl_platform_id*     cpPlatforms;

	cl_uint             numGpuDevices;          // OpenCL Gpu device
	cl_device_id*       cdGpuDevices;

	cl_context          cxGpuContext;           // OpenCL Gpu context
	cl_command_queue    cqGpuCommandQueue;      // OpenCL Gpu command queues

	bool profiling = true;
	int select_device = 0;

	// platform
	err = basicCL.getNumPlatform(&numPlatforms);
	if(err != CL_SUCCESS) return err;
	cout << "platform number: " << numPlatforms << ".  ";

	cpPlatforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);

	err = basicCL.getPlatformIDs(cpPlatforms, numPlatforms);
	if(err != CL_SUCCESS) return err;

	for (unsigned int i = 0; i < numPlatforms; i++)
	{
		err = basicCL.getPlatformInfo(cpPlatforms[i], platformVendor, platformVersion);
		if(err != CL_SUCCESS) return err;

		// Gpu device
		err = basicCL.getNumGpuDevices(cpPlatforms[i], &numGpuDevices);

		if (numGpuDevices > 0 && numGpuDevices < 5)
		{
			cdGpuDevices = (cl_device_id *)malloc(numGpuDevices * sizeof(cl_device_id) );

			err |= basicCL.getGpuDeviceIDs(cpPlatforms[i], numGpuDevices, cdGpuDevices);

			err |= basicCL.getDeviceInfo(cdGpuDevices[select_device], gpuDeviceName, gpuDeviceVersion,
					&gpuDeviceComputeUnits, &gpuDeviceGlobalMem,
					&gpuDeviceLocalMem, NULL);
			if(err != CL_SUCCESS) return err;

			cout << "Platform [" << i <<  "] Vendor: " << platformVendor << ", Version: " << platformVersion << endl;
			cout << "Using GPU device: " //<< numGpuDevices << " Gpu device: "
				<< gpuDeviceName << " ("
				<< gpuDeviceComputeUnits << " CUs, "
				<< gpuDeviceLocalMem / 1024 << " kB local, "
				<< gpuDeviceGlobalMem / (1024 * 1024) << " MB global, "
				<< gpuDeviceVersion << ")" << endl;

			break;
		}
		else
		{
			continue;
		}
	}

	// Gpu context
	err = basicCL.getContext(&cxGpuContext, cdGpuDevices, numGpuDevices);
	if(err != CL_SUCCESS) return err;

	// Gpu commandqueue
	if (profiling)
		err = basicCL.getCommandQueueProfilingEnable(&cqGpuCommandQueue, cxGpuContext, cdGpuDevices[select_device]);
	//else
	//    err = basicCL.getCommandQueue(&cqGpuCommandQueue, cxGpuContext, cdGpuDevices[select_device]);
	if(err != CL_SUCCESS) return err;


	double gb = getB<int, VALUE_TYPE>(m, nnzA);
	double gflop = getFLOP<int>(nnzA);

	// Define pointers of matrix A, vector x and y
	cl_mem      d_csrRowPtrA;
	cl_mem      d_csrColIdxA;
	cl_mem      d_csrValA;
	cl_mem      d_x;
	cl_mem      d_y;
	cl_mem      d_y_bench;

	// Matrix A
	d_csrRowPtrA = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, (m+1) * sizeof(int), NULL, &err);
	if(err != CL_SUCCESS) return err;
	d_csrColIdxA = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, nnzA  * sizeof(int), NULL, &err);
	if(err != CL_SUCCESS) return err;
	d_csrValA    = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, nnzA  * sizeof(VALUE_TYPE), NULL, &err);
	if(err != CL_SUCCESS) return err;

	err = clEnqueueWriteBuffer(cqGpuCommandQueue, d_csrRowPtrA, CL_TRUE, 0, (m+1) * sizeof(int), csrRowPtrA, 0, NULL, NULL);
	if(err != CL_SUCCESS) return err;
	err = clEnqueueWriteBuffer(cqGpuCommandQueue, d_csrColIdxA, CL_TRUE, 0, nnzA  * sizeof(int), csrColIdxA, 0, NULL, NULL);
	if(err != CL_SUCCESS) return err;
	err = clEnqueueWriteBuffer(cqGpuCommandQueue, d_csrValA, CL_TRUE, 0, nnzA  * sizeof(VALUE_TYPE), csrValA, 0, NULL, NULL);
	if(err != CL_SUCCESS) return err;

	// Vector x
	d_x    = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, n  * sizeof(VALUE_TYPE), NULL, &err);
	if(err != CL_SUCCESS) return err;
	err = clEnqueueWriteBuffer(cqGpuCommandQueue, d_x, CL_TRUE, 0, n  * sizeof(VALUE_TYPE), x, 0, NULL, NULL);
	if(err != CL_SUCCESS) return err;

	// Vector y
	d_y    = clCreateBuffer(cxGpuContext, CL_MEM_READ_WRITE, m  * sizeof(VALUE_TYPE), NULL, &err);
	if(err != CL_SUCCESS) return err;
	memset(y, 0, m  * sizeof(VALUE_TYPE));
	err = clEnqueueWriteBuffer(cqGpuCommandQueue, d_y, CL_TRUE, 0, m  * sizeof(VALUE_TYPE), y, 0, NULL, NULL);
	if(err != CL_SUCCESS) return err;

	d_y_bench    = clCreateBuffer(cxGpuContext, CL_MEM_READ_WRITE, m  * sizeof(VALUE_TYPE), NULL, &err);
	if(err != CL_SUCCESS) return err;
	err = clEnqueueWriteBuffer(cqGpuCommandQueue, d_y_bench, CL_TRUE, 0, m  * sizeof(VALUE_TYPE), y, 0, NULL, NULL);
	if(err != CL_SUCCESS) return err;




	double time = 0;

	anonymouslibHandle<int, unsigned int, VALUE_TYPE> A(m, n);
	err = A.setOCLENV(cxGpuContext, cqGpuCommandQueue);
	//cout << "setOCLENV err = " << err << endl;

	err = A.inputCSR(nnzA, d_csrRowPtrA, d_csrColIdxA, d_csrValA);
	//cout << "inputCSR err = " << err << endl;

	err = A.setX(d_x); // you only need to do it once!
	//cout << "setX err = " << err << endl;

	err = A.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA);
	//cout << "setSigma err = " << err << endl;

	// warmup device
	A.warmup();
	err = clFinish(cqGpuCommandQueue);

	anonymouslib_timer asCSR5_timer;
	asCSR5_timer.start();

	err = A.asCSR5();
	err = clFinish(cqGpuCommandQueue);

	cout << "CSR->CSR5 time = " << asCSR5_timer.stop() << " ms." << endl;
	//cout << "asCSR5 err = " << err << endl;

	// check correctness by running 1 time
	err = A.spmv(alpha, d_y, &time);
	//cout << "spmv err = " << err << endl;
	err = clEnqueueReadBuffer(cqGpuCommandQueue, d_y, CL_TRUE, 0, m * sizeof(VALUE_TYPE), y, 0, NULL, NULL);
	if(err != CL_SUCCESS) return err;

	// warm up by running 50 times
	if (NUM_RUN)
	{
		for (int i = 0; i < 50; i++)
			err = A.spmv(alpha, d_y_bench, &time);
	}

	err = clFinish(cqGpuCommandQueue);
	if(err != CL_SUCCESS) return err;

	double CSR5Spmv_time = 0;
	//anonymouslib_timer CSR5Spmv_timer;
	//CSR5Spmv_timer.start();

	// time spmv by running NUM_RUN times
	for (int i = 0; i < NUM_RUN; i++)
	{
		err = A.spmv(alpha, d_y_bench, &time);
		CSR5Spmv_time += time;
	}
	err = clFinish(cqGpuCommandQueue);
	//if(err != CL_SUCCESS) return err;

	//double CSR5Spmv_time = CSR5Spmv_timer.stop() / (double)NUM_RUN;
	CSR5Spmv_time = CSR5Spmv_time / (double)NUM_RUN;

	if (NUM_RUN)
		cout << "CSR5-based SpMV time = " << CSR5Spmv_time
			<< " ms. Bandwidth = " << gb/(1.0e+6 * CSR5Spmv_time)
			<< " GB/s. GFlops = " << gflop/(1.0e+6 * CSR5Spmv_time)  << " GFlops." << endl;

	A.destroy();

	if(d_csrRowPtrA) err = clReleaseMemObject(d_csrRowPtrA); if(err != CL_SUCCESS) return err;
	if(d_csrColIdxA) err = clReleaseMemObject(d_csrColIdxA); if(err != CL_SUCCESS) return err;
	if(d_csrValA) err = clReleaseMemObject(d_csrValA); if(err != CL_SUCCESS) return err;
	if(d_x) err = clReleaseMemObject(d_x); if(err != CL_SUCCESS) return err;
	if(d_y) err = clReleaseMemObject(d_y); if(err != CL_SUCCESS) return err;
	if(d_y_bench) err = clReleaseMemObject(d_y_bench); if(err != CL_SUCCESS) return err;

	return err;
}

void cdmf_csr5(smat_t &R, mat_t &W_c, mat_t &H_c, parameter &param)
{
	VALUE_TYPE alpha = 0.1f;
	unsigned m = R.rows;
	unsigned n = R.cols;
	unsigned nnz = R.nnz;
    	double gb = getB<int, VALUE_TYPE>(m, nnz);
	double gflop = getFLOP<int>(nnz);
	srand(time(NULL));
	VALUE_TYPE *x = (VALUE_TYPE *)malloc(n * sizeof(VALUE_TYPE));
	for (int i = 0; i < n; i++)
		x[i] = 1; 
	VALUE_TYPE *y = (VALUE_TYPE *)malloc(m * sizeof(VALUE_TYPE));
	VALUE_TYPE *y_ref = (VALUE_TYPE *)malloc(m * sizeof(VALUE_TYPE));

	// compute reference results on a cpu core
	anonymouslib_timer ref_timer;
	ref_timer.start();

	int ref_iter = 1;
	for (int iter = 0; iter < ref_iter; iter++)
	{
		for (int i = 0; i < m; i++)
		{
			VALUE_TYPE sum = 0;
			for (int j = R.row_ptr[i]; j < (R.row_ptr)[i+1]; j++)
				sum += x[(R.col_idx)[j]] * (R.val_t)[j] * alpha;
			y_ref[i] = sum;
		}
	}

	double ref_time = ref_timer.stop() / (double)ref_iter;
	cout << "cpu sequential time = " << ref_time
		<< " ms. Bandwidth = " << gb/(1.0e+6 * ref_time)
		<< " GB/s. GFlops = " << gflop/(1.0e+6 * ref_time)  << " GFlops." << endl << endl;

	call_anonymouslib(R.rows, R.cols, R.nnz, (int *)R.row_ptr, (int *)R.col_idx, R.val_t, x, y, alpha);

	// compare reference and anonymouslib results
	int error_count = 0;
	for (int i = 0; i < m; i++)
		if (fabs(y_ref[i] - y[i]) > 0.01 * fabs(y_ref[i]))
		{
			error_count++;
			//cout << "ROW [ " << i << " ], NNZ SPAN: "
			//	<< csrRowPtrA[i] << " - "
			//	<< csrRowPtrA[i+1]
			//	<< "\t ref = " << y_ref[i]
			//	<< ", \t csr5 = " << y[i]
			//	<< ", \t error = " << y_ref[i] - y[i]
			//	<< endl;
			//            break;

			//            //if (fabs(y_ref[i] - y[i]) > 0.00001)
			//            //    cout << ", \t error = " << y_ref[i] - y[i] << endl;
			//            //else
			//            //    cout << ". \t CORRECT!" << endl;
		}

	if (error_count == 0)
		cout << "Check... PASS!" << endl;
	else
		cout << "Check... NO PASS! #Error = " << error_count << " out of " << m << " entries." << endl;

	cout << "------------------------------------------------------" << endl;
	if(x!=NULL) free(x);
	if(y!=NULL) free(y);
	if(y_ref!=NULL) free(y_ref);

	/*	char device_type[4]={'g', 'p', 'u', '\0'};
		char input_file_name[1024];
		char *input_test_file;
		char filename[1024] = {"./kcode/ccd044.cl"};
		bool with_weights = false;

	// create context and build the kernel code
	cl_int status, err;
	cl_uint NumDevice;
	cl_platform_id platform;
	if(param.platform_id==0)
	{
	device_type[0] = 'c';
	device_type[1] = 'p';
	device_type[2] = 'u';
	}
	else
	{
	device_type[0] = 'g';
	device_type[1] = 'p';
	device_type[2] = 'u';
	}
	getPlatform (platform, param.platform_id);
	printf("[info] - the selected platform: %d, device type: %s\n", param.platform_id, device_type);
	cl_device_id * devices = getCl_device_id (platform, device_type);
	cl_context context = clCreateContext (NULL, 1, devices, NULL, NULL, NULL);
	status = clGetContextInfo (context, CL_CONTEXT_NUM_DEVICES, sizeof (cl_uint),
	&NumDevice, NULL);
	printf("[info] - %d devices found!\n", NumDevice);
	cl_command_queue commandQueue = clCreateCommandQueue (context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);

	string sourceStr;
	status = convertToString (filename, sourceStr);
	printf("[info] - The kernel to be compiled: %s\n", filename);
	const char *source = sourceStr.c_str ();
	size_t sourceSize[] = { strlen(source)};
	cl_program program = clCreateProgramWithSource (context, 1, &source, sourceSize, NULL);
	char options[1024];
	sprintf(options, "-DWG_SIZE=%d -DVALUE_TYPE=%s", param.nThreadsPerBlock, "double");
	status = clBuildProgram (program, 1, devices, options, NULL, NULL);
	if (status != CL_SUCCESS) 
	{
	size_t length;
	clGetProgramBuildInfo (program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &length);
	char *buffer = (char *) malloc (length + 1);
	clGetProgramBuildInfo (program, devices[0], CL_PROGRAM_BUILD_LOG, length, buffer, NULL);
	printf ("build info: %s\n", buffer);
	if(buffer!= NULL) free(buffer);
	}

	for (int t = 0; t < param.k; ++t)
	for (long c = 0; c < R.cols; ++c)
	H_c[t][c] = 0;
	unsigned num_updates = 0;
	unsigned k = param.k;
	VALUE_TYPE lambda = param.lambda;
	unsigned inneriter = param.maxinneriter;
	unsigned rows = R.rows;
	unsigned cols = R.cols;
	unsigned nBlocks = param.nBlocks;
	unsigned nThreadsPerBlock = param.nThreadsPerBlock;
	unsigned maxiter = param.maxiter;
	unsigned *col_ptr = R.col_ptr, *row_ptr = R.row_ptr;
	unsigned *row_idx = R.row_idx, *col_idx = R.col_idx;
	VALUE_TYPE *val = R.val;
	VALUE_TYPE *val_t = R.val_t;
	size_t nbits_u = R.rows * sizeof (VALUE_TYPE);
	size_t nbits_v = R.cols * sizeof (VALUE_TYPE);
	printf("[info] - blocks: %d, threads per block: %d\n", nBlocks, nThreadsPerBlock);

	VALUE_TYPE *Wt, *Ht;
	Wt = (VALUE_TYPE *) malloc (R.rows * sizeof (VALUE_TYPE));
	Ht = (VALUE_TYPE *) malloc (R.cols * sizeof (VALUE_TYPE));

	// creating buffers
	cl_mem    row_ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, R.nbits_row_ptr,(void *)row_ptr, &err);
	CHECK_ERROR(err);
	cl_mem    col_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,R.nbits_col_idx, (void *)col_idx, &err);
	CHECK_ERROR(err);
	cl_mem    col_ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, R.nbits_col_ptr,(void *)col_ptr, &err);
	CHECK_ERROR(err);
	cl_mem    row_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,R.nbits_row_idx, (void *)row_idx, &err);
	CHECK_ERROR(err);
	cl_mem    valBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,R.nbits_val, (void *)val, &err);
	CHECK_ERROR(err);
	cl_mem    val_tBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,R.nbits_val, (void *)val_t, &err);
	CHECK_ERROR(err);
	cl_mem WtBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_u, (void *) Wt, &err);	// u
	CHECK_ERROR(err);
	cl_mem HtBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_v, (void *) Ht, NULL);	// v
	CHECK_ERROR(err);

	// creating and building kernels
	cl_kernel RankOneUpdate_DUAL_kernel_u = clCreateKernel(program, "RankOneUpdate_DUAL_kernel_u", &err);
	CHECK_ERROR(err);
	cl_kernel RankOneUpdate_DUAL_kernel_v = clCreateKernel(program, "RankOneUpdate_DUAL_kernel_v", &err);
	CHECK_ERROR(err);
	cl_kernel UpdateRating_DUAL_kernel_NoLoss_r = clCreateKernel (program, "UpdateRating_DUAL_kernel_NoLoss_r", &err);
	CHECK_ERROR(err);
	cl_kernel UpdateRating_DUAL_kernel_NoLoss_c = clCreateKernel (program, "UpdateRating_DUAL_kernel_NoLoss_c", &err);
	CHECK_ERROR(err);
	cl_kernel UpdateRating_DUAL_kernel_NoLoss_r_ = clCreateKernel (program, "UpdateRating_DUAL_kernel_NoLoss_r_", &err);
	CHECK_ERROR(err);
	cl_kernel UpdateRating_DUAL_kernel_NoLoss_c_ = clCreateKernel (program, "UpdateRating_DUAL_kernel_NoLoss_c_", &err);
	CHECK_ERROR(err);

	// setting kernel arguments
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_u, 0, sizeof (unsigned), &cols));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_u, 1, sizeof (cl_mem), (void *) &col_ptrBuffer));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_u, 2, sizeof (cl_mem), (void *) &row_idxBuffer));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_u, 3, sizeof (cl_mem), (void *) &valBuffer));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_u, 4, sizeof (cl_mem), (void *) &WtBuffer));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_u, 5, sizeof (cl_mem), (void *) &HtBuffer));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_u, 6, sizeof (VALUE_TYPE), &lambda));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_u, 7, sizeof (unsigned), &rows));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_u, 8, sizeof (cl_mem), (void *) &row_ptrBuffer));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_u, 9, sizeof (cl_mem), (void *) &col_idxBuffer));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_u, 10, sizeof (cl_mem), (void *) &val_tBuffer));

	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_v, 0, sizeof (unsigned), &cols));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_v, 1, sizeof (cl_mem), (void *) &col_ptrBuffer));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_v, 2, sizeof (cl_mem), (void *) &row_idxBuffer));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_v, 3, sizeof (cl_mem), (void *) &valBuffer));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_v, 4, sizeof (cl_mem), (void *) &WtBuffer));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_v, 5, sizeof (cl_mem), (void *) &HtBuffer));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_v, 6, sizeof (VALUE_TYPE), &lambda));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_v, 7, sizeof (unsigned), &rows));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_v, 8, sizeof (cl_mem), (void *) &row_ptrBuffer));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_v, 9, sizeof (cl_mem), (void *) &col_idxBuffer));
	CL_CHECK(clSetKernelArg (RankOneUpdate_DUAL_kernel_v, 10, sizeof (cl_mem), (void *) &val_tBuffer));

	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r, 0, sizeof (unsigned), &cols)); 
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r, 1, sizeof (cl_mem), (void *) &col_ptrBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r, 2, sizeof (cl_mem), (void *) &row_idxBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r, 3, sizeof (cl_mem), (void *) &valBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r, 4, sizeof (cl_mem), &WtBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r, 5, sizeof (cl_mem), &HtBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r, 6, sizeof (unsigned), &rows));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r, 7, sizeof (cl_mem), (void *) &row_ptrBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r, 8, sizeof (cl_mem), (void *) &col_idxBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r, 9, sizeof (cl_mem), (void *) &val_tBuffer));

	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c, 0, sizeof (unsigned), &cols)); 
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c, 1, sizeof (cl_mem), (void *) &col_ptrBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c, 2, sizeof (cl_mem), (void *) &row_idxBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c, 3, sizeof (cl_mem), (void *) &valBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c, 4, sizeof (cl_mem), &WtBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c, 5, sizeof (cl_mem), &HtBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c, 6, sizeof (unsigned), &rows));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c, 7, sizeof (cl_mem), (void *) &row_ptrBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c, 8, sizeof (cl_mem), (void *) &col_idxBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c, 9, sizeof (cl_mem), (void *) &val_tBuffer));

	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r_, 0, sizeof (unsigned), &cols));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r_, 1, sizeof (cl_mem), (void *) &col_ptrBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r_, 2, sizeof (cl_mem), (void *) &row_idxBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r_, 3, sizeof (cl_mem), (void *) &valBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r_, 4, sizeof (cl_mem), &WtBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r_, 5, sizeof (cl_mem), &HtBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r_, 6, sizeof (unsigned), &rows));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r_, 7, sizeof (cl_mem), (void *) &row_ptrBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r_, 8, sizeof (cl_mem), (void *) &col_idxBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_r_, 9, sizeof (cl_mem), 	(void *) &val_tBuffer));

	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c_, 0, sizeof (unsigned), &cols));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c_, 1, sizeof (cl_mem), (void *) &col_ptrBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c_, 2, sizeof (cl_mem), (void *) &row_idxBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c_, 3, sizeof (cl_mem), (void *) &valBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c_, 4, sizeof (cl_mem), &WtBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c_, 5, sizeof (cl_mem), &HtBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c_, 6, sizeof (unsigned), &rows));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c_, 7, sizeof (cl_mem), (void *) &row_ptrBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c_, 8, sizeof (cl_mem), (void *) &col_idxBuffer));
	CL_CHECK(clSetKernelArg (UpdateRating_DUAL_kernel_NoLoss_c_, 9, sizeof (cl_mem), 	(void *) &val_tBuffer));

	size_t gws_row[1] = {rows * nThreadsPerBlock};
	size_t gws_col[1] = {cols * nThreadsPerBlock};

	cl_ulong t_update_ratings = 0;
	cl_ulong t_rank_one_update = 0;
	cl_ulong t_start;
	cl_ulong t_end;


	double t1 = gettime ();

	for (int oiter = 1; oiter <= maxiter; ++oiter)
	{
		//printf("[info] the %dth outter iteration.\n", oiter);
		size_t global_work_size[1] = {nBlocks *nThreadsPerBlock};
		size_t local_work_size[1] = {nThreadsPerBlock};
		for (int t = 0; t < k; ++t)
		{
			// Writing Buffer
			Wt = &(W_c[t][0]); // u
			Ht = &(H_c[t][0]); // v
			CL_CHECK(clEnqueueWriteBuffer(commandQueue, WtBuffer, CL_TRUE, 0, R.rows * sizeof (VALUE_TYPE), Wt, 0, NULL, NULL));
			CL_CHECK(clEnqueueWriteBuffer(commandQueue, HtBuffer, CL_TRUE, 0, R.cols * sizeof (VALUE_TYPE), Ht, 0, NULL, NULL));

			if (oiter > 1)
			{
				// update the rating matrix in CSC format (+)
				cl_event eventPoint;
				CL_CHECK(clEnqueueNDRangeKernel (commandQueue, UpdateRating_DUAL_kernel_NoLoss_c, 1, 
							NULL, gws_col, local_work_size, 0, NULL, &eventPoint));
				CL_CHECK(clWaitForEvents (1, &eventPoint));
				clGetEventProfilingInfo(eventPoint, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, NULL);
				clGetEventProfilingInfo(eventPoint, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, NULL);
				t_update_ratings += t_end - t_start;			  

				// update the rating matrix in CSR format (+)
				CL_CHECK(clEnqueueNDRangeKernel (commandQueue, UpdateRating_DUAL_kernel_NoLoss_r, 1, 
							NULL, gws_row, local_work_size, 0, NULL, &eventPoint));
				CL_CHECK(clWaitForEvents (1, &eventPoint));
				clGetEventProfilingInfo(eventPoint, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, NULL);
				clGetEventProfilingInfo(eventPoint, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, NULL);
				t_update_ratings += t_end - t_start;			  
				CL_CHECK(clReleaseEvent (eventPoint));
			}
			for (int iter = 1; iter <= inneriter; ++iter)
			{
				// update vector v
				cl_event eventPoint1v, eventPoint1u;
				CL_CHECK(clEnqueueNDRangeKernel (commandQueue, 	RankOneUpdate_DUAL_kernel_v, 1, NULL,
							gws_col, local_work_size, 0, NULL, &eventPoint1v));
				CL_CHECK(clWaitForEvents (1, &eventPoint1v));
				clGetEventProfilingInfo(eventPoint1v, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, NULL);
				clGetEventProfilingInfo(eventPoint1v, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, NULL);
				t_rank_one_update += t_end - t_start;

				// update vector u
				CL_CHECK(clEnqueueNDRangeKernel (commandQueue, 	RankOneUpdate_DUAL_kernel_u, 1, NULL,
							gws_row, local_work_size, 0, NULL, &eventPoint1u));
				CL_CHECK(clWaitForEvents (1, &eventPoint1u));
				clGetEventProfilingInfo(eventPoint1u, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, NULL);
				clGetEventProfilingInfo(eventPoint1u, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, NULL);
				t_rank_one_update += t_end - t_start;

				CL_CHECK(clReleaseEvent (eventPoint1v));
				CL_CHECK(clReleaseEvent (eventPoint1u));
			} 
			// Reading Buffer
			CL_CHECK(clEnqueueReadBuffer (commandQueue, WtBuffer, CL_TRUE, 0, R.rows * sizeof (VALUE_TYPE), Wt, 0, NULL, NULL));
			CL_CHECK(clEnqueueReadBuffer (commandQueue, HtBuffer, CL_TRUE, 0, R.cols * sizeof (VALUE_TYPE), Ht, 0, NULL, NULL));

			// update the rating matrix in CSC format (-)
			cl_event eventPoint2c, eventPoint2r;
			CL_CHECK(clEnqueueNDRangeKernel (commandQueue, 	UpdateRating_DUAL_kernel_NoLoss_c_, 1, NULL,
						gws_col, local_work_size, 0, NULL, &eventPoint2c));
			CL_CHECK(clWaitForEvents (1, &eventPoint2c));
			clGetEventProfilingInfo(eventPoint2c, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, NULL);
			clGetEventProfilingInfo(eventPoint2c, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, NULL);
			t_update_ratings += t_end - t_start;

			// update the rating matrix in CSR format (-)
			CL_CHECK(clEnqueueNDRangeKernel (commandQueue, UpdateRating_DUAL_kernel_NoLoss_r_, 1, NULL,
						gws_row, local_work_size, 0, NULL, &eventPoint2r));
			CL_CHECK(clWaitForEvents (1, &eventPoint2r));
			clGetEventProfilingInfo(eventPoint2r, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, NULL);
			clGetEventProfilingInfo(eventPoint2r, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, NULL);
			t_update_ratings += t_end - t_start;
			CL_CHECK(clReleaseEvent (eventPoint2c));
			CL_CHECK(clReleaseEvent (eventPoint2r));

		}
	}
	double t2 = gettime ();
	double deltaT = t2 - t1;
	printf("[info] - Training time: %lf s\n",  deltaT);
	printf("[info] - rank one updating time: %ld, R updating time: %ld\n", t_rank_one_update, t_update_ratings);
	// making prediction
	if(param.do_predict == 1)
	{
		double t5 = gettime ();
		int i, j;
		double vv, rmse = 0;
		size_t num_insts = 0;
		long vvv;
		FILE *test_fp = fopen (input_test_file, "r");
		if (test_fp == NULL)
		{
			printf ("can't open output file.\n");
			exit (1);
		}
		while (fscanf (test_fp, "%d %d %lf", &i, &j, &vv) != EOF)
		{
			double pred_v = 0;
			for (int t = 0; t < k; t++)
				pred_v += W_c[t][i - 1] * H_c[t][j - 1];
			num_insts++;
			rmse += (pred_v - vv) * (pred_v - vv);
		}
		rmse = sqrt (rmse / num_insts);
		printf ("[info] test RMSE = %lf.\n", rmse);
		double t6 = gettime ();
		double deltaT2 = t6 - t5;
		printf("[info] Predict time: %lf s\n", deltaT2);
	}

	// Release the context
	CL_CHECK(clReleaseMemObject(row_ptrBuffer));	//Release mem object.
	CL_CHECK(clReleaseMemObject(col_idxBuffer));	//Release mem object.
	CL_CHECK(clReleaseMemObject(col_ptrBuffer));	//Release mem object.
	CL_CHECK(clReleaseMemObject(row_idxBuffer));
	CL_CHECK(clReleaseMemObject(valBuffer));	//Release mem object.
	CL_CHECK(clReleaseMemObject(val_tBuffer));
	CL_CHECK(clReleaseMemObject (WtBuffer));
	CL_CHECK(clReleaseMemObject (HtBuffer));
	CL_CHECK(clReleaseCommandQueue(commandQueue));
	CL_CHECK(clReleaseKernel(UpdateRating_DUAL_kernel_NoLoss_c));
	CL_CHECK(clReleaseKernel(UpdateRating_DUAL_kernel_NoLoss_r));
	CL_CHECK(clReleaseKernel(UpdateRating_DUAL_kernel_NoLoss_c_));
	CL_CHECK(clReleaseKernel(UpdateRating_DUAL_kernel_NoLoss_r_));
	CL_CHECK(clReleaseKernel(RankOneUpdate_DUAL_kernel_u));	//Release kernel.
	CL_CHECK(clReleaseKernel(RankOneUpdate_DUAL_kernel_v));	//Release kernel.
	CL_CHECK(clReleaseProgram(program));	//Release the program object.
	CL_CHECK(clReleaseContext(context));
	free(devices);*/
		return ;
}

