#include "util.h"
#include "anonymouslib_opencl.h"

void cdmf_csr5(smat_t &R, mat_t &W_c, mat_t &H_c, parameter &param, char filename[])
{
    char device_type[4] = {'\0', '\0', '\0', '\0'};

    // create context and build the kernel code
    cl_int status;
    cl_int err;
    cl_platform_id platform;

    if (param.platform_id == 0) {
        device_type[0] = 'g';
        device_type[1] = 'p';
        device_type[2] = 'u';
    } else if (param.platform_id == 1) {
        device_type[0] = 'c';
        device_type[1] = 'p';
        device_type[2] = 'u';
    } else if (param.platform_id == 2) {
        device_type[0] = 'm';
        device_type[1] = 'i';
        device_type[2] = 'c';
    } else {
        printf("[info] unknown device type!\n");
        exit(-1);
    }
    printf("[info] - selected device type: %s\n", device_type);

    if (param.verbose) {
        print_all_the_platforms();
        print_all_the_info();
    }

    getPlatform(platform, 0);
    cl_device_id* devices = getDevice(platform, device_type);

    cl_context context = clCreateContext(nullptr, 1, devices, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_uint NumDevice;
    CL_CHECK(clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &NumDevice, nullptr));
    assert(NumDevice == 1);
    cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, nullptr);

    printf("[info] - The kernel to be compiled: %s\n", filename);
    string sourceStr;
    status = convertToString(filename, sourceStr);
    if (status == -1) { exit(-1); }
    const char* source = sourceStr.c_str();
    size_t sourceSize[] = {strlen(source)};
    cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, nullptr);
    char options[1024];
    sprintf(options, "-DWG_SIZE=%d -DVALUE_TYPE=%s", param.nThreadsPerBlock, getT(sizeof(VALUE_TYPE)));
    status = clBuildProgram(program, 1, devices, options, nullptr, nullptr);

    size_t length;
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
    char* buffer = (char*) malloc(length + 1);
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, length, buffer, nullptr);

    if (buffer != nullptr) {
        printf("[build info]:\n%s", buffer);
        free(buffer);
    }

    CL_CHECK(status)
    printf("[build info]: Compiled OpenCl code !\n");

    for (int t = 0; t < param.k; ++t)
        for (unsigned c = 0; c < R.cols; ++c)
            H_c[t][c] = 0;

    unsigned m = R.rows;
    unsigned n = R.cols;
    unsigned nnz = R.nnz;
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
    printf("[info] - blocks: %u, threads per block: %u\n", nBlocks, nThreadsPerBlock);

    double gb = getB<int, VALUE_TYPE>(m, nnz);
    double gflop = getFLOP<int>(nnz);

    VALUE_TYPE* Wt = (VALUE_TYPE*) malloc(R.rows * sizeof(VALUE_TYPE));
    VALUE_TYPE* Ht = (VALUE_TYPE*) malloc(R.cols * sizeof(VALUE_TYPE));
    memset(Ht, 0, cols * sizeof(VALUE_TYPE));
    memset(Wt, 0, rows * sizeof(VALUE_TYPE));

    // buffers to store the bottom results
    VALUE_TYPE* Hb = (VALUE_TYPE*) malloc(R.cols * sizeof(VALUE_TYPE));
    VALUE_TYPE* Wb = (VALUE_TYPE*) malloc(R.rows * sizeof(VALUE_TYPE));
    memset(Hb, 0, cols * sizeof(VALUE_TYPE));
    memset(Wb, 0, rows * sizeof(VALUE_TYPE));

    // creating buffers
    cl_mem row_ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, R.nbits_row_ptr,(void *)row_ptr, &err);
    CL_CHECK(err);
    cl_mem col_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,R.nbits_col_idx, (void *)col_idx, &err);
    CL_CHECK(err);
    cl_mem col_ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, R.nbits_col_ptr,(void *)col_ptr, &err);
    CL_CHECK(err);
    cl_mem row_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,R.nbits_row_idx, (void *)row_idx, &err);
    CL_CHECK(err);
    cl_mem valBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,R.nbits_val, (void *)val, &err);
    CL_CHECK(err);
    cl_mem val_tBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR ,R.nbits_val, (void *)val_t, &err);
    CL_CHECK(err);
    cl_mem WBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_u, (void *) Wt, &err); // u
    CL_CHECK(err);
    cl_mem WtBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_u, (void *) Wt, &err); // u
    CL_CHECK(err);
    cl_mem WbBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_u, (void *) Wb, &err); // u
    CL_CHECK(err);
    cl_mem HBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_v, (void *) Ht, &err);  // v
    CL_CHECK(err);
    cl_mem HtBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_v, (void *) Ht, &err); // v
    CL_CHECK(err);
    cl_mem HbBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_v, (void *) Hb, &err); // v
    CL_CHECK(err);

    // creating and building kernels
    cl_kernel UpdateRating_DUAL_kernel_NoLoss_r = clCreateKernel(program, "UpdateRating_DUAL_kernel_NoLoss_r", &err);
    CL_CHECK(err);
    cl_kernel UpdateRating_DUAL_kernel_NoLoss_c = clCreateKernel(program, "UpdateRating_DUAL_kernel_NoLoss_c", &err);
    CL_CHECK(err);
    cl_kernel UpdateRating_DUAL_kernel_NoLoss_r_ = clCreateKernel(program, "UpdateRating_DUAL_kernel_NoLoss_r_", &err);
    CL_CHECK(err);
    cl_kernel UpdateRating_DUAL_kernel_NoLoss_c_ = clCreateKernel(program, "UpdateRating_DUAL_kernel_NoLoss_c_", &err);
    CL_CHECK(err);
    cl_kernel _kernel_CALV = clCreateKernel(program, "CALV", &err);
    CL_CHECK(err);
    cl_kernel _kernel_CALU = clCreateKernel(program, "CALU", &err);
    CL_CHECK(err);

    // setting kernel arguments
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 0, sizeof(unsigned), &cols));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 1, sizeof(cl_mem), (void*) &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 2, sizeof(cl_mem), (void*) &row_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 3, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 4, sizeof(cl_mem), &WBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 5, sizeof(cl_mem), &HBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 6, sizeof(unsigned), &rows));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 7, sizeof(cl_mem), (void*) &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 8, sizeof(cl_mem), (void*) &col_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 9, sizeof(cl_mem), (void*) &val_tBuffer));

    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 0, sizeof(unsigned), &cols));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 1, sizeof(cl_mem), (void*) &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 2, sizeof(cl_mem), (void*) &row_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 3, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 4, sizeof(cl_mem), &WBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 5, sizeof(cl_mem), &HBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 6, sizeof(unsigned), &rows));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 7, sizeof(cl_mem), (void*) &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 8, sizeof(cl_mem), (void*) &col_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 9, sizeof(cl_mem), (void*) &val_tBuffer));

    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 0, sizeof(unsigned), &cols));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 1, sizeof(cl_mem), (void*) &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 2, sizeof(cl_mem), (void*) &row_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 3, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 4, sizeof(cl_mem), &WBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 5, sizeof(cl_mem), &HBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 6, sizeof(unsigned), &rows));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 7, sizeof(cl_mem), (void*) &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 8, sizeof(cl_mem), (void*) &col_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 9, sizeof(cl_mem), (void*) &val_tBuffer));

    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 0, sizeof(unsigned), &cols));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 1, sizeof(cl_mem), (void*) &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 2, sizeof(cl_mem), (void*) &row_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 3, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 4, sizeof(cl_mem), &WBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 5, sizeof(cl_mem), &HBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 6, sizeof(unsigned), &rows));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 7, sizeof(cl_mem), (void*) &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 8, sizeof(cl_mem), (void*) &col_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 9, sizeof(cl_mem), (void*) &val_tBuffer));

    CL_CHECK(clSetKernelArg(_kernel_CALV, 0, sizeof(unsigned), &cols));
    CL_CHECK(clSetKernelArg(_kernel_CALV, 1, sizeof(cl_mem), (void*) &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(_kernel_CALV, 2, sizeof(cl_mem), (void*) &HtBuffer));
    CL_CHECK(clSetKernelArg(_kernel_CALV, 3, sizeof(cl_mem), (void*) &HbBuffer));
    CL_CHECK(clSetKernelArg(_kernel_CALV, 4, sizeof(cl_mem), (void*) &HBuffer));
    CL_CHECK(clSetKernelArg(_kernel_CALV, 5, sizeof(VALUE_TYPE), &lambda));

    CL_CHECK(clSetKernelArg(_kernel_CALU, 0, sizeof(unsigned), &rows));
    CL_CHECK(clSetKernelArg(_kernel_CALU, 1, sizeof(cl_mem), (void*) &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(_kernel_CALU, 2, sizeof(cl_mem), (void*) &WtBuffer));
    CL_CHECK(clSetKernelArg(_kernel_CALU, 3, sizeof(cl_mem), (void*) &WbBuffer));
    CL_CHECK(clSetKernelArg(_kernel_CALU, 4, sizeof(cl_mem), (void*) &WBuffer));
    CL_CHECK(clSetKernelArg(_kernel_CALU, 5, sizeof(VALUE_TYPE), &lambda));

    size_t gws_row[1] = {rows * nThreadsPerBlock};
    size_t gws_col[1] = {cols * nThreadsPerBlock};

    double time = 0.0;
    anonymouslibHandle<int, unsigned int, VALUE_TYPE> Av(cols, rows);
    CL_CHECK(Av.setOCLENV(context, commandQueue, devices));
    CL_CHECK(Av.inputCSR(nnz, col_ptrBuffer, row_idxBuffer, valBuffer));
    CL_CHECK(Av.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA));
    anonymouslib_timer asCSR5_timer;
    asCSR5_timer.start();
    CL_CHECK(Av.asCSR5());
    CL_CHECK(clFinish(commandQueue));
    cout << "Av: CSR->CSR5 time = " << asCSR5_timer.stop() << " ms." << endl;

    anonymouslibHandle<int, unsigned int, VALUE_TYPE> Au(rows, cols);
    CL_CHECK(Au.setOCLENV(context, commandQueue, devices));
    CL_CHECK(Au.inputCSR(nnz, row_ptrBuffer, col_idxBuffer, val_tBuffer));
    CL_CHECK(Au.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA));
    asCSR5_timer.start();
    CL_CHECK(Au.asCSR5());
    CL_CHECK(clFinish(commandQueue));
    cout << "Au: CSR->CSR5 time = " << asCSR5_timer.stop() << " ms." << endl;

    CL_CHECK(Av.setX(WBuffer)); // you only need to do it once!
    CL_CHECK(Au.setX(HBuffer)); // you only need to do it once!

    cl_ulong t_update_ratings = 0;
    cl_ulong t_rank_one_update = 0;
    cl_ulong t_start;
    cl_ulong t_end;

    double t1 = gettime ();
    for (int oiter = 1; oiter <= maxiter; ++oiter)
    {
        size_t global_work_size[1] = {nBlocks *nThreadsPerBlock};
        size_t local_work_size[1] = {nThreadsPerBlock};
        for (int t = 0; t < k; ++t)
        {
            // Writing Buffer
            Wt = &(W_c[t][0]); // u
            Ht = &(H_c[t][0]); // v
            CL_CHECK(clEnqueueWriteBuffer(commandQueue, WBuffer, CL_TRUE, 0, R.rows * sizeof (VALUE_TYPE), Wt, 0,
                                          nullptr,
                                          nullptr));
            CL_CHECK(clEnqueueWriteBuffer(commandQueue, HBuffer, CL_TRUE, 0, R.cols * sizeof (VALUE_TYPE), Ht, 0,
                                          nullptr,
                                          nullptr));

            /*if (oiter > 1)
            {
                //Av.asCSR_();
                //Au.asCSR_();
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

                //Av.asCSR5_();
                //Au.asCSR5_();
            }*/
            for (int iter = 1; iter <= inneriter; ++iter)
            {
                cl_event eventPoint1v, eventPoint1u;

                // update vector v
                CL_CHECK(Av.spmv(1.0, HtBuffer, HbBuffer, &time));
                CL_CHECK(clEnqueueNDRangeKernel (commandQueue, _kernel_CALV, 1, nullptr, gws_col, local_work_size, 0,
                                                 nullptr, &eventPoint1v));
                CL_CHECK(clWaitForEvents (1, &eventPoint1v));

                // update vector u
                CL_CHECK(Au.spmv(1.0, WtBuffer, WbBuffer, &time));
                CL_CHECK(clEnqueueNDRangeKernel (commandQueue, _kernel_CALU, 1, nullptr, gws_row, local_work_size, 0,
                                                 nullptr, &eventPoint1u));
                CL_CHECK(clWaitForEvents (1, &eventPoint1u));

                CL_CHECK(clReleaseEvent (eventPoint1v));
                CL_CHECK(clReleaseEvent (eventPoint1u));
            }
            // Reading Buffer
            CL_CHECK(clEnqueueReadBuffer (commandQueue, WBuffer, CL_TRUE, 0, R.rows * sizeof (VALUE_TYPE), Wt, 0,
                                          nullptr,
                                          nullptr));
            CL_CHECK(clEnqueueReadBuffer (commandQueue, HBuffer, CL_TRUE, 0, R.cols * sizeof (VALUE_TYPE), Ht, 0,
                                          nullptr,
                                          nullptr));
            //Av.asCSR_();
            //Au.asCSR_();
            // update the rating matrix in CSC format (-)
            /*cl_event eventPoint2c, eventPoint2r;
            CL_CHECK(clEnqueueNDRangeKernel (commandQueue,  UpdateRating_DUAL_kernel_NoLoss_c_, 1, NULL,
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
            CL_CHECK(clReleaseEvent (eventPoint2r));*/

            //Av.asCSR5_();
            //Au.asCSR5_();

        }
    }
    double t2 = gettime ();
    double deltaT = t2 - t1;
    printf("[info] - training time: %lf s\n",  deltaT);
    printf("[info] - rank one updating time: %llu ms, R updating time: %llu ms\n", t_rank_one_update/1000000ULL, t_update_ratings/1000000ULL);

    // Release the context
    Au.destroy();
    Av.destroy();
    CL_CHECK(clReleaseMemObject(row_ptrBuffer));
    CL_CHECK(clReleaseMemObject(col_idxBuffer));
    CL_CHECK(clReleaseMemObject(col_ptrBuffer));
    CL_CHECK(clReleaseMemObject(row_idxBuffer));
    CL_CHECK(clReleaseMemObject(valBuffer));
    CL_CHECK(clReleaseMemObject(val_tBuffer));
    CL_CHECK(clReleaseMemObject(WBuffer));
    CL_CHECK(clReleaseMemObject(WtBuffer));
    CL_CHECK(clReleaseMemObject(WbBuffer));
    CL_CHECK(clReleaseMemObject(HBuffer));
    CL_CHECK(clReleaseMemObject(HtBuffer));
    CL_CHECK(clReleaseMemObject(HbBuffer));
    CL_CHECK(clReleaseCommandQueue(commandQueue));
    CL_CHECK(clReleaseKernel(UpdateRating_DUAL_kernel_NoLoss_c));
    CL_CHECK(clReleaseKernel(UpdateRating_DUAL_kernel_NoLoss_r));
    CL_CHECK(clReleaseKernel(UpdateRating_DUAL_kernel_NoLoss_c_));
    CL_CHECK(clReleaseKernel(UpdateRating_DUAL_kernel_NoLoss_r_));
    CL_CHECK(clReleaseKernel(_kernel_CALV));
    CL_CHECK(clReleaseKernel(_kernel_CALU));
    CL_CHECK(clReleaseProgram(program));
    CL_CHECK(clReleaseContext(context));
    if(devices) free(devices);
    if(Wb)  free(Wb);
    if(Hb)  free(Hb);

}
