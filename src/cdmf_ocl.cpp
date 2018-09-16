#include "util.h"

void cdmf_ocl(smat_t& R, mat_t& W_c, mat_t& H_c, parameter& param, char filename[]) {
    char device_type[4] = {'\0', '\0', '\0', '\0'};

    switch (param.platform_id) {
        case 0:
            snprintf(device_type, sizeof(device_type), "gpu");
            break;
        case 1:
            snprintf(device_type, sizeof(device_type), "cpu");
            break;
        case 2:
            snprintf(device_type, sizeof(device_type), "mic");
            break;
        default:
            printf("[info] unknown device type!\n");
            break;
    }
    printf("[info] - selected device type: %s\n", device_type);

    if (param.verbose) {
        print_all_the_platforms();
        print_all_the_info();
    }


    // create context and build the kernel code
    cl_int status;
    cl_int err;
    cl_platform_id platform;
    getPlatform(platform, 0);
    cl_device_id* devices = getDevice(platform, device_type);
    report_device(devices[0]);
    cl_context context = clCreateContext(nullptr, 1, devices, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_uint NumDevice;
    CL_CHECK(clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &NumDevice, nullptr));
    assert(NumDevice == 1);
    cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, nullptr);

    printf("[info] - The kernel to be compiled: %s\n", filename);
    string sourceStr;
    convertToString(filename, sourceStr);
    const char* source = sourceStr.c_str();
    size_t sourceSize[] = {strlen(source)};
    cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, nullptr);
    char options[1024];
    sprintf(options, "-DWG_SIZE=%u -DVALUE_TYPE=%s", param.nThreadsPerBlock, getT(sizeof(VALUE_TYPE)));
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

    for (int t = 0; t < param.k; ++t) {
        for (long c = 0; c < R.cols; ++c) {
            H_c[t][c] = 0;
        }
    }

    unsigned rows = R.rows;
    unsigned cols = R.cols;
    unsigned* col_ptr = R.col_ptr, * row_ptr = R.row_ptr;
    unsigned* row_idx = R.row_idx, * col_idx = R.col_idx;
    VALUE_TYPE* val = R.val;
    VALUE_TYPE* val_t = R.val_t;

    VALUE_TYPE* Wt = (VALUE_TYPE*) malloc(R.rows * sizeof(VALUE_TYPE));
    VALUE_TYPE* Ht = (VALUE_TYPE*) malloc(R.cols * sizeof(VALUE_TYPE));

    size_t nbits_u = R.rows * sizeof(VALUE_TYPE);
    size_t nbits_v = R.cols * sizeof(VALUE_TYPE);

    // creating buffers
    cl_mem row_ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_row_ptr, (void*) row_ptr, &err);
    CL_CHECK(err);
    cl_mem col_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_col_idx, (void*) col_idx, &err);
    CL_CHECK(err);
    cl_mem col_ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_col_ptr, (void*) col_ptr, &err);
    CL_CHECK(err);
    cl_mem row_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_row_idx, (void*) row_idx, &err);
    CL_CHECK(err);
    cl_mem valBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_val, (void*) val, &err);
    CL_CHECK(err);
    cl_mem val_tBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_val, (void*) val_t, &err);
    CL_CHECK(err);
    cl_mem WtBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_u, (void*) Wt, &err); // u
    CL_CHECK(err);
    cl_mem HtBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_v, (void*) Ht, &err); // v
    CL_CHECK(err);

    // creating and building kernels
    cl_kernel RankOneUpdate_DUAL_kernel_u = clCreateKernel(program, "RankOneUpdate_DUAL_kernel_u", &err);
    CL_CHECK(err);
    cl_kernel RankOneUpdate_DUAL_kernel_v = clCreateKernel(program, "RankOneUpdate_DUAL_kernel_v", &err);
    CL_CHECK(err);
    cl_kernel UpdateRating_DUAL_kernel_NoLoss_r = clCreateKernel(program, "UpdateRating_DUAL_kernel_NoLoss_r", &err);
    CL_CHECK(err);
    cl_kernel UpdateRating_DUAL_kernel_NoLoss_c = clCreateKernel(program, "UpdateRating_DUAL_kernel_NoLoss_c", &err);
    CL_CHECK(err);
    cl_kernel UpdateRating_DUAL_kernel_NoLoss_r_ = clCreateKernel(program, "UpdateRating_DUAL_kernel_NoLoss_r_", &err);
    CL_CHECK(err);
    cl_kernel UpdateRating_DUAL_kernel_NoLoss_c_ = clCreateKernel(program, "UpdateRating_DUAL_kernel_NoLoss_c_", &err);
    CL_CHECK(err);

    // setting kernel arguments
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 0, sizeof(unsigned), &cols));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 1, sizeof(cl_mem), (void*) &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 2, sizeof(cl_mem), (void*) &row_idxBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 3, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 4, sizeof(cl_mem), (void*) &WtBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 5, sizeof(cl_mem), (void*) &HtBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 6, sizeof(VALUE_TYPE), &param.lambda));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 7, sizeof(unsigned), &rows));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 8, sizeof(cl_mem), (void*) &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 9, sizeof(cl_mem), (void*) &col_idxBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 10, sizeof(cl_mem), (void*) &val_tBuffer));

    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 0, sizeof(unsigned), &cols));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 1, sizeof(cl_mem), (void*) &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 2, sizeof(cl_mem), (void*) &row_idxBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 3, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 4, sizeof(cl_mem), (void*) &WtBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 5, sizeof(cl_mem), (void*) &HtBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 6, sizeof(VALUE_TYPE), &param.lambda));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 7, sizeof(unsigned), &rows));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 8, sizeof(cl_mem), (void*) &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 9, sizeof(cl_mem), (void*) &col_idxBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 10, sizeof(cl_mem), (void*) &val_tBuffer));

    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 0, sizeof(unsigned), &cols));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 1, sizeof(cl_mem), (void*) &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 2, sizeof(cl_mem), (void*) &row_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 3, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 4, sizeof(cl_mem), &WtBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 5, sizeof(cl_mem), &HtBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 6, sizeof(unsigned), &rows));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 7, sizeof(cl_mem), (void*) &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 8, sizeof(cl_mem), (void*) &col_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 9, sizeof(cl_mem), (void*) &val_tBuffer));

    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 0, sizeof(unsigned), &cols));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 1, sizeof(cl_mem), (void*) &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 2, sizeof(cl_mem), (void*) &row_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 3, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 4, sizeof(cl_mem), &WtBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 5, sizeof(cl_mem), &HtBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 6, sizeof(unsigned), &rows));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 7, sizeof(cl_mem), (void*) &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 8, sizeof(cl_mem), (void*) &col_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 9, sizeof(cl_mem), (void*) &val_tBuffer));

    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 0, sizeof(unsigned), &cols));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 1, sizeof(cl_mem), (void*) &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 2, sizeof(cl_mem), (void*) &row_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 3, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 4, sizeof(cl_mem), &WtBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 5, sizeof(cl_mem), &HtBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 6, sizeof(unsigned), &rows));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 7, sizeof(cl_mem), (void*) &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 8, sizeof(cl_mem), (void*) &col_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 9, sizeof(cl_mem), (void*) &val_tBuffer));

    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 0, sizeof(unsigned), &cols));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 1, sizeof(cl_mem), (void*) &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 2, sizeof(cl_mem), (void*) &row_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 3, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 4, sizeof(cl_mem), &WtBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 5, sizeof(cl_mem), &HtBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 6, sizeof(unsigned), &rows));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 7, sizeof(cl_mem), (void*) &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 8, sizeof(cl_mem), (void*) &col_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 9, sizeof(cl_mem), (void*) &val_tBuffer));

    size_t gws_row[1] = {rows * param.nThreadsPerBlock};
    size_t gws_col[1] = {cols * param.nThreadsPerBlock};
    size_t local_work_size[1] = {param.nThreadsPerBlock};
    printf("[info] - threads per block: %u\n", param.nThreadsPerBlock);

    size_t local;
    CL_CHECK(clGetKernelWorkGroupInfo(RankOneUpdate_DUAL_kernel_u, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL));
    printf("local_work_size for RankOneUpdate_DUAL_kernel_u should be: %zu\n",local);
    CL_CHECK(clGetKernelWorkGroupInfo(RankOneUpdate_DUAL_kernel_v, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL));
    printf("local_work_size for RankOneUpdate_DUAL_kernel_u should be: %zu\n",local);

    cl_ulong t_update_ratings = 0;
    cl_ulong t_rank_one_update = 0;
    cl_ulong t_start;
    cl_ulong t_end;

    double t1 = gettime();
    for (int oiter = 1; oiter <= param.maxiter; ++oiter) {
        //printf("[info] the %dth outter iteration.\n", oiter);
        for (int t = 0; t < param.k; ++t) {
            // Writing Buffer
            Wt = &(W_c[t][0]); // u
            Ht = &(H_c[t][0]); // v
            CL_CHECK(clEnqueueWriteBuffer(commandQueue, WtBuffer, CL_TRUE, 0, R.rows * sizeof(VALUE_TYPE), Wt, 0, nullptr, nullptr));
            CL_CHECK(clEnqueueWriteBuffer(commandQueue, HtBuffer, CL_TRUE, 0, R.cols * sizeof(VALUE_TYPE), Ht, 0, nullptr, nullptr));

            if (oiter > 1) {
                // update the rating matrix in CSC format (+)
                cl_event eventPoint0c, eventPoint0r;
                CL_CHECK(clEnqueueNDRangeKernel(commandQueue, UpdateRating_DUAL_kernel_NoLoss_c, 1, nullptr, gws_col, local_work_size, 0, nullptr, &eventPoint0c));
                CL_CHECK(clWaitForEvents(1, &eventPoint0c));
                clGetEventProfilingInfo(eventPoint0c, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, nullptr);
                clGetEventProfilingInfo(eventPoint0c, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, nullptr);
                t_update_ratings += t_end - t_start;

                // update the rating matrix in CSR format (+)
                CL_CHECK(clEnqueueNDRangeKernel(commandQueue, UpdateRating_DUAL_kernel_NoLoss_r, 1, nullptr, gws_row, local_work_size, 0, nullptr, &eventPoint0r));
                CL_CHECK(clWaitForEvents(1, &eventPoint0r));
                clGetEventProfilingInfo(eventPoint0r, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, nullptr);
                clGetEventProfilingInfo(eventPoint0r, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, nullptr);
                t_update_ratings += t_end - t_start;
                CL_CHECK(clReleaseEvent(eventPoint0c));
                CL_CHECK(clReleaseEvent(eventPoint0r));
            }
            for (int iter = 1; iter <= param.maxinneriter; ++iter) {
                // update vector v
                cl_event eventPoint1v, eventPoint1u;
                CL_CHECK(clEnqueueNDRangeKernel(commandQueue, RankOneUpdate_DUAL_kernel_v, 1, nullptr, gws_col, local_work_size, 0, nullptr, &eventPoint1v));
                CL_CHECK(clWaitForEvents(1, &eventPoint1v));
                clGetEventProfilingInfo(eventPoint1v, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, nullptr);
                clGetEventProfilingInfo(eventPoint1v, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, nullptr);
                t_rank_one_update += t_end - t_start;

                // update vector u
                CL_CHECK(clEnqueueNDRangeKernel(commandQueue, RankOneUpdate_DUAL_kernel_u, 1, nullptr, gws_row, local_work_size, 0, nullptr, &eventPoint1u));
                CL_CHECK(clWaitForEvents(1, &eventPoint1u));
                clGetEventProfilingInfo(eventPoint1u, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, nullptr);
                clGetEventProfilingInfo(eventPoint1u, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, nullptr);
                t_rank_one_update += t_end - t_start;

                CL_CHECK(clReleaseEvent(eventPoint1v));
                CL_CHECK(clReleaseEvent(eventPoint1u));
            }
            // Reading Buffer
            CL_CHECK(clEnqueueReadBuffer(commandQueue, WtBuffer, CL_TRUE, 0, R.rows * sizeof(VALUE_TYPE), Wt, 0, nullptr, nullptr));
            CL_CHECK(clEnqueueReadBuffer(commandQueue, HtBuffer, CL_TRUE, 0, R.cols * sizeof(VALUE_TYPE), Ht, 0, nullptr, nullptr));

            // update the rating matrix in CSC format (-)
            cl_event eventPoint2c, eventPoint2r;
            CL_CHECK(clEnqueueNDRangeKernel(commandQueue, UpdateRating_DUAL_kernel_NoLoss_c_, 1, nullptr, gws_col, local_work_size, 0, nullptr, &eventPoint2c));
            CL_CHECK(clWaitForEvents(1, &eventPoint2c));
            clGetEventProfilingInfo(eventPoint2c, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, nullptr);
            clGetEventProfilingInfo(eventPoint2c, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, nullptr);
            t_update_ratings += t_end - t_start;

            // update the rating matrix in CSR format (-)
            CL_CHECK(clEnqueueNDRangeKernel(commandQueue, UpdateRating_DUAL_kernel_NoLoss_r_, 1, nullptr, gws_row, local_work_size, 0, nullptr, &eventPoint2r));
            CL_CHECK(clWaitForEvents(1, &eventPoint2r));
            clGetEventProfilingInfo(eventPoint2r, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, nullptr);
            clGetEventProfilingInfo(eventPoint2r, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, nullptr);
            t_update_ratings += t_end - t_start;

            CL_CHECK(clReleaseEvent(eventPoint2c));
            CL_CHECK(clReleaseEvent(eventPoint2r));

        }
    }
    double t2 = gettime();
    double deltaT = t2 - t1;
    printf("[info] - training time: %lf s\n", deltaT);
    printf("[info] - rank one updating time: %llu ms, R updating time: %llu ms\n", t_rank_one_update / 1000000ULL, t_update_ratings / 1000000ULL);

    CL_CHECK(clReleaseMemObject(row_ptrBuffer));
    CL_CHECK(clReleaseMemObject(col_idxBuffer));
    CL_CHECK(clReleaseMemObject(col_ptrBuffer));
    CL_CHECK(clReleaseMemObject(row_idxBuffer));
    CL_CHECK(clReleaseMemObject(valBuffer));
    CL_CHECK(clReleaseMemObject(val_tBuffer));
    CL_CHECK(clReleaseMemObject(WtBuffer));
    CL_CHECK(clReleaseMemObject(HtBuffer));
    CL_CHECK(clReleaseCommandQueue(commandQueue));
    CL_CHECK(clReleaseKernel(UpdateRating_DUAL_kernel_NoLoss_c));
    CL_CHECK(clReleaseKernel(UpdateRating_DUAL_kernel_NoLoss_r));
    CL_CHECK(clReleaseKernel(UpdateRating_DUAL_kernel_NoLoss_c_));
    CL_CHECK(clReleaseKernel(UpdateRating_DUAL_kernel_NoLoss_r_));
    CL_CHECK(clReleaseKernel(RankOneUpdate_DUAL_kernel_u));
    CL_CHECK(clReleaseKernel(RankOneUpdate_DUAL_kernel_v));
    CL_CHECK(clReleaseProgram(program));
    CL_CHECK(clReleaseContext(context));
    free(devices);
}

