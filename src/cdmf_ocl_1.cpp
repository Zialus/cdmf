#include "tools.h"

extern std::chrono::duration<double> deltaT12;
extern std::chrono::duration<double> deltaTAB;

void cdmf_ocl_01(smat_t& R, mat_t& W_c, mat_t& H_c, testset_t &T, parameter& param, char filename[]) {
    auto tA = std::chrono::high_resolution_clock::now();

    cl_int status;
    cl_int err;
    cl_platform_id platform = getPlatform(param.platform_id);
    cl_device_id* devices = getDevices(platform, param.device_type);
    report_device(devices[0]);
    cl_context context = clCreateContext(nullptr, 1, devices, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_uint NumDevice;
    CL_CHECK(clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &NumDevice, nullptr));
    assert(NumDevice == 1);
    cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);
    printf("[INFO] Connected!\n");

    printf("[INFO] - The kernel to be compiled: %s\n", filename);
    std::string sourceStr;
    convertToString(filename, sourceStr);
    const char* source = sourceStr.c_str();
    size_t sourceSize[] = {strlen(source)};
    cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, &err);
    CL_CHECK(err);

    char options[1024];
    snprintf(options, sizeof(options), "-DVALUE_TYPE=%s", getT(sizeof(VALUE_TYPE)));
    status = clBuildProgram(program, 1, devices, options, nullptr, nullptr);

    size_t length;
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
    char* buffer = (char*) malloc(length + 1);
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, length, buffer, nullptr);

    if (buffer != nullptr && strcmp(buffer, "") != 0 && strcmp(buffer, "\n") != 0) {
        printf("[OpenCL Compiler INFO]:\n%s\n", buffer);
        free(buffer);
    } else {
        printf("[OpenCL Compiler]: No info to print\n");
    }

    CL_CHECK(status);
    std::cout << "[INFO]: Compiled OpenCl code successfully!\n";

    CL_CHECK(clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, 0, nullptr, &length));
    char* buffer2 = (char*) malloc(length + 1);
    CL_CHECK(clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, length, buffer2, nullptr));
    if (buffer2 != nullptr && param.verbose) {
        printf("[Kernels]: %s\n", buffer2);
        free(buffer2);
    }

    auto tB = std::chrono::high_resolution_clock::now();
    deltaTAB = tB - tA;
    std::cout << "[INFO] Initiating OpenCL Time: " << deltaTAB.count() << " s.\n";

    unsigned k = param.k;

    for (unsigned t = 0; t < param.k; ++t) {
        for (unsigned c = 0; c < R.cols; ++c) {
            H_c[t][c] = 0;
        }
    }

    VALUE_TYPE* W = (VALUE_TYPE*) malloc(k * R.rows * sizeof(VALUE_TYPE));
    for (unsigned i = 0; i < k; ++i) {
        for (unsigned j = 0; j < R.rows; ++j) {
            W[i * R.rows + j] = W_c[i][j];
        }
    }

    VALUE_TYPE* H = (VALUE_TYPE*) malloc(k * R.cols * sizeof(VALUE_TYPE));
    for (unsigned i = 0; i < k; ++i) {
        for (unsigned j = 0; j < R.cols; ++j) {
            H[i * R.cols + j] = 0;
        }
    }

    VALUE_TYPE* Wt = (VALUE_TYPE*) malloc(R.rows * sizeof(VALUE_TYPE));
    VALUE_TYPE* Ht = (VALUE_TYPE*) malloc(R.cols * sizeof(VALUE_TYPE));

    size_t nbits_u = R.rows * sizeof(VALUE_TYPE);
    size_t nbits_v = R.cols * sizeof(VALUE_TYPE);

    size_t nbits_W_ = R.rows * k * sizeof(VALUE_TYPE);
    size_t nbits_H_ = R.cols * k * sizeof(VALUE_TYPE);

    // creating buffers
    cl_mem WBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_W_, (void*) W, &err);
    CL_CHECK(err);
    cl_mem HBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_H_, (void*) H, &err);
    CL_CHECK(err);

    cl_mem row_ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_row_ptr, (void*) R.row_ptr, &err);
    CL_CHECK(err);
    cl_mem col_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_col_idx, (void*) R.col_idx, &err);
    CL_CHECK(err);
    cl_mem col_ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_col_ptr, (void*) R.col_ptr, &err);
    CL_CHECK(err);
    cl_mem row_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_row_idx, (void*) R.row_idx, &err);
    CL_CHECK(err);
    cl_mem valBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_val, (void*) R.val, &err);
    CL_CHECK(err);
    cl_mem val_tBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_val, (void*) R.val_t, &err);
    CL_CHECK(err);
    cl_mem WtBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_u, (void*) Wt, &err); // u
    CL_CHECK(err);
    cl_mem HtBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_v, (void*) Ht, &err); // v
    CL_CHECK(err);

    // RMSE related buffers
    cl_mem test_rowBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, T.nnz * sizeof(unsigned), (void*) T.test_row, &err);
    CL_CHECK(err);
    cl_mem test_colBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, T.nnz * sizeof(unsigned), (void*) T.test_col, &err);
    CL_CHECK(err);
    cl_mem test_valBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, T.nnz * sizeof(unsigned), (void*) T.test_val, &err);
    CL_CHECK(err);
    cl_mem pred_vBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, T.nnz * sizeof(VALUE_TYPE), nullptr, &err);
    CL_CHECK(err);
    cl_mem rmseBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, T.nnz * sizeof(VALUE_TYPE), nullptr, &err);
    CL_CHECK(err);
    cl_mem emptyBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, T.nnz * sizeof(VALUE_TYPE), nullptr, &err);
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

    cl_kernel gpuRMSE_kernel = clCreateKernel(program, "GPU_rmse", &err);
    CL_CHECK(err);

    // setting kernel arguments
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 0, sizeof(unsigned), &R.cols));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 1, sizeof(cl_mem), (void*) &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 2, sizeof(cl_mem), (void*) &row_idxBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 3, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 4, sizeof(cl_mem), (void*) &WtBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 5, sizeof(cl_mem), (void*) &HtBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 6, sizeof(VALUE_TYPE), &param.lambda));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 7, sizeof(unsigned), &R.rows));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 8, sizeof(cl_mem), (void*) &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 9, sizeof(cl_mem), (void*) &col_idxBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_u, 10, sizeof(cl_mem), (void*) &val_tBuffer));

    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 0, sizeof(unsigned), &R.cols));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 1, sizeof(cl_mem), (void*) &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 2, sizeof(cl_mem), (void*) &row_idxBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 3, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 4, sizeof(cl_mem), (void*) &WtBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 5, sizeof(cl_mem), (void*) &HtBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 6, sizeof(VALUE_TYPE), &param.lambda));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 7, sizeof(unsigned), &R.rows));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 8, sizeof(cl_mem), (void*) &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 9, sizeof(cl_mem), (void*) &col_idxBuffer));
    CL_CHECK(clSetKernelArg(RankOneUpdate_DUAL_kernel_v, 10, sizeof(cl_mem), (void*) &val_tBuffer));

    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 0, sizeof(unsigned), &R.cols));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 1, sizeof(cl_mem), (void*) &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 2, sizeof(cl_mem), (void*) &row_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 3, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 4, sizeof(cl_mem), &WtBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 5, sizeof(cl_mem), &HtBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 6, sizeof(unsigned), &R.rows));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 7, sizeof(cl_mem), (void*) &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 8, sizeof(cl_mem), (void*) &col_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 9, sizeof(cl_mem), (void*) &val_tBuffer));

    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 0, sizeof(unsigned), &R.cols));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 1, sizeof(cl_mem), (void*) &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 2, sizeof(cl_mem), (void*) &row_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 3, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 4, sizeof(cl_mem), &WtBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 5, sizeof(cl_mem), &HtBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 6, sizeof(unsigned), &R.rows));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 7, sizeof(cl_mem), (void*) &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 8, sizeof(cl_mem), (void*) &col_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 9, sizeof(cl_mem), (void*) &val_tBuffer));

    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 0, sizeof(unsigned), &R.cols));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 1, sizeof(cl_mem), (void*) &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 2, sizeof(cl_mem), (void*) &row_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 3, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 4, sizeof(cl_mem), &WtBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 5, sizeof(cl_mem), &HtBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 6, sizeof(unsigned), &R.rows));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 7, sizeof(cl_mem), (void*) &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 8, sizeof(cl_mem), (void*) &col_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 9, sizeof(cl_mem), (void*) &val_tBuffer));

    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 0, sizeof(unsigned), &R.cols));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 1, sizeof(cl_mem), (void*) &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 2, sizeof(cl_mem), (void*) &row_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 3, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 4, sizeof(cl_mem), &WtBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 5, sizeof(cl_mem), &HtBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 6, sizeof(unsigned), &R.rows));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 7, sizeof(cl_mem), (void*) &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 8, sizeof(cl_mem), (void*) &col_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 9, sizeof(cl_mem), (void*) &val_tBuffer));

    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 0, sizeof(cl_mem), (void*) &test_rowBuffer));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 1, sizeof(cl_mem), (void*) &test_colBuffer));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 2, sizeof(cl_mem), (void*) &test_valBuffer));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 3, sizeof(cl_mem), (void*) &pred_vBuffer));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 4, sizeof(cl_mem), (void*) &rmseBuffer));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 5, sizeof(cl_mem), (void*) &WBuffer));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 6, sizeof(cl_mem), (void*) &HBuffer));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 7, sizeof(unsigned), &T.nnz));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 8, sizeof(unsigned), &param.k));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 9, sizeof(unsigned), &R.rows));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 10, sizeof(unsigned), &R.cols));

//    size_t gws_row[1] = {static_cast<size_t>(R.rows * param.nThreadsPerBlock)};
//    size_t gws_col[1] = {static_cast<size_t>(R.cols * param.nThreadsPerBlock)};
    size_t global_work_size[1] = {static_cast<size_t>(param.nBlocks * param.nThreadsPerBlock)};
    size_t local_work_size[1] = {static_cast<size_t>(param.nThreadsPerBlock)};
    printf("[INFO] - blocks: %d | threads per block: %d | global_work_size: %zu | local_work_size: %zu !\n",
           param.nBlocks, param.nThreadsPerBlock, global_work_size[0], local_work_size[0]);

    if (param.verbose) {
        size_t local;
        CL_CHECK(clGetKernelWorkGroupInfo(RankOneUpdate_DUAL_kernel_u, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, nullptr));
        printf("[VERBOSE] local_work_size for RankOneUpdate_DUAL_kernel_u should be: %zu\n", local);
        CL_CHECK(clGetKernelWorkGroupInfo(RankOneUpdate_DUAL_kernel_v, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, nullptr));
        printf("[VERBOSE] local_work_size for RankOneUpdate_DUAL_kernel_u should be: %zu\n", local);
        CL_CHECK(clGetKernelWorkGroupInfo(UpdateRating_DUAL_kernel_NoLoss_r, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, nullptr));
        printf("[VERBOSE] local_work_size for UpdateRating_DUAL_kernel_NoLoss_r should be: %zu\n", local);
        CL_CHECK(clGetKernelWorkGroupInfo(UpdateRating_DUAL_kernel_NoLoss_c, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, nullptr));
        printf("[VERBOSE] local_work_size for UpdateRating_DUAL_kernel_NoLoss_c should be: %zu\n", local);
        CL_CHECK(clGetKernelWorkGroupInfo(UpdateRating_DUAL_kernel_NoLoss_r_, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, nullptr));
        printf("[VERBOSE] local_work_size for UpdateRating_DUAL_kernel_NoLoss_r_ should be: %zu\n", local);
        CL_CHECK(clGetKernelWorkGroupInfo(UpdateRating_DUAL_kernel_NoLoss_c_, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, nullptr));
        printf("[VERBOSE] local_work_size for UpdateRating_DUAL_kernel_NoLoss_c_ should be: %zu\n", local);
        CL_CHECK(clGetKernelWorkGroupInfo(gpuRMSE_kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, nullptr));
        printf("[VERBOSE] local_work_size for gpuRMSE_kernel should be: %zu\n",local);
    }

    double t_update_ratings_acc = 0;
    double t_rank_one_update_acc = 0;

    float* rmseVec = (float*) malloc((T.nnz) * sizeof(float));

    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "[INFO] Computing cdmf OpenCL..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int oiter = 1; oiter <= param.maxiter; ++oiter) {

        double t_update_ratings = 0;
        double t_rank_one_update = 0;

        for (unsigned t = 0; t < param.k; ++t) {
            // Writing Buffer
            Wt = &(W_c[t][0]); // u
            Ht = &(H_c[t][0]); // v
            CL_CHECK(clEnqueueWriteBuffer(commandQueue, WtBuffer, CL_TRUE, 0, R.rows * sizeof(VALUE_TYPE), Wt, 0, nullptr, nullptr));
            CL_CHECK(clEnqueueWriteBuffer(commandQueue, HtBuffer, CL_TRUE, 0, R.cols * sizeof(VALUE_TYPE), Ht, 0, nullptr, nullptr));

            if (oiter > 1) {
                // update the rating matrix in CSC format (+)
                cl_event eventPoint0c, eventPoint0r;
                CL_CHECK(clEnqueueNDRangeKernel(commandQueue, UpdateRating_DUAL_kernel_NoLoss_c, 1, nullptr, global_work_size, local_work_size, 0, nullptr, &eventPoint0c));
                CL_CHECK(clWaitForEvents(1, &eventPoint0c));

                t_update_ratings += executionTime(eventPoint0c);

                // update the rating matrix in CSR format (+)
                CL_CHECK(clEnqueueNDRangeKernel(commandQueue, UpdateRating_DUAL_kernel_NoLoss_r, 1, nullptr, global_work_size, local_work_size, 0, nullptr, &eventPoint0r));
                CL_CHECK(clWaitForEvents(1, &eventPoint0r));

                t_update_ratings += executionTime(eventPoint0r);

                CL_CHECK(clReleaseEvent(eventPoint0c));
                CL_CHECK(clReleaseEvent(eventPoint0r));
            }

            for (int iter = 1; iter <= param.maxinneriter; ++iter) {
                // update vector v
                cl_event eventPoint1v, eventPoint1u;
                CL_CHECK(clEnqueueNDRangeKernel(commandQueue, RankOneUpdate_DUAL_kernel_v, 1, nullptr, global_work_size, local_work_size, 0, nullptr, &eventPoint1v));
                CL_CHECK(clWaitForEvents(1, &eventPoint1v));

                t_rank_one_update += executionTime(eventPoint1v);

                // update vector u
                CL_CHECK(clEnqueueNDRangeKernel(commandQueue, RankOneUpdate_DUAL_kernel_u, 1, nullptr, global_work_size, local_work_size, 0, nullptr, &eventPoint1u));
                CL_CHECK(clWaitForEvents(1, &eventPoint1u));

                t_rank_one_update += executionTime(eventPoint1u);

                CL_CHECK(clReleaseEvent(eventPoint1v));
                CL_CHECK(clReleaseEvent(eventPoint1u));
            }
            // Reading Buffer
            CL_CHECK(clEnqueueReadBuffer(commandQueue, WtBuffer, CL_TRUE, 0, R.rows * sizeof(VALUE_TYPE), Wt, 0, nullptr, nullptr));
            CL_CHECK(clEnqueueReadBuffer(commandQueue, HtBuffer, CL_TRUE, 0, R.cols * sizeof(VALUE_TYPE), Ht, 0, nullptr, nullptr));

            // update the rating matrix in CSC format (-)
            cl_event eventPoint2c, eventPoint2r;
            CL_CHECK(clEnqueueNDRangeKernel(commandQueue, UpdateRating_DUAL_kernel_NoLoss_c_, 1, nullptr, global_work_size, local_work_size, 0, nullptr, &eventPoint2c));
            CL_CHECK(clWaitForEvents(1, &eventPoint2c));

            t_update_ratings += executionTime(eventPoint2c);

            // update the rating matrix in CSR format (-)
            CL_CHECK(clEnqueueNDRangeKernel(commandQueue, UpdateRating_DUAL_kernel_NoLoss_r_, 1, nullptr, global_work_size, local_work_size, 0, nullptr, &eventPoint2r));
            CL_CHECK(clWaitForEvents(1, &eventPoint2r));

            t_update_ratings += executionTime(eventPoint2r);

            CL_CHECK(clReleaseEvent(eventPoint2c));
            CL_CHECK(clReleaseEvent(eventPoint2r));
        }

        t_update_ratings_acc += t_update_ratings;
        t_rank_one_update_acc += t_rank_one_update;

        /** Calculate RMSE*/
        cl_event eventPoint3;

        CL_CHECK(clEnqueueCopyBuffer(commandQueue, emptyBuffer, rmseBuffer, 0, 0, (T.nnz) * sizeof(float), 0, nullptr, &eventPoint3));
        CL_CHECK(clEnqueueCopyBuffer(commandQueue, emptyBuffer, pred_vBuffer, 0, 0, (T.nnz) * sizeof(float), 0, nullptr, &eventPoint3));

//        size_t gws_rmse[1] = {((T.nnz + 1023) / 1024) * 1024};
//        size_t lws_rmse[1] = {1024};
//        CL_CHECK(clEnqueueNDRangeKernel(commandQueue, gpuRMSE_kernel, 1, nullptr, gws_rmse, lws_rmse, 0, nullptr, &eventPoint3));
        CL_CHECK(clEnqueueNDRangeKernel(commandQueue, gpuRMSE_kernel, 1, nullptr, global_work_size, local_work_size, 0, nullptr, &eventPoint3));

        CL_CHECK(clWaitForEvents(1, &eventPoint3));

        double rmse_time = executionTime(eventPoint3);

        CL_CHECK(clReleaseEvent(eventPoint3));

        CL_CHECK(clEnqueueReadBuffer(commandQueue, rmseBuffer, CL_TRUE, 0, (T.nnz) * sizeof(float), rmseVec, 0, nullptr, nullptr));

        double tot_rmse = 0;
        double f_rmse = 0;

        for (unsigned i = 0; i < T.nnz; ++i) {
            tot_rmse += rmseVec[i];
        }
        f_rmse = sqrt(tot_rmse / T.nnz);

        printf("[-INFO-] iteration num %d \trank_time %.4lf|%.4lf s \tupdate_time %.4lf|%.4lfs \tRMSE=%f time:%fs\n",
                oiter, t_rank_one_update, t_rank_one_update_acc, t_update_ratings, t_update_ratings_acc, f_rmse, rmse_time);

    }
    auto t2 = std::chrono::high_resolution_clock::now();
    deltaT12 = t2 - t1;
    printf("[INFO] OCL Training time: %lf s\n", deltaT12.count());

    CL_CHECK(clReleaseKernel(UpdateRating_DUAL_kernel_NoLoss_c));
    CL_CHECK(clReleaseKernel(UpdateRating_DUAL_kernel_NoLoss_r));
    CL_CHECK(clReleaseKernel(UpdateRating_DUAL_kernel_NoLoss_c_));
    CL_CHECK(clReleaseKernel(UpdateRating_DUAL_kernel_NoLoss_r_));
    CL_CHECK(clReleaseKernel(RankOneUpdate_DUAL_kernel_u));
    CL_CHECK(clReleaseKernel(RankOneUpdate_DUAL_kernel_v));
    CL_CHECK(clReleaseProgram(program));
    CL_CHECK(clReleaseMemObject(row_ptrBuffer));
    CL_CHECK(clReleaseMemObject(col_idxBuffer));
    CL_CHECK(clReleaseMemObject(col_ptrBuffer));
    CL_CHECK(clReleaseMemObject(row_idxBuffer));
    CL_CHECK(clReleaseMemObject(valBuffer));
    CL_CHECK(clReleaseMemObject(val_tBuffer));
    CL_CHECK(clReleaseMemObject(WtBuffer));
    CL_CHECK(clReleaseMemObject(HtBuffer));
    CL_CHECK(clReleaseCommandQueue(commandQueue));
    CL_CHECK(clReleaseContext(context));
    CL_CHECK(clReleaseDevice(devices[0]));
    free(devices);
}
