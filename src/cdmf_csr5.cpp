#include "tools.h"
#include "anonymouslib_opencl.h"

extern std::chrono::duration<double> deltaT12;
extern std::chrono::duration<double> deltaTAB;

void cdmf_csr5(SparseMatrix& R, MatData& W_c, MatData& H_c, parameter& param, char filename[]) {
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
    snprintf(options, sizeof(options), "-DWG_SIZE=%d -DVALUE_TYPE=%s", param.nThreadsPerBlock, getT(sizeof(VALUE_TYPE)));
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

    for (unsigned t = 0; t < param.k; ++t) {
        for (long c = 0; c < R.cols; ++c) {
            H_c[t][c] = 0;
        }
    }

//    unsigned m = R.rows;
//    unsigned n = R.cols;
//
//    double gb = getB<int, VALUE_TYPE>(m, R.nnz);
//    double gflop = getFLOP<int>(R.nnz);

    VALUE_TYPE* Wt = (VALUE_TYPE*) malloc(R.rows * sizeof(VALUE_TYPE));
    VALUE_TYPE* Ht = (VALUE_TYPE*) malloc(R.cols * sizeof(VALUE_TYPE));
    memset(Ht, 0, R.cols * sizeof(VALUE_TYPE));
    memset(Wt, 0, R.rows * sizeof(VALUE_TYPE));

    // buffers to store the bottom results
    VALUE_TYPE* Hb = (VALUE_TYPE*) malloc(R.cols * sizeof(VALUE_TYPE));
    VALUE_TYPE* Wb = (VALUE_TYPE*) malloc(R.rows * sizeof(VALUE_TYPE));
    memset(Hb, 0, R.cols * sizeof(VALUE_TYPE));
    memset(Wb, 0, R.rows * sizeof(VALUE_TYPE));

    size_t nbits_u = R.rows * sizeof(VALUE_TYPE);
    size_t nbits_v = R.cols * sizeof(VALUE_TYPE);

    size_t nbits_col_ptr = (R.cols + 1) * sizeof(long);
    size_t nbits_row_ptr = (R.rows + 1) * sizeof(long);

    size_t nbits_idx = R.nnz * sizeof(long);

    size_t nbits_val = R.nnz * sizeof(VALUE_TYPE);


    // creating buffers
    cl_mem row_ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_row_ptr, R.get_csr_row_ptr(), &err);
    CL_CHECK(err);
    cl_mem col_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_idx, R.get_csr_col_indx(), &err);
    CL_CHECK(err);
    cl_mem col_ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_col_ptr, R.get_csc_col_ptr(), &err);
    CL_CHECK(err);
    cl_mem row_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_idx, R.get_csc_row_indx(), &err);
    CL_CHECK(err);
    cl_mem valBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_val, R.get_csc_val(), &err);
    CL_CHECK(err);
    cl_mem val_tBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_val, R.get_csr_val(), &err);
    CL_CHECK(err);
    cl_mem WBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_u, Wt, &err); // u
    CL_CHECK(err);
    cl_mem WtBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_u, Wt, &err); // u
    CL_CHECK(err);
    cl_mem WbBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_u, Wb, &err); // u
    CL_CHECK(err);
    cl_mem HBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_v, Ht, &err);  // v
    CL_CHECK(err);
    cl_mem HtBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_v, Ht, &err); // v
    CL_CHECK(err);
    cl_mem HbBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_v, Hb, &err); // v
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
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 0, sizeof(unsigned), &R.cols));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 1, sizeof(cl_mem), &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 2, sizeof(cl_mem), &row_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 3, sizeof(cl_mem), &valBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 4, sizeof(cl_mem), &WBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 5, sizeof(cl_mem), &HBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 6, sizeof(unsigned), &R.rows));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 7, sizeof(cl_mem), &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 8, sizeof(cl_mem), &col_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r, 9, sizeof(cl_mem), &val_tBuffer));

    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 0, sizeof(unsigned), &R.cols));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 1, sizeof(cl_mem), &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 2, sizeof(cl_mem), &row_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 3, sizeof(cl_mem), &valBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 4, sizeof(cl_mem), &WBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 5, sizeof(cl_mem), &HBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 6, sizeof(unsigned), &R.rows));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 7, sizeof(cl_mem), &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 8, sizeof(cl_mem), &col_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c, 9, sizeof(cl_mem), &val_tBuffer));

    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 0, sizeof(unsigned), &R.cols));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 1, sizeof(cl_mem), &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 2, sizeof(cl_mem), &row_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 3, sizeof(cl_mem), &valBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 4, sizeof(cl_mem), &WBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 5, sizeof(cl_mem), &HBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 6, sizeof(unsigned), &R.rows));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 7, sizeof(cl_mem), &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 8, sizeof(cl_mem), &col_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_r_, 9, sizeof(cl_mem), &val_tBuffer));

    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 0, sizeof(unsigned), &R.cols));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 1, sizeof(cl_mem), &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 2, sizeof(cl_mem), &row_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 3, sizeof(cl_mem), &valBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 4, sizeof(cl_mem), &WBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 5, sizeof(cl_mem), &HBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 6, sizeof(unsigned), &R.rows));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 7, sizeof(cl_mem), &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 8, sizeof(cl_mem), &col_idxBuffer));
    CL_CHECK(clSetKernelArg(UpdateRating_DUAL_kernel_NoLoss_c_, 9, sizeof(cl_mem), &val_tBuffer));

    CL_CHECK(clSetKernelArg(_kernel_CALV, 0, sizeof(unsigned), &R.cols));
    CL_CHECK(clSetKernelArg(_kernel_CALV, 1, sizeof(cl_mem), &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(_kernel_CALV, 2, sizeof(cl_mem), &HtBuffer));
    CL_CHECK(clSetKernelArg(_kernel_CALV, 3, sizeof(cl_mem), &HbBuffer));
    CL_CHECK(clSetKernelArg(_kernel_CALV, 4, sizeof(cl_mem), &HBuffer));
    CL_CHECK(clSetKernelArg(_kernel_CALV, 5, sizeof(VALUE_TYPE), &param.lambda));

    CL_CHECK(clSetKernelArg(_kernel_CALU, 0, sizeof(unsigned), &R.rows));
    CL_CHECK(clSetKernelArg(_kernel_CALU, 1, sizeof(cl_mem), &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(_kernel_CALU, 2, sizeof(cl_mem), &WtBuffer));
    CL_CHECK(clSetKernelArg(_kernel_CALU, 3, sizeof(cl_mem), &WbBuffer));
    CL_CHECK(clSetKernelArg(_kernel_CALU, 4, sizeof(cl_mem), &WBuffer));
    CL_CHECK(clSetKernelArg(_kernel_CALU, 5, sizeof(VALUE_TYPE), &param.lambda));

    size_t gws_row[1] = {static_cast<size_t>(R.rows * param.nThreadsPerBlock)};
    size_t gws_col[1] = {static_cast<size_t>(R.cols * param.nThreadsPerBlock)};
    size_t local_work_size[1] = {static_cast<size_t>(param.nThreadsPerBlock)};
    printf("[info] - blocks: %d | threads per block: %d | GWS_ROW: %zu | GWS_COL: %zu | local_work_size: %zu !\n",
           param.nBlocks, param.nThreadsPerBlock, gws_row[0], gws_col[0], local_work_size[0]);

    double time = 0;
    double final_time = 0;
    double convert_time = 0;
    anonymouslibHandle<int, unsigned int, VALUE_TYPE> Av(R.cols, R.rows);
    CL_CHECK(Av.setOCLENV(context, commandQueue, devices));
    CL_CHECK(Av.inputCSR(R.nnz, col_ptrBuffer, row_idxBuffer, valBuffer));
    CL_CHECK(Av.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA));
    anonymouslib_timer asCSR5_timer;
    asCSR5_timer.start();
    CL_CHECK(Av.asCSR5());
    CL_CHECK(clFinish(commandQueue));
    std::cout << "Av: CSR->CSR5 time = " << asCSR5_timer.stop() << " s." << std::endl;

    anonymouslibHandle<int, unsigned int, VALUE_TYPE> Au(R.rows, R.cols);
    CL_CHECK(Au.setOCLENV(context, commandQueue, devices));
    CL_CHECK(Au.inputCSR(R.nnz, row_ptrBuffer, col_idxBuffer, val_tBuffer));
    CL_CHECK(Au.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA));
    asCSR5_timer.start();
    CL_CHECK(Au.asCSR5());
    CL_CHECK(clFinish(commandQueue));
    std::cout << "Au: CSR->CSR5 time = " << asCSR5_timer.stop() << " s." << std::endl;

    CL_CHECK(Av.setX(WBuffer)); // you only need to do it once!
    CL_CHECK(Au.setX(HBuffer)); // you only need to do it once!

    double t_update_ratings_acc = 0;
    double t_rank_one_update_acc = 0;

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
            CL_CHECK(clEnqueueWriteBuffer(commandQueue, WBuffer, CL_TRUE, 0, R.rows * sizeof(VALUE_TYPE), Wt, 0, nullptr, nullptr));
            CL_CHECK(clEnqueueWriteBuffer(commandQueue, HBuffer, CL_TRUE, 0, R.cols * sizeof(VALUE_TYPE), Ht, 0, nullptr, nullptr));

            if (oiter > 1) {
                Av.asCSR_(&time);
                convert_time+=time;

                Au.asCSR_(&time);
                convert_time+=time;

                // update the rating matrix in CSC format (+)
                cl_event eventPoint;
                CL_CHECK(clEnqueueNDRangeKernel(commandQueue, UpdateRating_DUAL_kernel_NoLoss_c, 1, nullptr, gws_col, local_work_size, 0, nullptr, &eventPoint));
                CL_CHECK(clWaitForEvents(1, &eventPoint));

                t_update_ratings += executionTime(eventPoint);

                // update the rating matrix in CSR format (+)
                CL_CHECK(clEnqueueNDRangeKernel(commandQueue, UpdateRating_DUAL_kernel_NoLoss_r, 1, nullptr, gws_row, local_work_size, 0, nullptr, &eventPoint));
                CL_CHECK(clWaitForEvents(1, &eventPoint));

                t_update_ratings += executionTime(eventPoint);

                CL_CHECK(clReleaseEvent(eventPoint));

                Av.asCSR5_(&time);
                convert_time+=time;

                Au.asCSR5_(&time);
                convert_time+=time;

            }
            for (int iter = 1; iter <= param.maxinneriter; ++iter) {
                cl_event eventPoint1v, eventPoint1u;

                // update vector v
                CL_CHECK(Av.spmv(1.0, HtBuffer, HbBuffer, &time));
                CL_CHECK(clEnqueueNDRangeKernel(commandQueue, _kernel_CALV, 1, nullptr, gws_col, local_work_size, 0, nullptr, &eventPoint1v));
                CL_CHECK(clWaitForEvents(1, &eventPoint1v));

                t_rank_one_update += executionTime(eventPoint1v) + time;
                final_time+=time;

                // update vector u
                CL_CHECK(Au.spmv(1.0, WtBuffer, WbBuffer, &time));
                CL_CHECK(clEnqueueNDRangeKernel(commandQueue, _kernel_CALU, 1, nullptr, gws_row, local_work_size, 0, nullptr, &eventPoint1u));
                CL_CHECK(clWaitForEvents(1, &eventPoint1u));

                t_rank_one_update += executionTime(eventPoint1u) + time;
                final_time+=time;

                CL_CHECK(clReleaseEvent(eventPoint1v));
                CL_CHECK(clReleaseEvent(eventPoint1u));
            }
            // Reading Buffer
            CL_CHECK(clEnqueueReadBuffer(commandQueue, WBuffer, CL_TRUE, 0, R.rows * sizeof(VALUE_TYPE), Wt, 0, nullptr, nullptr));
            CL_CHECK(clEnqueueReadBuffer(commandQueue, HBuffer, CL_TRUE, 0, R.cols * sizeof(VALUE_TYPE), Ht, 0, nullptr, nullptr));
            Av.asCSR_(&time);
            convert_time+=time;

            Au.asCSR_(&time);
            convert_time+=time;

            // update the rating matrix in CSC format (-)
            cl_event eventPoint2c, eventPoint2r;
            CL_CHECK(clEnqueueNDRangeKernel(commandQueue, UpdateRating_DUAL_kernel_NoLoss_c_, 1, nullptr, gws_col, local_work_size, 0, nullptr, &eventPoint2c));
            CL_CHECK(clWaitForEvents(1, &eventPoint2c));

            t_update_ratings += executionTime(eventPoint2c);

            // update the rating matrix in CSR format (-)
            CL_CHECK(clEnqueueNDRangeKernel(commandQueue, UpdateRating_DUAL_kernel_NoLoss_r_, 1, nullptr, gws_row, local_work_size, 0, nullptr, &eventPoint2r));
            CL_CHECK(clWaitForEvents(1, &eventPoint2r));

            t_update_ratings += executionTime(eventPoint2r);

            CL_CHECK(clReleaseEvent(eventPoint2c));
            CL_CHECK(clReleaseEvent(eventPoint2r));

            Av.asCSR5_(&time);
            final_time+=time;

            Au.asCSR5_(&time);
            final_time+=time;

        }

        t_update_ratings_acc += t_update_ratings;
        t_rank_one_update_acc += t_rank_one_update;

        if (param.verbose) {
            printf("[VERBOSE] outter iteration num %d \trank_time %.4lf|%.4lf s \tupdate_time %.4lf|%.4lf s \n",
                   oiter, t_rank_one_update, t_rank_one_update_acc, t_update_ratings, t_update_ratings_acc);
        }

    }
    auto t2 = std::chrono::high_resolution_clock::now();
    deltaT12 = t2 - t1;
    printf("[INFO] OCL Training time: %lf s\n", deltaT12.count());
    printf("[INFO] spmv time: %lf s | conversion time %lf s\n", final_time, convert_time);

    CL_CHECK(Au.destroy());
    CL_CHECK(Av.destroy());
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
    free(devices);
    free(Wb);
    free(Hb);
}
