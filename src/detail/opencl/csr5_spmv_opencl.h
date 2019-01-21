#ifndef CSR5_SPMV_OPENCL_H
#define CSR5_SPMV_OPENCL_H

#include "utils_opencl.h"

template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT, typename ANONYMOUSLIB_VT>
void csr5_spmv(cl_kernel ocl_kernel_spmv_csr5_compute,
              cl_kernel ocl_kernel_spmv_csr5_calibrate,
              cl_kernel ocl_kernel_spmv_csr5_tail_partition,
              cl_command_queue ocl_command_queue,
              const int sigma,
              const ANONYMOUSLIB_IT p,
              const ANONYMOUSLIB_IT m,
              const int bit_y_offset,
              const int bit_scansum_offset,
              const int num_packet,
              const cl_mem row_pointer,
              const cl_mem column_index,
              const cl_mem value,
              const cl_mem partition_pointer,
              const cl_mem partition_descriptor,
              const cl_mem partition_descriptor_offset_pointer,
              const cl_mem partition_descriptor_offset,
              cl_mem calibrator_t,
              cl_mem calibrator_b,
              const ANONYMOUSLIB_IT tail_partition_start,
              const ANONYMOUSLIB_VT alpha,
              const cl_mem x,
              cl_mem y_t,
              cl_mem y_b,
              double* time) {
    double spmv_time = 0;

    BasicCL basicCL;
    cl_event ceTimer;
    cl_ulong queuedTime;
    cl_ulong submitTime;
    cl_ulong startTime;
    cl_ulong endTime;
    cl_int err;

    int num_threads;
    int num_blocks;
    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];


    num_threads = ANONYMOUSLIB_THREAD_GROUP;
    num_blocks = (int) ceil((double) (p - 1) / (double) (num_threads / ANONYMOUSLIB_CSR5_OMEGA));
    szLocalWorkSize[0] = (size_t) num_threads;
    szGlobalWorkSize[0] = (size_t) num_blocks * szLocalWorkSize[0];

    err = clSetKernelArg(ocl_kernel_spmv_csr5_compute, 0, sizeof(cl_mem), (void*) &column_index);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_compute, 1, sizeof(cl_mem), (void*) &value);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_compute, 2, sizeof(cl_mem), (void*) &row_pointer);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_compute, 3, sizeof(cl_mem), (void*) &x);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_compute, 4, sizeof(cl_mem), (void*) &partition_pointer);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_compute, 5, sizeof(cl_mem), (void*) &partition_descriptor);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_compute, 6, sizeof(cl_mem), (void*) &partition_descriptor_offset_pointer);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_compute, 7, sizeof(cl_mem), (void*) &partition_descriptor_offset);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_compute, 8, sizeof(cl_mem), (void*) &calibrator_t);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_compute, 9, sizeof(cl_mem), (void*) &calibrator_b);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_compute, 10, sizeof(cl_mem), (void*) &y_t);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_compute, 11, sizeof(cl_mem), (void*) &y_b);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_compute, 12, sizeof(ANONYMOUSLIB_IT), (void*) &p);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_compute, 13, sizeof(cl_int), (void*) &num_packet);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_compute, 14, sizeof(cl_int), (void*) &bit_y_offset);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_compute, 15, sizeof(cl_int), (void*) &bit_scansum_offset);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_compute, 16, sizeof(ANONYMOUSLIB_VT), (void*) &alpha);
    CL_CHECK(err);

    err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_spmv_csr5_compute, 1, nullptr,
                                 szGlobalWorkSize, szLocalWorkSize, 0, nullptr, &ceTimer);
    CL_CHECK(err);

    err = clWaitForEvents(1, &ceTimer);
    CL_CHECK(err);

    basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
    spmv_time +=  double(endTime - startTime) * 1.0e-9;



    num_threads = ANONYMOUSLIB_THREAD_GROUP;
    num_blocks = (int) ceil((double) (p - 1) / (double) num_threads);
    szLocalWorkSize[0] = (size_t) num_threads;
    szGlobalWorkSize[0] = (size_t) num_blocks * szLocalWorkSize[0];

    err = clSetKernelArg(ocl_kernel_spmv_csr5_calibrate, 0, sizeof(cl_mem), (void*) &partition_pointer);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_calibrate, 1, sizeof(cl_mem), (void*) &calibrator_t);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_calibrate, 2, sizeof(cl_mem), (void*) &calibrator_b);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_calibrate, 3, sizeof(cl_mem), (void*) &y_t);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_calibrate, 4, sizeof(cl_mem), (void*) &y_b);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_calibrate, 5, sizeof(cl_int), (void*) &p);
    CL_CHECK(err);

    err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_spmv_csr5_calibrate, 1, nullptr,
                                 szGlobalWorkSize, szLocalWorkSize, 0, nullptr, &ceTimer);
    CL_CHECK(err);

    err = clWaitForEvents(1, &ceTimer);
    CL_CHECK(err);

    basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
    spmv_time += double(endTime - startTime) * 1.0e-9;



    num_threads = ANONYMOUSLIB_CSR5_OMEGA;
    num_blocks = m - tail_partition_start;
    szLocalWorkSize[0] = (size_t) num_threads;
    szGlobalWorkSize[0] = (size_t) num_blocks * szLocalWorkSize[0];

    err = clSetKernelArg(ocl_kernel_spmv_csr5_tail_partition, 0, sizeof(cl_mem), (void*) &row_pointer);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_tail_partition, 1, sizeof(cl_mem), (void*) &column_index);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_tail_partition, 2, sizeof(cl_mem), (void*) &value);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_tail_partition, 3, sizeof(cl_mem), (void*) &x);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_tail_partition, 4, sizeof(cl_mem), (void*) &y_t);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_tail_partition, 5, sizeof(cl_mem), (void*) &y_b);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_tail_partition, 6, sizeof(cl_int), (void*) &tail_partition_start);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_tail_partition, 7, sizeof(cl_int), (void*) &p);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_tail_partition, 8, sizeof(cl_int), (void*) &sigma);
    CL_CHECK(err);
    err = clSetKernelArg(ocl_kernel_spmv_csr5_tail_partition, 9, sizeof(ANONYMOUSLIB_VT), (void*) &alpha);
    CL_CHECK(err);

    err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_spmv_csr5_tail_partition, 1, nullptr,
                                 szGlobalWorkSize, szLocalWorkSize, 0, nullptr, &ceTimer);
    CL_CHECK(err);

    err = clWaitForEvents(1, &ceTimer);
    CL_CHECK(err);

    basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
    spmv_time += double(endTime - startTime) * 1.0e-9;


    *time = spmv_time;
}

#endif // CSR5_SPMV_OPENCL_H
