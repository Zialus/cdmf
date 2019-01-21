#ifndef ANONYMOUSLIB_OPENCL_H
#define ANONYMOUSLIB_OPENCL_H

#include "detail/utils.h"
#include "detail/opencl/utils_opencl.h"

#include "detail/opencl/common_opencl.h"
#include "detail/opencl/format_opencl.h"
#include "detail/opencl/csr5_spmv_opencl.h"

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
class anonymouslibHandle
{
    public:
        anonymouslibHandle(ANONYMOUSLIB_IT m, ANONYMOUSLIB_IT n);
        int setOCLENV(cl_context _ocl_context, cl_command_queue _ocl_command_queue, cl_device_id *_ocl_device);
        int warmup();
        int inputCSR(ANONYMOUSLIB_IT  nnz, cl_mem csr_row_pointer, cl_mem csr_column_index, cl_mem csr_value);
        int asCSR();
        int asCSR_(double* time);
        int asCSR5();
        int asCSR5_(double *time);
        int setX(cl_mem x);
        int spmv(const ANONYMOUSLIB_VT alpha, cl_mem y, cl_mem yb, double *time);
        int destroy();
        int setSigma(int sigma);

    private:
        cl_context          _ocl_context;
        cl_command_queue    _ocl_command_queue;
        cl_device_id       *_ocl_device;

        cl_program          _ocl_program_format;
        cl_kernel           _ocl_kernel_warmup;
        cl_kernel           _ocl_kernel_generate_partition_pointer_s1;
        cl_kernel           _ocl_kernel_generate_partition_pointer_s2;
        cl_kernel           _ocl_kernel_generate_partition_descriptor_s0;
        cl_kernel           _ocl_kernel_generate_partition_descriptor_s1;
        cl_kernel           _ocl_kernel_generate_partition_descriptor_s2;
        cl_kernel           _ocl_kernel_generate_partition_descriptor_s3;
        cl_kernel           _ocl_kernel_generate_partition_descriptor_offset;
        cl_kernel           _ocl_kernel_aosoa_transpose_smem_iT;
        cl_kernel           _ocl_kernel_aosoa_transpose_smem_vT;

        cl_program          _ocl_program_csr5_spmv;
        cl_kernel           _ocl_kernel_spmv_csr5_compute;
        cl_kernel           _ocl_kernel_spmv_csr5_calibrate;
        cl_kernel           _ocl_kernel_spmv_csr5_tail_partition;

        std::string _ocl_source_code_string_format_const;
        std::string _ocl_source_code_string_csr5_spmv_const;

        int computeSigma();
        int _format;
        ANONYMOUSLIB_IT _m;
        ANONYMOUSLIB_IT _n;
        ANONYMOUSLIB_IT _nnz;

        cl_mem _csr_row_pointer;
        cl_mem _csr_column_index;
        cl_mem _csr_value;

        int         _csr5_sigma;
        int         _bit_y_offset;
        int         _bit_scansum_offset;
        int         _num_packet;
        ANONYMOUSLIB_IT _tail_partition_start;

        ANONYMOUSLIB_IT _p;
        cl_mem _csr5_partition_pointer;
        cl_mem _csr5_partition_descriptor;

        ANONYMOUSLIB_IT   _num_offsets;
        cl_mem  _csr5_partition_descriptor_offset_pointer;
        cl_mem  _csr5_partition_descriptor_offset;
        cl_mem  _temp_calibrator_t;
        cl_mem  _temp_calibrator_b;

        cl_mem         _x;
};

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::anonymouslibHandle(ANONYMOUSLIB_IT m,
                                                                                           ANONYMOUSLIB_IT n) {
    _m = m;
    _n = n;
}

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::warmup() {
    format_warmup(_ocl_kernel_warmup, _ocl_context, _ocl_command_queue);

    return ANONYMOUSLIB_SUCCESS;
}

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::inputCSR(ANONYMOUSLIB_IT nnz,
                                                                                     cl_mem csr_row_pointer,
                                                                                     cl_mem csr_column_index,
                                                                                     cl_mem csr_value) {
    _format = ANONYMOUSLIB_FORMAT_CSR;

    _nnz = nnz;

    _csr_row_pointer = csr_row_pointer;
    _csr_column_index = csr_column_index;
    _csr_value = csr_value;

    return ANONYMOUSLIB_SUCCESS;
}

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::asCSR() {
    int err = ANONYMOUSLIB_SUCCESS;

    if (_format == ANONYMOUSLIB_FORMAT_CSR) {
        return err;
    }

    if (_format == ANONYMOUSLIB_FORMAT_CSR5) { // convert csr5 data to csr data
        double time = 0;
        err = aosoa_transpose<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>(_ocl_kernel_aosoa_transpose_smem_iT,
                                                                                  _ocl_kernel_aosoa_transpose_smem_vT,
                                                                                  _ocl_command_queue, _csr5_sigma, _nnz,
                                                                                  _csr5_partition_pointer,
                                                                                  _csr_column_index, _csr_value, 0,
                                                                                  &time);
        CL_CHECK(err);

        // free the two newly added CSR5 arrays
        if (_csr5_partition_pointer) { err = clReleaseMemObject(_csr5_partition_pointer); CL_CHECK(err); }
        if (_csr5_partition_descriptor) { err = clReleaseMemObject(_csr5_partition_descriptor); CL_CHECK(err); }
        if (_temp_calibrator_t) { err = clReleaseMemObject(_temp_calibrator_t); CL_CHECK(err); }
        if (_temp_calibrator_b) { err = clReleaseMemObject(_temp_calibrator_b); CL_CHECK(err); }
        if (_csr5_partition_descriptor_offset_pointer) { err = clReleaseMemObject(_csr5_partition_descriptor_offset_pointer); CL_CHECK(err); }
        if (_csr5_partition_descriptor_offset) { err = clReleaseMemObject(_csr5_partition_descriptor_offset); CL_CHECK(err); }

        _format = ANONYMOUSLIB_FORMAT_CSR;
    }

    return err;
}

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::asCSR_(double* time) {
    int err = ANONYMOUSLIB_SUCCESS;

    if (_format == ANONYMOUSLIB_FORMAT_CSR) {
        return err;
    }

    if (_format == ANONYMOUSLIB_FORMAT_CSR5) {
        // convert csr5 data to csr data
        err = aosoa_transpose<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>(_ocl_kernel_aosoa_transpose_smem_iT,
                                                                                  _ocl_kernel_aosoa_transpose_smem_vT,
                                                                                  _ocl_command_queue, _csr5_sigma, _nnz,
                                                                                  _csr5_partition_pointer,
                                                                                  _csr_column_index, _csr_value, 0,
                                                                                  time);
        CL_CHECK(err);

        _format = ANONYMOUSLIB_FORMAT_CSR;
    }

    return err;
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::asCSR5()
{
    int err = ANONYMOUSLIB_SUCCESS;

    if (_format == ANONYMOUSLIB_FORMAT_CSR5)
        return err;

    if (_format == ANONYMOUSLIB_FORMAT_CSR)
    {
        double malloc_time = 0, tile_ptr_time = 0, tile_desc_time = 0, transpose_time = 0;
        anonymouslib_timer malloc_timer;
        // anonymouslib_timer tile_ptr_timer, tile_desc_timer, transpose_timer;
        double time = 0;

        // compute sigma
        _csr5_sigma = computeSigma();
        std::cout << "omega = " << ANONYMOUSLIB_CSR5_OMEGA << ", sigma = " << _csr5_sigma << ". ";

        // compute how many bits required for `y_offset' and `carry_offset'
        int base = 2;
        _bit_y_offset = 1;
        while (base < ANONYMOUSLIB_CSR5_OMEGA * _csr5_sigma) { base *= 2; _bit_y_offset++; }

        base = 2;
        _bit_scansum_offset = 1;
        while (base < ANONYMOUSLIB_CSR5_OMEGA) { base *= 2; _bit_scansum_offset++; }

        if (_bit_y_offset + _bit_scansum_offset > (int) sizeof(ANONYMOUSLIB_UIT) * 8 - 1) //the 1st bit of bit-flag should be in the first packet
            return ANONYMOUSLIB_UNSUPPORTED_CSR5_OMEGA;

        int bit_all = _bit_y_offset + _bit_scansum_offset + _csr5_sigma;
        _num_packet = (int) ceil((double) bit_all / (double) (sizeof(ANONYMOUSLIB_UIT) * 8));
        //std::cout << "#num_packet = " << _num_packet << std::endl;

        // calculate the number of partitions
        _p = (int) ceil((double) _nnz / (double) (ANONYMOUSLIB_CSR5_OMEGA * _csr5_sigma));
        //std::cout << "#partition = " << _p << std::endl;

        malloc_timer.start();
        // malloc the newly added arrays for CSR5
        _csr5_partition_pointer = clCreateBuffer(_ocl_context, CL_MEM_READ_WRITE, (_p + 1) * sizeof(ANONYMOUSLIB_UIT), NULL, &err);
        if(err != CL_SUCCESS) return err;

        _csr5_partition_descriptor = clCreateBuffer(_ocl_context, CL_MEM_READ_WRITE, _p * ANONYMOUSLIB_CSR5_OMEGA * _num_packet * sizeof(ANONYMOUSLIB_UIT), NULL, &err);
        if(err != CL_SUCCESS) return err;

        _temp_calibrator_t = clCreateBuffer(_ocl_context, CL_MEM_READ_WRITE, _p * sizeof(ANONYMOUSLIB_VT), NULL, &err);
        if(err != CL_SUCCESS) return err;
        _temp_calibrator_b = clCreateBuffer(_ocl_context, CL_MEM_READ_WRITE, _p * sizeof(ANONYMOUSLIB_VT), NULL, &err);
        if(err != CL_SUCCESS) return err;

        _csr5_partition_descriptor_offset_pointer = clCreateBuffer(_ocl_context, CL_MEM_READ_WRITE, (_p + 1) * sizeof(ANONYMOUSLIB_IT), NULL, &err);
        if(err != CL_SUCCESS) return err;
        err = clFinish(_ocl_command_queue);
        malloc_time += malloc_timer.stop();

        // convert csr data to csr5 data (3 steps)
        // step 1. generate partition pointer
        //tile_ptr_timer.start();
        err = generate_partition_pointer<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>
            (_ocl_kernel_generate_partition_pointer_s1, _ocl_kernel_generate_partition_pointer_s2, _ocl_command_queue,
             _csr5_sigma, _p, _m, _nnz, _csr5_partition_pointer, _csr_row_pointer, &time);
        if (err != ANONYMOUSLIB_SUCCESS)
            return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
        //err = clFinish(_ocl_command_queue);
        //tile_ptr_time += tile_ptr_timer.stop();
        tile_ptr_time += time;

        malloc_timer.start();
        ANONYMOUSLIB_UIT tail;

        err = clEnqueueReadBuffer(_ocl_command_queue, _csr5_partition_pointer, CL_TRUE,
                (_p-1) * sizeof(ANONYMOUSLIB_UIT), sizeof(ANONYMOUSLIB_UIT), &tail, 0, NULL, NULL);
        if(err != CL_SUCCESS) return err;
        err = clFinish(_ocl_command_queue);
        malloc_time += malloc_timer.stop();

        _tail_partition_start = (tail << 1) >> 1;
        //std::cout << "_tail_partition_start = " << _tail_partition_start << std::endl;

        // step 2. generate partition descriptor
        //tile_desc_timer.start();
        _num_offsets = 0;
        err = generate_partition_descriptor<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>
            (_ocl_kernel_generate_partition_descriptor_s0,
             _ocl_kernel_generate_partition_descriptor_s1,
             _ocl_kernel_generate_partition_descriptor_s2,
             _ocl_kernel_generate_partition_descriptor_s3,
             _ocl_command_queue,
             _csr5_sigma, _p, _m,
             _bit_y_offset, _bit_scansum_offset, _num_packet,
             _csr_row_pointer, _csr5_partition_pointer, _csr5_partition_descriptor,
             _csr5_partition_descriptor_offset_pointer, &_num_offsets, &time);
        if (err != ANONYMOUSLIB_SUCCESS)
            return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
        //err = clFinish(_ocl_command_queue);
        //tile_desc_time += tile_desc_timer.stop();
        tile_desc_time += time;

        if (_num_offsets)
        {
            //std::cout << "has empty rows, _num_offsets = " << _num_offsets << std::endl;

            malloc_timer.start();

            _csr5_partition_descriptor_offset = clCreateBuffer(_ocl_context, CL_MEM_READ_WRITE, _num_offsets * sizeof(ANONYMOUSLIB_IT), NULL, &err);
            if(err != CL_SUCCESS) return err;
            err = clFinish(_ocl_command_queue);
            malloc_time += malloc_timer.stop();

            //tile_desc_timer.start();
            err = generate_partition_descriptor_offset<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>
                (_ocl_kernel_generate_partition_descriptor_offset, _ocl_command_queue,
                 _csr5_sigma, _p,
                 _bit_y_offset, _bit_scansum_offset, _num_packet,
                 _csr_row_pointer, _csr5_partition_pointer, _csr5_partition_descriptor,
                 _csr5_partition_descriptor_offset_pointer, _csr5_partition_descriptor_offset, &time);
            if (err != ANONYMOUSLIB_SUCCESS)
                return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
            //err = clFinish(_ocl_command_queue);
            //tile_desc_time += tile_desc_timer.stop();
            tile_desc_time += time;
        }
        else
        {
            _csr5_partition_descriptor_offset = clCreateBuffer(_ocl_context, CL_MEM_READ_WRITE, 1 * sizeof(ANONYMOUSLIB_IT), NULL, &err);
            if(err != CL_SUCCESS) return err;
        }

        // step 3. transpose column_index and value arrays
        //transpose_timer.start();
        err = aosoa_transpose<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>
            (_ocl_kernel_aosoa_transpose_smem_iT, _ocl_kernel_aosoa_transpose_smem_vT, _ocl_command_queue,
             _csr5_sigma, _nnz, _csr5_partition_pointer, _csr_column_index, _csr_value, 1, &time);
        if (err != ANONYMOUSLIB_SUCCESS)
            return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
        //err = clFinish(_ocl_command_queue);
        //transpose_time += transpose_timer.stop();
        transpose_time += time;

        std::cout << std::endl << "CSR->CSR5 malloc time = " << malloc_time << " s." << std::endl;
        std::cout << "CSR->CSR5 tile_ptr time = " << tile_ptr_time << " s." << std::endl;
        std::cout << "CSR->CSR5 tile_desc time = " << tile_desc_time << " s." << std::endl;
        std::cout << "CSR->CSR5 transpose time = " << transpose_time << " s." << std::endl;

        _format = ANONYMOUSLIB_FORMAT_CSR5;
    }

    return err;
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::asCSR5_(double *time)
{
    int err = ANONYMOUSLIB_SUCCESS;

    if (_format == ANONYMOUSLIB_FORMAT_CSR5)
        return err;

    if (_format == ANONYMOUSLIB_FORMAT_CSR)
    {
        // double malloc_time = 0, tile_ptr_time = 0, tile_desc_time = 0;
        // double transpose_time = 0;
        // anonymouslib_timer malloc_timer, tile_ptr_timer, tile_desc_timer, transpose_timer;
        // compute sigma

        // calculate the number of partitions

        // malloc the newly added arrays for CSR5

        // convert csr data to csr5 data (3 steps)
        // step 1. generate partition pointer

        // step 2. generate partition descriptor

        // step 3. transpose column_index and value arrays
        err = aosoa_transpose<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>
            (_ocl_kernel_aosoa_transpose_smem_iT, _ocl_kernel_aosoa_transpose_smem_vT, _ocl_command_queue,
             _csr5_sigma, _nnz, _csr5_partition_pointer, _csr_column_index, _csr_value, 1, time);
        if (err != ANONYMOUSLIB_SUCCESS)
            return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
        //transpose_time += time;


        _format = ANONYMOUSLIB_FORMAT_CSR5;
    }

    return err;
}

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::setX(cl_mem x) {
    int err = ANONYMOUSLIB_SUCCESS;

    _x = x;

    return err;
}

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::spmv(const ANONYMOUSLIB_VT alpha,
                                                                                 cl_mem y_t,
                                                                                 cl_mem y_b,
                                                                                 double* time) {
    int err = ANONYMOUSLIB_SUCCESS;

    if (_format == ANONYMOUSLIB_FORMAT_CSR) {
        return ANONYMOUSLIB_UNSUPPORTED_CSR_SPMV;
    }

    if (_format == ANONYMOUSLIB_FORMAT_CSR5) {
        csr5_spmv<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>(_ocl_kernel_spmv_csr5_compute, _ocl_kernel_spmv_csr5_calibrate,
                                                                      _ocl_kernel_spmv_csr5_tail_partition, _ocl_command_queue,
                                                                      _csr5_sigma, _p, _m, _bit_y_offset, _bit_scansum_offset, _num_packet,
                                                                      _csr_row_pointer, _csr_column_index, _csr_value, _csr5_partition_pointer,
                                                                      _csr5_partition_descriptor, _csr5_partition_descriptor_offset_pointer,
                                                                      _csr5_partition_descriptor_offset,_temp_calibrator_t, _temp_calibrator_b,
                                                                      _tail_partition_start, alpha, _x, /*beta,*/ y_t, y_b, time);
    }

    return err;
}

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::destroy() {
    return asCSR();
}

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::setSigma(int sigma) {
    int err = ANONYMOUSLIB_SUCCESS;

    if (sigma == ANONYMOUSLIB_AUTO_TUNED_SIGMA) {
        int r = 4;
        int s = 7;
        int t = 256;
        int u = 4;

        int nnz_per_row = _nnz / _m;
        if (nnz_per_row <= r) {
            _csr5_sigma = r;
        } else if (nnz_per_row > r && nnz_per_row <= s) {
            _csr5_sigma = nnz_per_row;
        } else if (nnz_per_row <= t && nnz_per_row > s) {
            _csr5_sigma = s;
        } else { // nnz_per_row > t
            _csr5_sigma = u;
        }
    } else {
        _csr5_sigma = sigma;
    }

    char omega_str[3]; //supports up to 999
    snprintf(omega_str, sizeof(omega_str), "%d", ANONYMOUSLIB_CSR5_OMEGA);
    char sigma_str[3]; //supports up to 999
    snprintf(sigma_str, sizeof(sigma_str), "%d", _csr5_sigma);
    char threadgroup_str[4]; //supports up to 9999
    snprintf(threadgroup_str, sizeof(threadgroup_str), "%d", ANONYMOUSLIB_THREAD_GROUP);
    char threadbunch_str[3]; //supports up to 999
    snprintf(threadbunch_str, sizeof(threadbunch_str), "%d", ANONYMOUSLIB_THREAD_BUNCH);

    char* it_str = (char*) "int";
    char* uit_str = (char*) "unsigned";

    char* vt_str;
    if (sizeof(ANONYMOUSLIB_VT) == 8) {
        vt_str = (char*) "double";
    } else if (sizeof(ANONYMOUSLIB_VT) == 4) {
        vt_str = (char*) "float";
    } else {
        perror("wrong ANONYMOUSLIB_VT size");
        exit(EXIT_FAILURE);
    }

    std::string ocl_source_code_string_format = _ocl_source_code_string_format_const;
    std::string ocl_source_code_string_csr5_spmv = _ocl_source_code_string_csr5_spmv_const;

    std::string format_ocl_options;
    std::string csr5_spmv_ocl_options;

    // replace 'omega_replace_str' by 'omega_str'
    format_ocl_options.append(" -D_REPLACE_ANONYMOUSLIB_CSR5_OMEGA_SEGMENT_=");
    format_ocl_options.append(omega_str);
    csr5_spmv_ocl_options.append(" -D_REPLACE_ANONYMOUSLIB_CSR5_OMEGA_SEGMENT_=");
    csr5_spmv_ocl_options.append(omega_str);

    // replace 'sigma_replace_str' by 'sigma_str'
    format_ocl_options.append(" -D_REPLACE_ANONYMOUSLIB_CSR5_SIGMA_SEGMENT_=");
    format_ocl_options.append(sigma_str);
    csr5_spmv_ocl_options.append(" -D_REPLACE_ANONYMOUSLIB_CSR5_SIGMA_SEGMENT_=");
    csr5_spmv_ocl_options.append(sigma_str);

    // replace 'threadgroup_replace_str' by 'threadgroup_str'
    format_ocl_options.append(" -D_REPLACE_ANONYMOUSLIB_THREAD_GROUP_SEGMENT_=");
    format_ocl_options.append(threadgroup_str);
    csr5_spmv_ocl_options.append(" -D_REPLACE_ANONYMOUSLIB_THREAD_GROUP_SEGMENT_=");
    csr5_spmv_ocl_options.append(threadgroup_str);

    // replace 'threadbunch_replace_str' by 'threadbunch_str'
    csr5_spmv_ocl_options.append(" -D_REPLACE_ANONYMOUSLIB_THREAD_BUNCH_SEGMENT_=");
    csr5_spmv_ocl_options.append(threadbunch_str);

    // replace 'it_replace_str' by 'it_str'
    format_ocl_options.append(" -D_REPLACE_ANONYMOUSLIB_CSR5_INDEX_TYPE_SEGMENT_=");
    format_ocl_options.append(it_str);
    csr5_spmv_ocl_options.append(" -D_REPLACE_ANONYMOUSLIB_CSR5_INDEX_TYPE_SEGMENT_=");
    csr5_spmv_ocl_options.append(it_str);

    // replace 'uit_replace_str' by 'uit_str'
    format_ocl_options.append(" -D_REPLACE_ANONYMOUSLIB_CSR5_UNSIGNED_INDEX_TYPE_SEGMENT_=");
    format_ocl_options.append(uit_str);
    csr5_spmv_ocl_options.append(" -D_REPLACE_ANONYMOUSLIB_CSR5_UNSIGNED_INDEX_TYPE_SEGMENT_=");
    csr5_spmv_ocl_options.append(uit_str);

    // replace 'vt_replace_str' by 'vt_str'
    format_ocl_options.append(" -D_REPLACE_ANONYMOUSLIB_CSR5_VALUE_TYPE_SEGMENT_=");
    format_ocl_options.append(vt_str);
    csr5_spmv_ocl_options.append(" -D_REPLACE_ANONYMOUSLIB_CSR5_VALUE_TYPE_SEGMENT_=");
    csr5_spmv_ocl_options.append(vt_str);

    const char* ocl_source_code_format = ocl_source_code_string_format.c_str();
    const char* ocl_source_code_csr5_spmv = ocl_source_code_string_csr5_spmv.c_str();

//    std::cout << ocl_source_code_csr5_spmv << std::endl;
//    std::cout << ocl_source_code_format << std::endl;

//    std::cout << csr5_spmv_ocl_options << std::endl;
//    std::cout << format_ocl_options << std::endl;

    // Create the program
    size_t source_size_format[] = {strlen(ocl_source_code_format)};
    _ocl_program_format = clCreateProgramWithSource(_ocl_context, 1, &ocl_source_code_format, source_size_format, &err);
    CL_CHECK(err);
    size_t source_size_csr5_spmv[] = {strlen(ocl_source_code_csr5_spmv)};
    _ocl_program_csr5_spmv = clCreateProgramWithSource(_ocl_context, 1, &ocl_source_code_csr5_spmv, source_size_csr5_spmv, &err);
    CL_CHECK(err);


    // Build the program
    build_and_check(_ocl_program_format,format_ocl_options.c_str(), _ocl_device[0]);

    build_and_check(_ocl_program_csr5_spmv,csr5_spmv_ocl_options.c_str(), _ocl_device[0]);


    // Create kernels
    _ocl_kernel_warmup = clCreateKernel(_ocl_program_format, "warmup_kernel", &err);
    CL_CHECK(err);
    _ocl_kernel_generate_partition_pointer_s1 = clCreateKernel(_ocl_program_format, "generate_partition_pointer_s1_kernel", &err);
    CL_CHECK(err);
    _ocl_kernel_generate_partition_pointer_s2 = clCreateKernel(_ocl_program_format, "generate_partition_pointer_s2_kernel", &err);
    CL_CHECK(err);
    _ocl_kernel_generate_partition_descriptor_s0 = clCreateKernel(_ocl_program_format, "generate_partition_descriptor_s0_kernel", &err);
    CL_CHECK(err);
    _ocl_kernel_generate_partition_descriptor_s1 = clCreateKernel(_ocl_program_format, "generate_partition_descriptor_s1_kernel", &err);
    CL_CHECK(err);
    _ocl_kernel_generate_partition_descriptor_s2 = clCreateKernel(_ocl_program_format, "generate_partition_descriptor_s2_kernel", &err);
    CL_CHECK(err);
    _ocl_kernel_generate_partition_descriptor_s3 = clCreateKernel(_ocl_program_format, "generate_partition_descriptor_s3_kernel", &err);
    CL_CHECK(err);
    _ocl_kernel_generate_partition_descriptor_offset = clCreateKernel(_ocl_program_format, "generate_partition_descriptor_offset_kernel", &err);
    CL_CHECK(err);
    _ocl_kernel_aosoa_transpose_smem_iT = clCreateKernel(_ocl_program_format, "aosoa_transpose_kernel_smem_iT", &err);
    CL_CHECK(err);
    _ocl_kernel_aosoa_transpose_smem_vT = clCreateKernel(_ocl_program_format, "aosoa_transpose_kernel_smem_vT", &err);
    CL_CHECK(err);

    _ocl_kernel_spmv_csr5_compute = clCreateKernel(_ocl_program_csr5_spmv, "spmv_csr5_compute_kernel", &err);
    CL_CHECK(err);
    _ocl_kernel_spmv_csr5_calibrate = clCreateKernel(_ocl_program_csr5_spmv, "spmv_csr5_calibrate_kernel", &err);
    CL_CHECK(err);
    _ocl_kernel_spmv_csr5_tail_partition = clCreateKernel(_ocl_program_csr5_spmv, "spmv_csr5_tail_partition_kernel", &err);
    CL_CHECK(err);

    return err;
}

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::computeSigma() {
    return _csr5_sigma;
}

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::setOCLENV(cl_context ocl_context,
                                                                                      cl_command_queue ocl_command_queue,
                                                                                      cl_device_id* ocl_device) {
    int err = ANONYMOUSLIB_SUCCESS;

    _ocl_context = ocl_context;
    _ocl_command_queue = ocl_command_queue;
    _ocl_device = ocl_device;

    convertToString("../src/detail/kcode/format.cl", _ocl_source_code_string_format_const);

    convertToString("../src/detail/kcode/csr5_spmv.cl", _ocl_source_code_string_csr5_spmv_const);

    return err;
}

#endif // ANONYMOUSLIB_OPENCL_H
