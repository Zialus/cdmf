#ifndef UTILS_OPENCL_H
#define UTILS_OPENCL_H

#include "common_opencl.h"

#define CL_CHECK(res) \
    {if (res != CL_SUCCESS) {fprintf(stderr,"Error \"%s\" (%d) in file %s on line %d\n", \
        get_error_string(res), res, __FILE__,__LINE__); abort();}}

struct anonymouslib_timer {
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;

    void start() {
        t1 = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> deltaT = t2 - t1;
        return deltaT.count();
    }

};

void build_and_check(cl_program program, const char* options, cl_device_id device) {
    cl_int status = clBuildProgram(program, 0, nullptr, options, nullptr, nullptr);

    size_t length;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
    char* buffer = (char*) malloc(length + 1);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, length, buffer, nullptr);

    if (buffer != nullptr && strcmp(buffer, "") != 0 && strcmp(buffer, "\n") != 0) {
        printf("[OpenCL Compiler INFO]:\n%s\n", buffer);
        free(buffer);
    } else {
        printf("[OpenCL Compiler]: No info to print\n");
    }

    CL_CHECK(status);
    std::cout << "[INFO]: Compiled OpenCl code successfully!\n";
}

#endif // UTILS_OPENCL_H
