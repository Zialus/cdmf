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

#endif // UTILS_OPENCL_H
