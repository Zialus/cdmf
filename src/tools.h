#ifndef TOOLS_H
#define TOOLS_H

#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/cl.h>
#else
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#endif

#include "util.h"
#include "pmf.h"

inline char* getT(unsigned sz) {
    if (sz == 8) { return (char*) "double"; }
    if (sz == 4) { return (char*) "float"; }
    return (char*) "float";
}

template<typename T>
inline std::string to_string(T value) {
    std::ostringstream os;
    os << value;
    return os.str();
}

const char* get_error_string(cl_int err);

void convertToString(const char* filename, std::string& s);

cl_platform_id getPlatform(int id);

cl_device_id* getDevices(cl_platform_id& platform, char* device_type);

void print_all_the_info();

void print_device_info(cl_device_id* devices, unsigned int j);

void print_platform_info(cl_platform_id* platforms, unsigned int id);

int report_device(cl_device_id device_id);

void load(const char* srcdir, smat_t& R, bool ifALS, bool with_weights);

void initial_col(mat_t& X, unsigned k, unsigned n);

void exit_with_help();

parameter parse_command_line(int argc, char** argv);

void golden_compare(mat_t W, mat_t W_ref, unsigned k, unsigned m);

void calculate_rmse_ocl(const mat_t& W_c, const mat_t& H_c, const int k, const char* srcdir);

void print_matrix(mat_t M, unsigned k, unsigned n);

#endif // TOOLS_H
