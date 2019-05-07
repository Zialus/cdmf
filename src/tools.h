#ifndef TOOLS_H
#define TOOLS_H

#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/cl.h>
#else
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#endif

#include "pmf_util.h"
#include "pmf.h"

inline char* getT(unsigned sz) {
    if (sz == 8) { return (char*) "double"; }
    if (sz == 4) { return (char*) "float"; }
    return (char*) "float";
}

const char* get_error_string(cl_int err);

void convertToString(const char* filename, std::string& s);

cl_platform_id getPlatform(unsigned id);

cl_device_id* getDevices(cl_platform_id& platform, char* device_type);

void print_all_the_info();

void print_device_info(cl_device_id* devices, unsigned j);

void print_platform_info(cl_platform_id* platforms, unsigned id);

int report_device(cl_device_id device_id);

void load(const char* srcdir, SparseMatrix& R, TestData& T);

void initial_col(MatData& X, unsigned k, unsigned n);

void exit_with_help();

parameter parse_command_line(int argc, char** argv);

void golden_compare(MatData W, MatData W_ref, unsigned k, unsigned m);

void calculate_rmse(const MatData& W_c, const MatData& H_c, const char* srcdir, unsigned k);

double calculate_rmse_directly(MatData& W, MatData& H, TestData& T, unsigned rank, bool ifALS);

void print_matrix(MatData M, unsigned k, unsigned n);

double executionTime(cl_event& event);

#endif //TOOLS_H
