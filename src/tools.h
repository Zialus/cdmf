#ifndef TOOLS_H
#define TOOLS_H

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

void load(const char* srcdir, smat_t& R, bool ifALS, bool with_weights);

void initial_col(mat_t& X, unsigned int k, unsigned int n);

void convertToString(const char* filename, std::string& s);

cl_platform_id getPlatform(int id);

cl_device_id* getDevices(cl_platform_id& platform, char* device_type);

void print_all_the_info();

void print_platform_info(cl_platform_id* platforms, unsigned int id);

void print_device_info(cl_device_id* devices, unsigned int j);

int report_device(cl_device_id device_id);

parameter parse_command_line(int argc, char** argv);

void print_matrix(mat_t M, unsigned k, unsigned n);

void exit_with_help();

const char* get_error_string(cl_int err);

void golden_compare(mat_t W, mat_t W_ref, unsigned k, unsigned m);

void calculate_rmse_ocl(const mat_t& W_c, const mat_t& H_c, const int k, const char* srcdir);

#endif // TOOLS_H
