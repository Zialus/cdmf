#ifndef TOOLS_H
#define TOOLS_H

#include "util.h"

using namespace std;

inline double gettime() {
    struct timeval t;
    gettimeofday(&t, nullptr);
    return t.tv_sec + t.tv_usec * 1e-6;
}

inline char* getT(unsigned sz) {
    if (sz == 8) { return (char*) "double"; }
    if (sz == 4) { return (char*) "float"; }
    return (char*) "float";
}

template<typename iT, typename vT>
double getB(const iT m, const iT nnz) {
    return (double) ((m + 1 + nnz) * sizeof(iT) + (2 * nnz + m) * sizeof(vT));
}

template<typename iT>
double getFLOP(const iT nnz) {
    return (double) (2 * nnz);
}

template<typename T>
inline std::string to_string(T value) {
    std::ostringstream os;
    os << value;
    return os.str();
}


void load(const char* srcdir, smat_t& R, bool ifALS, bool with_weights);

void initial_col(mat_t& X, unsigned int k, unsigned int n);

void convertToString(const char* filename, string& s);

int getPlatform(cl_platform_id& platform, int id);

cl_device_id* getDevice(cl_platform_id& platform, char* device_type);

void print_all_the_info();

void print_all_the_platforms();

int report_device(cl_device_id device_id);

parameter parse_command_line(int argc, char** argv, char* input_file_name, char* kernel_code);

void print_matrix(mat_t M, unsigned k, unsigned n);

void exit_with_help();

const char* get_error_string(cl_int err);

void golden_compare(mat_t W, mat_t W_ref, unsigned k, unsigned m);

void calculate_rmse_ocl(const mat_t& W_c, const mat_t& H_c, const int k, const char* srcdir);

#endif // TOOLS_H
