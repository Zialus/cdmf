#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <string>
#include <fstream>

#include "util.h"

using namespace std;

void cdmf_ref(smat_t& R, mat_t& W, mat_t& H, parameter& param);
void cdmf_ocl(smat_t& R, mat_t& W, mat_t& H, parameter& param, char filename[]);
void cdmf_csr5(smat_t& R, mat_t& W, mat_t& H, parameter& param, char filename[]);
void calculate_rmse_ocl(const mat_t& W_c, const mat_t& H_c, const parameter& param, const char* srcdir);

int main(int argc, char** argv) {
    char input_file_name[1024];
    parameter param = parse_command_line(argc, argv, input_file_name, nullptr);

    // reading rating matrix
    smat_t R;   // val: csc, val_t: csr
    mat_t W;
    mat_t W_ref;
    mat_t H;
    mat_t H_ref;

    cout << "[info] load rating data." << endl;
    double t1 = gettime();
    load(input_file_name, R, false, false);
    double t2 = gettime();
    double deltaT = t2 - t1;
    printf("[info] - loading time: %lf s\n", deltaT);

    // W, H  here are k*m, k*n
    cout << "[info] initializ W and H matrix." << endl;
    initial_col(W, param.k, R.rows);
    initial_col(W_ref, param.k, R.rows);
    initial_col(H, param.k, R.cols);
    initial_col(H_ref, param.k, R.cols);

    // compute cdmf on the ocl device
    cout << "------------------------------------------------------" << endl;
    cout << "[info] compute cdmf on the selected ocl device." << endl;

    switch (param.version) {
        case 1: {
            cout << "[info] Picked Version 1: Native" << endl;
            char kcode_filename[1024] = {"../kcode/ccd01.cl"};
            cdmf_ocl(R, W, H, param, kcode_filename);
            calculate_rmse_ocl(W, H, param, input_file_name);
            break;
        }
        case 2: {
            cout << "[info] Picked Version 2: Thread Batching" << endl;
            char kcode_filename[1024] = {"../kcode/ccd033.cl"};
            cdmf_ocl(R, W, H, param, kcode_filename);
            calculate_rmse_ocl(W, H, param, input_file_name);
            break;
        }
        case 3: {
            cout << "[info] Picked Version 3: Load Balancing" << endl;
            char kcode_filename[1024] = {"../kcode/ccd033.cl"};
            cdmf_csr5(R, W, H, param, kcode_filename);
            calculate_rmse_ocl(W, H, param, input_file_name);
            break;
        }
        default: {
            printf("Wrong version");
            break;
        }
    }

    cout << "------------------------------------------------------" << endl;
    cout << "[info] now computing cdmf reference results on a cpu core." << endl;
    cdmf_ref(R, W_ref, H_ref, param);

    for (int t = 0; t < param.k; t++) {
        printf("|%lf - %lf |", W_ref[t][56915 - 1], H_ref[t][62245 - 1]);
    }

    // compare reference and anonymouslib results
    cout << "[info] validate the results." << endl;
    golden_compare(W, W_ref, param.k, R.rows);
    golden_compare(H, H_ref, param.k, R.cols);

    cout << "------------------------------------------------------" << endl;

    return 0;
}
