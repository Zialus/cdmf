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

int main(int argc, char** argv) {
    char scr_dir[1024];
//    char kcode_filename[1024];
    parameter param = parse_command_line(argc, argv, scr_dir, nullptr);

    // reading rating matrix
    smat_t R;   // val: csc, val_t: csr
    mat_t W;
    mat_t W_ref;
    mat_t H;
    mat_t H_ref;

    cout << "[info] load rating data." << endl;
    double t1 = gettime();
    load(scr_dir, R, false, false);
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
            break;
        }
        case 2: {
            cout << "[info] Picked Version 2: Thread Batching" << endl;
            char kcode_filename[1024] = {"../kcode/ccd033.cl"};
            cdmf_ocl(R, W, H, param, kcode_filename);
            break;
        }
        case 3: {
            cout << "[info] Picked Version 3: Load Balancing" << endl;
            char kcode_filename[1024] = {"../kcode/ccd033.cl"};
            cdmf_csr5(R, W, H, param, kcode_filename);
            break;
        }
        default: {
            printf("Wrong version");
            break;
        }
    }


    if (param.do_predict == 1) {
        calculate_rmse_ocl(W, H, param.k, scr_dir);
    }

    // compare reference and OpenCL results
    if (param.do_ref == 1) {
        cout << "--------------------------------------------------" << endl;
        cout << "[info] now computing cdmf reference results on a cpu core." << endl;
        cdmf_ref(R, W_ref, H_ref, param);
        cout << "[info] validate the results." << endl;
        golden_compare(W, W_ref, param.k, R.rows);
        golden_compare(H, H_ref, param.k, R.cols);
    }
    cout << "------------------------------------------------------" << endl;

    // Some print debugging
    print_matrix(W,param.k,R.rows);
    print_matrix(H,param.k,R.cols);


    print_matrix(W_ref,param.k,R.rows);
    print_matrix(H_ref,param.k,R.cols);

    return 0;
}
