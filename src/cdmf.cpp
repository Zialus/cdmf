#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <string>
#include <fstream>

#include "util.h"
#include "tools.h"

void cdmf_ref(smat_t& R, mat_t& W, mat_t& H, parameter& param);

void cdmf_ocl(smat_t& R, mat_t& W, mat_t& H, parameter& param, char filename[]);

void cdmf_csr5(smat_t& R, mat_t& W, mat_t& H, parameter& param, char filename[]);

int main(int argc, char** argv) {
    parameter param = parse_command_line(argc, argv);

    // reading rating matrix
    smat_t R;   // val: csc, val_t: csr
    mat_t W;
    mat_t W_ref;
    mat_t H;
    mat_t H_ref;

    std::cout << "[info] load rating data." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    load(param.scr_dir, R, false, false);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT = t2 - t1;;
    printf("[info] - loading time: %lf s\n", deltaT.count());

    // W, H  here are k*m, k*n
    std::cout << "[info] initializ W and H matrix." << std::endl;
    initial_col(W, param.k, R.rows);
    initial_col(W_ref, param.k, R.rows);
    initial_col(H, param.k, R.cols);
    initial_col(H_ref, param.k, R.cols);

    // compute cdmf on the ocl device
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "[info] compute cdmf on the selected ocl device." << std::endl;

    switch (param.version) {
        case 1: {
            std::cout << "[info] Picked Version 1: Native" << std::endl;
            char kcode_filename[1024+10];
            sprintf(kcode_filename, "%s/ccd01.cl", param.kcode_path);
            cdmf_ocl(R, W, H, param, kcode_filename);
            break;
        }
        case 2: {
            std::cout << "[info] Picked Version 2: Thread Batching" << std::endl;
            char kcode_filename[1024+10];
            sprintf(kcode_filename, "%s/ccd033.cl", param.kcode_path);
            cdmf_ocl(R, W, H, param, kcode_filename);
            break;
        }
        case 3: {
            std::cout << "[info] Picked Version 3: Load Balancing" << std::endl;
            char kcode_filename[1024+10];
            sprintf(kcode_filename, "%s/ccd033.cl", param.kcode_path);
            cdmf_csr5(R, W, H, param, kcode_filename);
            break;
        }
        default: {
            printf("Wrong version");
            return EXIT_FAILURE;
        }
    }


    if (param.do_predict == 1) {
        calculate_rmse_ocl(W, H, param.k, param.scr_dir);
    }

    // compare reference and OpenCL results
    if (param.do_ref == 1) {
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "[info] now computing cdmf reference results on a cpu core." << std::endl;
        cdmf_ref(R, W_ref, H_ref, param);
        std::cout << "[info] validate the results." << std::endl;
        golden_compare(W, W_ref, param.k, R.rows);
        golden_compare(H, H_ref, param.k, R.cols);
//        calculate_rmse_ocl(W_ref, H_ref, param.k, scr_dir);
    }
    std::cout << "------------------------------------------------------" << std::endl;

    // Some print debugging
//    print_matrix(W,param.k,R.rows);
//    print_matrix(H,param.k,R.cols);
//
//
//    print_matrix(W_ref,param.k,R.rows);
//    print_matrix(H_ref,param.k,R.cols);

    return EXIT_SUCCESS;
}
