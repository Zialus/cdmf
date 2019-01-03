#include "util.h"
#include "tools.h"

void cdmf_ref(smat_t& R, mat_t& W, mat_t& H, parameter& param);

void cdmf_ocl(smat_t& R, mat_t& W, mat_t& H, parameter& param, char filename[]);

void cdmf_csr5(smat_t& R, mat_t& W, mat_t& H, parameter& param, char filename[]);

int main(int argc, char** argv) {
    parameter param = parse_command_line(argc, argv);

    if (param.verbose) {
        print_all_the_info();
    }

    std::cout << "[info] Loading R matrix..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    smat_t R;
    bool with_weights = false;
    bool ifALS = false;
    load(param.scr_dir, R, ifALS, with_weights);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT = t2 - t1;;
    std::cout << "[INFO] Loading rating data time: " << deltaT.count() << "s.\n";
    std::cout << "------------------------------------------------------" << std::endl;

    mat_t W;
    mat_t H;
    initial_col(W, param.k, R.rows);
    initial_col(H, param.k, R.cols);

    mat_t W_ref;
    mat_t H_ref;
    initial_col(W_ref, param.k, R.rows);
    initial_col(H_ref, param.k, R.cols);


    switch (param.version) {
        case 1: {
            std::cout << "[info] Picked Version 1: Native" << std::endl;
            char kcode_filename[1024 + 10];
            snprintf(kcode_filename, sizeof(kcode_filename), "%s/ccd01.cl", param.kcode_path);
            cdmf_ocl(R, W, H, param, kcode_filename);
            break;
        }
        case 2: {
            std::cout << "[info] Picked Version 2: Thread Batching" << std::endl;
            char kcode_filename[1024 + 10];
            snprintf(kcode_filename, sizeof(kcode_filename), "%s/ccd033.cl", param.kcode_path);
            cdmf_ocl(R, W, H, param, kcode_filename);
            break;
        }
        case 3: {
            std::cout << "[info] Picked Version 3: Load Balancing" << std::endl;
            char kcode_filename[1024 + 10];
            snprintf(kcode_filename, sizeof(kcode_filename), "%s/ccd033.cl", param.kcode_path);
            cdmf_csr5(R, W, H, param, kcode_filename);
            break;
        }
        default: {
            printf("[FAILED] Wrong version");
            return EXIT_FAILURE;
        }
    }

    // Predict RMSE with the W and H matrices produced by OpenCL kernels
    if (param.do_predict == 1) {
        auto t5 = std::chrono::high_resolution_clock::now();
        std::cout << "------------------------------------------------------" << std::endl;
        calculate_rmse(W, H, param.scr_dir, param.k);
        auto t6 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> deltaT2 = t6 - t5;;
        printf("[info] Predict time: %lf s\n", deltaT2.count());
    }

    // Compare OpenCL results with reference OpenMP results
    if (param.do_ref == 1) {
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "[info] Computing cdmf OpenMP reference results on CPU." << std::endl;
        cdmf_ref(R, W_ref, H_ref, param);
        std::cout << "[info] validate the results." << std::endl;
        golden_compare(W, W_ref, param.k, R.rows);
        golden_compare(H, H_ref, param.k, R.cols);
        calculate_rmse(W_ref, H_ref, param.scr_dir, param.k);
    }
    std::cout << "------------------------------------------------------" << std::endl;

    // Some print debugging
//    print_matrix(W, param.k, R.rows);
//    print_matrix(H, param.k, R.cols);
//
//    print_matrix(W_ref, param.k, R.rows);
//    print_matrix(H_ref, param.k, R.cols);

    return EXIT_SUCCESS;
}
