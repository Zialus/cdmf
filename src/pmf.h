#ifndef PMF_H
#define PMF_H

class parameter {
public:
    int version = 1;
    unsigned k = 10;
    int threads = 4;
    int maxinneriter = 5;
    int maxiter = 5;
    VALUE_TYPE lambda = 0.1f;
    int do_predict = 0; // predict RMSE
    int verbose = 0;
    unsigned platform_id = 0;
    int do_nmf = 0;  // non-negative matrix factorization
    int nBlocks = 16;
    int nThreadsPerBlock = 32;
    int do_ref = 0; // compare opencl results to reference results
    char scr_dir[1024] = "../data/simple";
    char kcode_path[1024] = "../kcode";
    char device_type[4] = {'g', 'p', 'u', '\0'};
};

#endif
