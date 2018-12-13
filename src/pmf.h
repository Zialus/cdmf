#ifndef PMF_H
#define PMF_H

class parameter {
public:
    int version;
    int k;
    int threads;
    int maxinneriter;
    int maxiter;
    VALUE_TYPE lambda;
    VALUE_TYPE eps; // for the fundec stop-cond in ccdr1
    int do_predict;
    int verbose;
    int platform_id;
    int do_nmf;  // non-negative matrix factorization
    int nBlocks;
    unsigned int nThreadsPerBlock;
    int do_ref;

    parameter() {
        version = 1;
        k = 10;
        maxiter = 5;
        maxinneriter = 5;
        lambda = 0.1f;
        threads = 4;
        eps = 1e-3f;
        do_predict = 0;
        platform_id = 0;
        verbose = 0;
        do_ref = 0;
        do_nmf = 0;
        nBlocks = 16;
        nThreadsPerBlock = 32;
    }
};

#endif
