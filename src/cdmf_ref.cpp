#include <omp.h>

#include "tools.h"

#define kind dynamic,500

inline VALUE_TYPE RankOneUpdate(const smat_t& R, const long j, const vec_t& u, const VALUE_TYPE lambda, int do_nmf) {
    VALUE_TYPE g = 0, h = lambda;
    if (R.col_ptr[j + 1] == R.col_ptr[j]) { return 0; }
    for (unsigned idx = R.col_ptr[j]; idx < R.col_ptr[j + 1]; ++idx) {
        unsigned i = R.row_idx[idx];
        g += u[i] * R.val[idx];
        h += u[i] * u[i];
    }
    VALUE_TYPE newvj = g / h;
    if (do_nmf > 0 && newvj < 0) {
        newvj = 0;
    }
    return newvj;
}

inline VALUE_TYPE UpdateRating(smat_t& R, const vec_t& Wt, const vec_t& Ht, const vec_t& oldWt, const vec_t& oldHt) {
    VALUE_TYPE loss = 0;
#pragma omp parallel for  schedule(kind) reduction(+:loss)
    for (long c = 0; c < R.cols; ++c) {
        VALUE_TYPE Htc = Ht[c], oldHtc = oldHt[c], loss_inner = 0;
        for (unsigned idx = R.col_ptr[c]; idx < R.col_ptr[c + 1]; ++idx) {
            R.val[idx] -= Wt[R.row_idx[idx]] * Htc - oldWt[R.row_idx[idx]] * oldHtc;
            loss_inner += R.val[idx] * R.val[idx];
        }
        loss += loss_inner;
    }
    return loss;
}

inline VALUE_TYPE UpdateRating(smat_t& R, const vec_t& Wt2, const vec_t& Ht2) {
    VALUE_TYPE loss = 0;
#pragma omp parallel for schedule(kind) reduction(+:loss)
    for (long c = 0; c < R.cols; ++c) {
        VALUE_TYPE Htc = Ht2[2 * c], oldHtc = Ht2[2 * c + 1], loss_inner = 0;
        for (unsigned idx = R.col_ptr[c]; idx < R.col_ptr[c + 1]; ++idx) {
            R.val[idx] -= Wt2[2 * R.row_idx[idx]] * Htc - Wt2[2 * R.row_idx[idx] + 1] * oldHtc;
            loss_inner += R.val[idx] * R.val[idx];
        }
        loss += loss_inner;
    }
    return loss;
}

inline VALUE_TYPE UpdateRating(smat_t& R, const vec_t& Wt, const vec_t& Ht, bool add) {
    VALUE_TYPE loss = 0;
    if (add) {
#pragma omp parallel for schedule(kind) reduction(+:loss)
        for (long c = 0; c < R.cols; ++c) {
            VALUE_TYPE Htc = Ht[c], loss_inner = 0;
            for (unsigned idx = R.col_ptr[c]; idx < R.col_ptr[c + 1]; ++idx) {
                R.val[idx] += Wt[R.row_idx[idx]] * Htc;
                loss_inner += R.val[idx] * R.val[idx];
            }
            loss += loss_inner;
        }
        return loss;
    } else {
#pragma omp parallel for schedule(kind) reduction(+:loss)
        for (long c = 0; c < R.cols; ++c) {
            VALUE_TYPE Htc = Ht[c], loss_inner = 0;
            for (unsigned idx = R.col_ptr[c]; idx < R.col_ptr[c + 1]; ++idx) {
                R.val[idx] -= Wt[R.row_idx[idx]] * Htc;
                loss_inner += R.val[idx] * R.val[idx];
            }
            loss += loss_inner;
        }
        return loss;
    }
}

// Matrix Factorization based on Coordinate Descent
void cdmf_ref(smat_t& R, mat_t& W, mat_t& H, parameter& param) {
    VALUE_TYPE lambda = param.lambda;

    int num_threads_old = omp_get_num_threads();
    omp_set_num_threads(param.threads);

    // Create transpose view of R
    smat_t Rt;
    Rt = R.transpose();
    // H is a zero matrix now.
    for (unsigned t = 0; t < param.k; ++t) {
        for (unsigned c = 0; c < R.cols; ++c) {
            H[t][c] = 0;
        }
    }

    vec_t u(R.rows);
    vec_t v(R.cols);

    for (int oiter = 1; oiter <= param.maxiter; ++oiter) {

        for (unsigned t = 0; t < param.k; ++t) {
            vec_t& Wt = W[t];
            vec_t& Ht = H[t];

#pragma omp parallel for
            for (long i = 0; i < R.rows; ++i) { u[i] = Wt[i]; }
#pragma omp parallel for
            for (long i = 0; i < R.cols; ++i) { v[i] = Ht[i]; }

            // Create Rhat = R - Wt Ht^T
            if (oiter > 1) {
                UpdateRating(R, Wt, Ht, true);
                UpdateRating(Rt, Ht, Wt, true);
            }

            for (int iter = 1; iter <= param.maxinneriter; ++iter) {
                // Update H[t]
#pragma omp parallel for schedule(kind) shared(u, v)
                for (long c = 0; c < R.cols; ++c) {
                    v[c] = RankOneUpdate(R, c, u, lambda * (R.col_ptr[c + 1] - R.col_ptr[c]), param.do_nmf);
                }
                // Update W[t]
#pragma omp parallel for schedule(kind) shared(u, v)
                for (long c = 0; c < Rt.cols; ++c) {
                    u[c] = RankOneUpdate(Rt, c, v, lambda * (Rt.col_ptr[c + 1] - Rt.col_ptr[c]), param.do_nmf);
                }
            }

            // Update R and Rt
#pragma omp parallel for
            for (long i = 0; i < R.rows; ++i) { Wt[i] = u[i]; }
#pragma omp parallel for
            for (long i = 0; i < R.cols; ++i) { Ht[i] = v[i]; }

            UpdateRating(R, u, v, false);
            UpdateRating(Rt, v, u, false);
        }
    }

    omp_set_num_threads(num_threads_old);
}
