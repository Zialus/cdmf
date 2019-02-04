static void UV(const unsigned cols,
               __global const unsigned* col_ptr,
               __global VALUE_TYPE* Ht,
               __global VALUE_TYPE* Hb,
               __global VALUE_TYPE* H,
               const VALUE_TYPE lambda) {
    unsigned i = get_global_id(0);
    //for(unsigned i=0; i<cols; i++)
    //{
    if (i < cols) {
        VALUE_TYPE t = Ht[i];
        VALUE_TYPE b = Hb[i];
        VALUE_TYPE l = lambda * (col_ptr[i + 1] - col_ptr[i]);
        VALUE_TYPE s = t / (l + b);
        H[i] = s;
        Ht[i] = 0;
        Hb[i] = 0;
    }
    //}
}

__kernel void CALV(const unsigned cols,
                   __global const unsigned* col_ptr,
                   __global VALUE_TYPE* Ht,
                   __global VALUE_TYPE* Hb,
                   __global VALUE_TYPE* H,
                   const VALUE_TYPE lambda) {
    UV(cols, col_ptr, Ht, Hb, H, lambda);
}

__kernel void CALU(const unsigned rows,
                   __global const unsigned* row_ptr,
                   __global VALUE_TYPE* Wt,
                   __global VALUE_TYPE* Wb,
                   __global VALUE_TYPE* W,
                   const VALUE_TYPE lambda) {
    UV(rows, row_ptr, Wt, Wb, W, lambda);
}

__kernel void RankOneUpdate_dev(__global const unsigned* col_ptr,
                                __global const unsigned* row_idx,
                                __global const VALUE_TYPE* val,
                                __global const VALUE_TYPE* u_vec_t,
                                const VALUE_TYPE lambda,
                                __global VALUE_TYPE* v) {
//  unsigned ii = get_global_id(0);
//  unsigned jj = get_global_size(0);
    unsigned ss = get_local_id(0);
    unsigned gg = get_local_size(0);
    unsigned aa = get_group_id(0);
//  unsigned dd = get_num_groups(0);

    __local VALUE_TYPE g[WG_SIZE], h[WG_SIZE];
    g[ss] = 0;
    h[ss] = 0;
    size_t j = aa;
    VALUE_TYPE nlambda = lambda * (col_ptr[j + 1] - col_ptr[j]);
    if ((col_ptr[j + 1] == col_ptr[j]) && (ss == 0)) {
        v[j] = 0.0;
    }
    for (unsigned idx = col_ptr[j] + ss; idx < col_ptr[j + 1]; idx += gg) {
        unsigned i = row_idx[idx];
        g[ss] += u_vec_t[i] * val[idx];
        h[ss] += u_vec_t[i] * u_vec_t[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned i = WG_SIZE/2; i > 0; i = i / 2) {
        if (ss < i) {
            g[ss] += g[ss + i];
            h[ss] += h[ss + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    v[j] = g[0] / (h[0] + nlambda);
}

/**
    Update vector v
**/
__kernel void RankOneUpdate_DUAL_kernel_v(const unsigned cols,
                                          __global const unsigned* col_ptr,
                                          __global const unsigned* row_idx,
                                          __global VALUE_TYPE* val,
                                          __global VALUE_TYPE* u,
                                          __global VALUE_TYPE* v,
                                          const VALUE_TYPE lambda,
                                          const unsigned cols_t,
                                          __global const unsigned* row_ptr,
                                          __global const unsigned* col_idx,
                                          __global VALUE_TYPE* val_t) {
    RankOneUpdate_dev(col_ptr, row_idx, val, u, lambda, v);
}

/**
    Update vector u
**/
__kernel void RankOneUpdate_DUAL_kernel_u(const unsigned cols,
                                          __global const unsigned* col_ptr,
                                          __global const unsigned* row_idx,
                                          __global VALUE_TYPE* val,
                                          __global VALUE_TYPE* u,
                                          __global VALUE_TYPE* v,
                                          const VALUE_TYPE lambda,
                                          const unsigned cols_t,
                                          __global const unsigned* row_ptr,
                                          __global const unsigned* col_idx,
                                          __global VALUE_TYPE* val_t) {
    RankOneUpdate_dev(row_ptr, col_idx, val_t, v, lambda, u);
}

static void UpdateRating_dev(const unsigned cols,
                             __global const unsigned* col_ptr,
                             __global const unsigned* row_idx,
                             __global VALUE_TYPE* val,
                             __global VALUE_TYPE* u,
                             __global VALUE_TYPE* v,
                             const int isAdd) {

//  unsigned ii = get_global_id(0);
//  unsigned jj = get_global_size(0);
    unsigned ss = get_local_id(0);
    unsigned gg = get_local_size(0);
    unsigned aa = get_group_id(0);
//  unsigned dd = get_num_groups(0);

    if (isAdd == 1) // +
    {
        size_t i = aa;
        VALUE_TYPE Htc = v[i];
        unsigned nnz = col_ptr[i + 1] - col_ptr[i];
        size_t bidx = col_ptr[i];
        if (nnz <= WG_SIZE && ss < nnz) {
            size_t idx = bidx + ss;
            unsigned rIdx = row_idx[idx];
            val[idx] += u[rIdx] * Htc;
        } else {
            for (unsigned iter = ss; iter < nnz; iter += gg) {
                size_t idx = bidx + iter;
                unsigned rIdx = row_idx[idx];
                val[idx] += u[rIdx] * Htc;
            }
        }
    } else // -
    {
        size_t i = aa;
        VALUE_TYPE Htc = v[i];
        unsigned nnz = col_ptr[i + 1] - col_ptr[i];
        size_t bidx = col_ptr[i];
        if (nnz <= WG_SIZE && ss < nnz) {
            size_t idx = bidx + ss;
            unsigned rIdx = row_idx[idx];
            val[idx] -= u[rIdx] * Htc;
        } else {
            for (unsigned iter = ss; iter < nnz; iter += gg) {
                size_t idx = bidx + iter;
                unsigned rIdx = row_idx[idx];
                val[idx] -= u[rIdx] * Htc;
            }
        }

    }

}

/**
    Update the rating matrix in the CSC format (value: +)
**/
__kernel void UpdateRating_DUAL_kernel_NoLoss_c(const unsigned cols,
                                                __global const unsigned* col_ptr,
                                                __global const unsigned* row_idx,
                                                __global VALUE_TYPE* val,
                                                __global VALUE_TYPE* Wt_vec_t,
                                                __global VALUE_TYPE* Ht_vec_t,
                                                const unsigned rows,
                                                __global const unsigned* row_ptr,
                                                __global const unsigned* col_idx,
                                                __global VALUE_TYPE* val_t) {
    UpdateRating_dev(cols, col_ptr, row_idx, val, Wt_vec_t, Ht_vec_t, 1);
}

/**
    Update the rating matrix in the CSR format (value_t: +)
**/
__kernel void UpdateRating_DUAL_kernel_NoLoss_r(const unsigned cols,
                                                __global const unsigned* col_ptr,
                                                __global const unsigned* row_idx,
                                                __global VALUE_TYPE* val,
                                                __global VALUE_TYPE* Wt_vec_t,
                                                __global VALUE_TYPE* Ht_vec_t,
                                                const unsigned rows,
                                                __global const unsigned* row_ptr,
                                                __global const unsigned* col_idx,
                                                __global VALUE_TYPE* val_t) {
    UpdateRating_dev(rows, row_ptr, col_idx, val_t, Ht_vec_t, Wt_vec_t, 1);
}


/**
    Update the rating matrix in the CSC format (value: -)
**/
__kernel void UpdateRating_DUAL_kernel_NoLoss_c_(const unsigned cols,
                                                 __global const unsigned* col_ptr,
                                                 __global const unsigned* row_idx,
                                                 __global VALUE_TYPE* val,
                                                 __global VALUE_TYPE* Wt_vec_t,
                                                 __global VALUE_TYPE* Ht_vec_t,
                                                 const unsigned rows,
                                                 __global const unsigned* row_ptr,
                                                 __global const unsigned* col_idx,
                                                 __global VALUE_TYPE* val_t) {
    UpdateRating_dev(cols, col_ptr, row_idx, val, Wt_vec_t, Ht_vec_t, 0);
}

/**
    Update the rating matrix in the CSR format (value_t: -)
**/
__kernel void UpdateRating_DUAL_kernel_NoLoss_r_(const unsigned cols,
                                                 __global const unsigned* col_ptr,
                                                 __global const unsigned* row_idx,
                                                 __global VALUE_TYPE* val,
                                                 __global VALUE_TYPE* Wt_vec_t,
                                                 __global VALUE_TYPE* Ht_vec_t,
                                                 const unsigned rows,
                                                 __global const unsigned* row_ptr,
                                                 __global const unsigned* col_idx,
                                                 __global VALUE_TYPE* val_t) {
    UpdateRating_dev(rows, row_ptr, col_idx, val_t, Ht_vec_t, Wt_vec_t, 0);
}
