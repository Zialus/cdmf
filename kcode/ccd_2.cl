__kernel void RankOneUpdate_dev(__global const unsigned* col_ptr,
                                __global const unsigned* row_idx,
                                __global const VALUE_TYPE* val,
                                __global const VALUE_TYPE* u_vec_t,
                                const VALUE_TYPE lambda,
                                __global VALUE_TYPE* v) {
    size_t local_id = get_local_id(0);
    size_t group_size = get_local_size(0);
    size_t group_id = get_group_id(0);

    __local VALUE_TYPE g[WG_SIZE], h[WG_SIZE];
    g[local_id] = 0;
    h[local_id] = 0;
    size_t j = group_id;
    VALUE_TYPE nlambda = lambda * (col_ptr[j + 1] - col_ptr[j]);
    if ((col_ptr[j + 1] == col_ptr[j]) && (local_id == 0)) {
        v[j] = 0.0;
    }
    for (size_t idx = col_ptr[j] + local_id; idx < col_ptr[j + 1]; idx += group_size) {
        unsigned i = row_idx[idx];
        g[local_id] += u_vec_t[i] * val[idx];
        h[local_id] += u_vec_t[i] * u_vec_t[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned i = WG_SIZE / 2; i > 0; i = i / 2) {
        if (local_id < i) {
            g[local_id] += g[local_id + i];
            h[local_id] += h[local_id + i];
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
                             const bool isAdd) {
    size_t local_id = get_local_id(0);
    size_t group_size = get_local_size(0);
    size_t group_id = get_group_id(0);

    size_t i = group_id;
    VALUE_TYPE Htc = v[i];
    unsigned nnz = col_ptr[i + 1] - col_ptr[i];

    if (nnz <= WG_SIZE && local_id < nnz) {
        size_t idx = col_ptr[i] + local_id;
        unsigned rIdx = row_idx[idx];
        if (isAdd){     /// +
            val[idx] += u[rIdx] * Htc;
        } else {        /// -
            val[idx] -= u[rIdx] * Htc;
        }
    } else {
        for (size_t idx = col_ptr[i] + local_id; idx < col_ptr[i + 1]; idx += group_size) {
            unsigned rIdx = row_idx[idx];
            if (isAdd) {     /// +
                val[idx] += u[rIdx] * Htc;
            } else {         /// -
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
    UpdateRating_dev(cols, col_ptr, row_idx, val, Wt_vec_t, Ht_vec_t, true);
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
    UpdateRating_dev(rows, row_ptr, col_idx, val_t, Ht_vec_t, Wt_vec_t, true);
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
    UpdateRating_dev(cols, col_ptr, row_idx, val, Wt_vec_t, Ht_vec_t, false);
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
    UpdateRating_dev(rows, row_ptr, col_idx, val_t, Ht_vec_t, Wt_vec_t, false);
}
