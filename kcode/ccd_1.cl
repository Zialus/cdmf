__kernel void GPU_rmse(__global unsigned const* test_row,
                       __global unsigned const* test_col,
                       __global VALUE_TYPE const* test_val,
                       __global VALUE_TYPE* pred_v,
                       __global VALUE_TYPE* rmse,
                       __global VALUE_TYPE const* W,
                       __global VALUE_TYPE const* H,
                       const unsigned nnz,
                       const unsigned k,
                       const unsigned rows,
                       const unsigned cols) {
    size_t global_id = get_global_id(0);
    size_t global_size = get_global_size(0);
//    size_t local_id = get_local_id(0);
//    size_t group_size = get_local_size(0);
//    size_t global_group_id = get_group_id(0);
//    size_t num_group = get_num_groups(0);

//    size_t c = global_id;
//    if (c < nnz) {
    for (size_t c = global_id; c < nnz; c += global_size) {
        for (unsigned t = 0; t < k; t++) {
            unsigned i = test_row[c];
            unsigned j = test_col[c];
//            pred_v[c] += W[i * k + t] * H[j * k + t]; //W[i][t] * H[j][t];
            pred_v[c] += W[t * rows + i] * H[t * cols + j]; //W[i][t] * H[j][t];
        }

        rmse[c] = (pred_v[c] - test_val[c]) * (pred_v[c] - test_val[c]);
    }

//    for (size_t stride = group_size / 2; stride > 0; stride /= 2) {
//        barrier(CLK_LOCAL_MEM_FENCE);
//        if (local_id < stride) {
//            rmse[local_id] += rmse[local_id + stride];
//        }
//    }
}

inline VALUE_TYPE RankOneUpdate_dev(__global const unsigned* col_ptr,
                                    __global const unsigned* row_idx,
                                    __global const VALUE_TYPE* val,
                                    const size_t j,
                                    __global const VALUE_TYPE* u_vec_t,
                                    const VALUE_TYPE lambda) {
    VALUE_TYPE g = 0, h = 0;

    if (col_ptr[j + 1] == col_ptr[j]) {
        return 0;
    }
    for (unsigned idx = col_ptr[j]; idx < col_ptr[j + 1]; ++idx) {
        unsigned i = row_idx[idx];
        g += u_vec_t[i] * val[idx];
        h += u_vec_t[i] * u_vec_t[i];
    }
    VALUE_TYPE newvj = g / (h + lambda);
    return newvj;
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
    size_t global_id = get_global_id(0);
    size_t global_size = get_global_size(0);

    for (size_t c = global_id; c < cols; c += global_size) {
        v[c] = RankOneUpdate_dev(col_ptr, row_idx, val, c, u, lambda * (col_ptr[c + 1] - col_ptr[c]));
    }

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
    size_t global_id = get_global_id(0);
    size_t global_size = get_global_size(0);

    for (size_t c = global_id; c < cols_t; c += global_size) {
        u[c] = RankOneUpdate_dev(row_ptr, col_idx, val_t, c, v, lambda * (row_ptr[c + 1] - row_ptr[c]));
    }
}

inline void UpdateRating_dev(const unsigned cols,
                             __global const unsigned* col_ptr,
                             __global const unsigned* row_idx,
                             __global VALUE_TYPE* val,
                             __global VALUE_TYPE* u,
                             __global VALUE_TYPE* v,
                             const bool isAdd) {
    size_t global_id = get_global_id(0);
    size_t global_size = get_global_size(0);

    for (size_t i = global_id; i < cols; i += global_size) {
        VALUE_TYPE Htc = v[i];
        for (size_t idx = col_ptr[i]; idx < col_ptr[i + 1]; ++idx) {
            unsigned rIdx = row_idx[idx];
            if (isAdd) {    /// +
                val[idx] += u[rIdx] * Htc;
            } else {        /// -
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
