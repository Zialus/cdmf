static VALUE_TYPE RankOneUpdate_dev(__global const unsigned* col_ptr,
                                    __global const unsigned* row_idx,
                                    __global const VALUE_TYPE* val,
                                    const unsigned j,
                                    __global const VALUE_TYPE* u_vec_t,
                                    const VALUE_TYPE lambda,
                                    const VALUE_TYPE vj) {
    VALUE_TYPE g = 0, h = lambda;
    if (col_ptr[j + 1] == col_ptr[j]) {
        return 0;
    }
    for (unsigned idx = col_ptr[j]; idx < col_ptr[j + 1]; ++idx) {
        unsigned i = row_idx[idx];
        g += u_vec_t[i] * val[idx];
        h += u_vec_t[i] * u_vec_t[i];
    }
    VALUE_TYPE newvj = g / h;
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
                                          __global const unsigned* col_ptr_t,
                                          __global const unsigned* row_idx_t,
                                          __global VALUE_TYPE* val_t) {
    unsigned ii = get_global_id(0);

    size_t c = ii;
    if (c < cols) {
        v[c] = RankOneUpdate_dev(col_ptr, row_idx, val, c, u, lambda * (col_ptr[c + 1] - col_ptr[c]), v[c]);
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
                                          __global const unsigned* col_ptr_t,
                                          __global const unsigned* row_idx_t,
                                          __global VALUE_TYPE* val_t) {
    unsigned ii = get_global_id(0);

    size_t c = ii;
    if (c < cols_t) {
        u[c] = RankOneUpdate_dev(col_ptr_t, row_idx_t, val_t, c, v, lambda * (col_ptr_t[c + 1] - col_ptr_t[c]), u[c]);
    }
}


static void UpdateRating_dev(const unsigned cols,
                             __global const unsigned* col_ptr,
                             __global const unsigned* row_idx,
                             __global VALUE_TYPE* val,
                             __global VALUE_TYPE* u,
                             __global VALUE_TYPE* v,
                             const int isAdd) {

    unsigned ii = get_global_id(0);
//    unsigned jj = get_global_size(0);
//    unsigned ss = get_local_id(0);
//    unsigned gg = get_local_size(0);
//    unsigned aa = get_group_id(0);
//    unsigned dd = get_num_groups(0);

    if (isAdd == 1) // +
    {
        size_t i = ii;
        if (i < cols) {
            VALUE_TYPE Htc = v[i];
            for (size_t idx = col_ptr[i]; idx < col_ptr[i + 1]; ++idx) {
                val[idx] += u[row_idx[idx]] * Htc;
            }
        }
    } else  // -
    {
        size_t i = ii;
        if (i < cols) {
            VALUE_TYPE Htc = v[i];
            for (size_t idx = col_ptr[i]; idx < col_ptr[i + 1]; ++idx) {
                val[idx] -= u[row_idx[idx]] * Htc;
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
                                                const unsigned cols_t,
                                                __global const unsigned* col_ptr_t,
                                                __global const unsigned* row_idx_t,
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
                                                const unsigned cols_t,
                                                __global const unsigned* col_ptr_t,
                                                __global const unsigned* row_idx_t,
                                                __global VALUE_TYPE* val_t) {
    UpdateRating_dev(cols_t, col_ptr_t, row_idx_t, val_t, Ht_vec_t, Wt_vec_t, 1);
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
                                                 const unsigned cols_t,
                                                 __global const unsigned* col_ptr_t,
                                                 __global const unsigned* row_idx_t,
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
                                                 const unsigned cols_t,
                                                 __global const unsigned* col_ptr_t,
                                                 __global const unsigned* row_idx_t,
                                                 __global VALUE_TYPE* val_t) {
    UpdateRating_dev(cols_t, col_ptr_t, row_idx_t, val_t, Ht_vec_t, Wt_vec_t, 0);
}
