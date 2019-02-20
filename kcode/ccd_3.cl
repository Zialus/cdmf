static void UV(const unsigned cols,
               __global const unsigned* col_ptr,
               __global VALUE_TYPE* Ht,
               __global VALUE_TYPE* Hb,
               __global VALUE_TYPE* H,
               const VALUE_TYPE lambda) {
    size_t i = get_global_id(0);

    if (i < cols) {
        VALUE_TYPE t = Ht[i];
        VALUE_TYPE b = Hb[i];
        VALUE_TYPE l = lambda * (col_ptr[i + 1] - col_ptr[i]);
        VALUE_TYPE s = t / (l + b);
        H[i] = s;
        Ht[i] = 0;
        Hb[i] = 0;
    }
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
    size_t bidx = col_ptr[i];
    if (nnz <= WG_SIZE && local_id < nnz) {
        size_t idx = bidx + local_id;
        unsigned rIdx = row_idx[idx];
        if (isAdd){     /// +
            val[idx] += u[rIdx] * Htc;
        } else {        /// -
            val[idx] -= u[rIdx] * Htc;
        }
    } else {
        for (size_t iter = local_id; iter < nnz; iter += group_size) {
            size_t idx = bidx + iter;
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
