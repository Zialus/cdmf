float
RankOneUpdate_dev (__global const long *col_ptr,
		__global const unsigned *row_idx,
		__global const float *val, 
		const int j,
		__global const float *u_vec_t, 
		const float lambda,
		const float vj){
	float g = 0, h = lambda;
	if (col_ptr[j + 1] == col_ptr[j])
		return 0;
	for (long idx = col_ptr[j]; idx < col_ptr[j + 1]; ++idx){
		int i = row_idx[idx];
		g += u_vec_t[i] * val[idx];
		h += u_vec_t[i] * u_vec_t[i];
	} 
	float newvj = g / h;
	return newvj;
}

__kernel void
RankOneUpdate_DUAL_kernel (const long cols,
		__global const long *col_ptr,
		__global const unsigned int *row_idx,
		__global float *val,
		__global float *u,
		__global float *v, 
		const float lambda,
		const long cols_t,
		__global const long *col_ptr_t,
		__global const unsigned int *row_idx_t,
		__global float *val_t){
	int ii = get_global_id (0);
	int jj = get_global_size (0);
	for (size_t c = ii; c < cols; c += jj){
		v[c] = 	RankOneUpdate_dev (col_ptr, row_idx, val, c, u, lambda * (col_ptr[c + 1] - col_ptr[c]), v[c]);
		//if (c < 10){
		//	printf ("v[%d]=%f.\n", c, v[c]);
		//}
	}
	for (size_t c = ii; c < cols_t; c += jj){
		u[c] = RankOneUpdate_dev (col_ptr_t, row_idx_t, val_t, c, v, lambda * (col_ptr_t[c + 1] - col_ptr_t[c]), u[c]);
		//if (c < 10){
		//	printf ("u[%d]=%f.\n", c, u[c]);
		//}
	}
}

__kernel void
UpdateRating_DUAL_kernel_NoLoss (const long cols,
		__global const long *col_ptr,
		__global const unsigned int *row_idx,
		__global float *val,
		__global float *Wt_vec_t,
		__global float *Ht_vec_t,
		const long cols_t,
		__global const long *col_ptr_t,
		__global const unsigned int *row_idx_t,
		__global float *val_t){
	int ii = get_global_id (0);
	int jj = get_global_size (0);
	for (size_t i = ii; i < cols; i += jj){
		float Htc = Ht_vec_t[i];
		for (size_t idx = col_ptr[i]; idx < col_ptr[i + 1]; ++idx){
			val[idx] += Wt_vec_t[row_idx[idx]] * Htc;
		}
	}
	for (size_t i = ii; i < cols_t; i += jj){
		float Htc = Wt_vec_t[i];
		for (size_t idx = col_ptr_t[i]; idx < col_ptr_t[i + 1]; ++idx){
			val_t[idx] += Ht_vec_t[row_idx_t[idx]] * Htc;
		}
	}
}

__kernel void
UpdateRating_DUAL_kernel_NoLoss_ (const long cols,
		__global const long *col_ptr,
		__global const unsigned int *row_idx,
		__global float *val,
		__global float *Wt_vec_t,
		__global float *Ht_vec_t,
		const long cols_t,
		__global const long *col_ptr_t,
		__global const unsigned int *row_idx_t,
		__global float *val_t){
	int ii = get_global_id (0);
	int jj = get_global_size (0);
	for (size_t i = ii; i < cols; i += jj){
		float Htc = Ht_vec_t[i];
		for (size_t idx = col_ptr[i]; idx < col_ptr[i + 1]; ++idx){
			val[idx] -= Wt_vec_t[row_idx[idx]] * Htc;
		}
	}
	for (size_t i = ii; i < cols_t; i += jj){
		float Htc = Wt_vec_t[i];
		for (size_t idx = col_ptr_t[i]; idx < col_ptr_t[i + 1]; ++idx){
			val_t[idx] -= Ht_vec_t[row_idx_t[idx]] * Htc;
		}
	}
}


