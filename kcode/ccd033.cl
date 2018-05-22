#define WG_HALF WG_SIZE/2

void UV(const unsigned int cols,
		__global const unsigned int *col_ptr,
		__global VALUE_TYPE * Ht,
		__global VALUE_TYPE * Hb,
		__global VALUE_TYPE * H,
		const VALUE_TYPE lambda)
{
	unsigned int i = get_global_id(0);
	//for(unsigned i=0; i<cols; i++)
	//{
	if(i < cols)
	{
	VALUE_TYPE t = Ht[i];
	VALUE_TYPE b = Hb[i];
	VALUE_TYPE l = lambda * (col_ptr[i+1] - col_ptr[i]);
	VALUE_TYPE s = t/(l + b);
	H[i] = s;
	Ht[i] = 0;
	Hb[i] = 0;								
	}
	return ;
	//}
}
__kernel
void CALV(const unsigned int cols,
		__global const unsigned int *col_ptr,
		__global VALUE_TYPE * Ht,
		__global VALUE_TYPE * Hb,
		__global VALUE_TYPE * H,
		const VALUE_TYPE lambda)
{
	UV(cols, col_ptr, Ht, Hb, H, lambda);
	return ;
}

__kernel
void CALU(const unsigned int rows,
		__global const unsigned int *row_ptr,
		__global VALUE_TYPE * Wt,
		__global VALUE_TYPE * Wb,
		__global VALUE_TYPE * W,
		const VALUE_TYPE lambda)
{
	UV(rows, row_ptr, Wt, Wb, W, lambda);
	return ;
}

void RankOneUpdate_dev(__global const unsigned int *col_ptr,
		__global const unsigned int *row_idx,
		__global const VALUE_TYPE *val,
		__global const VALUE_TYPE * u_vec_t,
		const VALUE_TYPE lambda, 
		__global VALUE_TYPE* v)
{
	unsigned int ii = get_global_id(0);
	unsigned int jj = get_global_size(0);
	unsigned int ss = get_local_id(0);
	unsigned int gg = get_local_size(0);
	unsigned int aa = get_group_id(0);
	unsigned int dd = get_num_groups(0);

	__local VALUE_TYPE g[WG_SIZE],h[WG_SIZE];
	g[ss]=0, h[ss]=0;
	size_t j = aa;
	VALUE_TYPE nlambda = lambda*(col_ptr[j + 1] - col_ptr[j]);
	if ((col_ptr[j + 1] == col_ptr[j]) && (ss == 0))
	{
		v[j] = 0.0f;
	}
	for (unsigned idx = col_ptr[j]+ss; idx <col_ptr[j + 1]; idx+=gg)
	{	
		int i = row_idx[idx];
		g[ss] += u_vec_t[i] * val[idx];
		h[ss] += u_vec_t[i] * u_vec_t[i];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for(unsigned i = WG_HALF;i > 0; i = i/2)
	{
		if(ss<i)
		{
			g[ss] += g[ss+i];
			h[ss] += h[ss+i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	v[j] = g[0]/(h[0]+nlambda);
	return ;
}

/**
	Update vector v
**/
__kernel void RankOneUpdate_DUAL_kernel_v( const unsigned int cols,
		__global const unsigned int *col_ptr,
		__global const unsigned int *row_idx,
		__global VALUE_TYPE *val,
		__global VALUE_TYPE *u,
		__global VALUE_TYPE *v,
		const VALUE_TYPE lambda,
		const unsigned int cols_t,
		__global const unsigned int *row_ptr,
		__global const unsigned int *col_idx,
		__global VALUE_TYPE *val_t)
{
	RankOneUpdate_dev(col_ptr, row_idx, val, u, lambda, v);
	return ;
}

/**
	Update vector u
**/
__kernel void RankOneUpdate_DUAL_kernel_u( const unsigned int cols,
		__global const unsigned int *col_ptr,
		__global const unsigned int *row_idx,
		__global VALUE_TYPE *val,
		__global VALUE_TYPE *u,
		__global VALUE_TYPE *v,
		const VALUE_TYPE lambda,
		const unsigned int cols_t,
		__global const unsigned int *row_ptr,
		__global const unsigned int *col_idx,
		__global VALUE_TYPE *val_t)
{
	RankOneUpdate_dev(row_ptr, col_idx, val_t, v, lambda, u);
	return ;
}

void UpdateRating_dev(const unsigned int cols, \
		__global const unsigned int *col_ptr,	\
		__global const unsigned int *row_idx,	\
		__global VALUE_TYPE *val,	\
		__global VALUE_TYPE *u,	\
		__global VALUE_TYPE *v,	\
		const int isAdd)
{

	unsigned int ii = get_global_id(0);
	unsigned int jj = get_global_size(0);
	unsigned int ss = get_local_id(0);
	unsigned int gg = get_local_size(0);
	unsigned int aa = get_group_id(0);
	unsigned int dd = get_num_groups(0);

	if(isAdd == 1) // +
	{
		size_t i = aa;
		VALUE_TYPE Htc = v[i];
		unsigned nnz = col_ptr[i+1] - col_ptr[i];
		size_t bidx = col_ptr[i];
		if(nnz <= WG_SIZE && ss < nnz)
		{
			size_t idx = bidx + ss;
			unsigned rIdx = row_idx[idx];
			val[idx] += u[rIdx] * Htc;		
		}
		else
		{
			for(unsigned iter=ss; iter<nnz; iter+=gg)
			{
				size_t idx = bidx + iter;
				unsigned rIdx = row_idx[idx];
				val[idx] += u[rIdx] * Htc;
			}		
		}
	}
	else // -
	{
		size_t i = aa;
		VALUE_TYPE Htc = v[i];
		unsigned nnz = col_ptr[i+1] - col_ptr[i];
		size_t bidx = col_ptr[i];
		if(nnz <= WG_SIZE && ss < nnz)
		{
			size_t idx = bidx + ss;
			unsigned rIdx = row_idx[idx];
			val[idx] -= u[rIdx] * Htc;		
		}
		else
		{
			for(unsigned iter=ss; iter<nnz; iter+=gg)
			{
				size_t idx = bidx + iter;
				unsigned rIdx = row_idx[idx];
				val[idx] -= u[rIdx] * Htc;
			}		
		}

	}	

	return ;
}

/**
	Update the rating matrix in the CSC format (value: +)
**/
__kernel void UpdateRating_DUAL_kernel_NoLoss_c(const unsigned int cols,
		__global const unsigned int *col_ptr,
		__global const unsigned int *row_idx,
		__global VALUE_TYPE *val,
		__global VALUE_TYPE * Wt_vec_t,
		__global VALUE_TYPE * Ht_vec_t,
		const unsigned int rows,
		__global const unsigned int *row_ptr,
		__global const unsigned int *col_idx,
		__global VALUE_TYPE *val_t)
{
	UpdateRating_dev(cols, col_ptr, row_idx, val, Wt_vec_t, Ht_vec_t, 1);
	return ;
}

/**
	Update the rating matrix in the CSR format (value_t: +)
**/
__kernel void UpdateRating_DUAL_kernel_NoLoss_r(const unsigned int cols,
		__global const unsigned int *col_ptr,
		__global const unsigned int *row_idx,
		__global VALUE_TYPE *val,
		__global VALUE_TYPE * Wt_vec_t,
		__global VALUE_TYPE * Ht_vec_t,
		const unsigned int rows,
		__global const unsigned int *row_ptr,
		__global const unsigned int *col_idx,
		__global VALUE_TYPE *val_t)
{
	UpdateRating_dev(rows, row_ptr, col_idx, val_t, Ht_vec_t, Wt_vec_t, 1);
	return ;
}


/**
	Update the rating matrix in the CSC format (value: -)
**/
__kernel void UpdateRating_DUAL_kernel_NoLoss_c_(const unsigned int cols,
		__global const unsigned int *col_ptr,
		__global const unsigned int *row_idx,
		__global VALUE_TYPE *val,
		__global VALUE_TYPE * Wt_vec_t,
		__global VALUE_TYPE * Ht_vec_t,
		const unsigned int rows,
		__global const unsigned int *row_ptr,
		__global const unsigned int *col_idx,
		__global VALUE_TYPE *val_t)
{
	UpdateRating_dev(cols, col_ptr, row_idx, val, Wt_vec_t, Ht_vec_t, 0);
	return ;
}

/**
	Update the rating matrix in the CSR format (value_t: -)
**/
__kernel void UpdateRating_DUAL_kernel_NoLoss_r_(const unsigned int cols,
		__global const unsigned int *col_ptr,
		__global const unsigned int *row_idx,
		__global VALUE_TYPE *val,
		__global VALUE_TYPE * Wt_vec_t,
		__global VALUE_TYPE * Ht_vec_t,
		const unsigned int rows,
		__global const unsigned int *row_ptr,
		__global const unsigned int *col_idx,
		__global VALUE_TYPE *val_t)
{
	UpdateRating_dev(rows, row_ptr, col_idx, val_t, Ht_vec_t, Wt_vec_t, 0);
	return ;
}
