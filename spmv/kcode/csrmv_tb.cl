#define WG_HALF WG_SIZE/2
__kernel void csrmv_tb(__global const unsigned int *col_ptr,
		__global const unsigned int *row_idx,
		__global const VALUE_TYPE *val,
		__global const VALUE_TYPE * u_vec_t,
		__global VALUE_TYPE* v)
{
	unsigned int ii = get_global_id(0);
	unsigned int jj = get_global_size(0);
	unsigned int ss = get_local_id(0);
	unsigned int gg = get_local_size(0);
	unsigned int aa = get_group_id(0);
	unsigned int dd = get_num_groups(0);

	__local VALUE_TYPE g[WG_SIZE];
	g[ss]=0;
	size_t j = aa;
	if ((col_ptr[j + 1] == col_ptr[j]) && (ss == 0))
	{
		v[j] = 0.0f;
	}
	for (unsigned idx = col_ptr[j]+ss; idx <col_ptr[j + 1]; idx+=gg)
	{	
		int i = row_idx[idx];
		g[ss] += u_vec_t[i] * val[idx];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for(unsigned i = WG_HALF;i > 0; i = i/2)
	{
		if(ss<i)
		{
			g[ss] += g[ss+i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	v[j] = g[0];
	return ;
}
