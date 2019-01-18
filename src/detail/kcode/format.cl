//#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define  ANONYMOUSLIB_CSR5_OMEGA _REPLACE_ANONYMOUSLIB_CSR5_OMEGA_SEGMENT_
#define  ANONYMOUSLIB_CSR5_SIGMA _REPLACE_ANONYMOUSLIB_CSR5_SIGMA_SEGMENT_
#define  ANONYMOUSLIB_THREAD_GROUP _REPLACE_ANONYMOUSLIB_THREAD_GROUP_SEGMENT_
typedef _REPLACE_ANONYMOUSLIB_CSR5_INDEX_TYPE_SEGMENT_   iT;
typedef _REPLACE_ANONYMOUSLIB_CSR5_UNSIGNED_INDEX_TYPE_SEGMENT_   uiT;
typedef _REPLACE_ANONYMOUSLIB_CSR5_VALUE_TYPE_SEGMENT_   vT;

inline
iT binary_search_right_boundary_kernel(__global const iT *d_row_pointer,
                                       const iT  key_input,
                                       const iT  size)
{
    iT start = 0;
    iT stop  = size - 1;
    iT median;
    iT key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;
        key_median = d_row_pointer[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start;
}


inline
void scan_32_plus1(__local volatile int *s_scan,
                   const int      lane_id)
{
    int ai, bi;
    int baseai = 1 + 2 * lane_id;
    int basebi = baseai + 1;
    int temp;

    if (lane_id < 16) { ai =  baseai - 1;  bi =  basebi - 1;   s_scan[bi] += s_scan[ai]; } //barrier(CLK_LOCAL_MEM_FENCE);
    if (lane_id < 8)  { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (lane_id < 4)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (lane_id < 2)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (lane_id == 0) { s_scan[31] += s_scan[15]; s_scan[32] = s_scan[31]; s_scan[31] = 0; temp = s_scan[15]; s_scan[15] = 0; s_scan[31] += temp; }
    if (lane_id < 2)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (lane_id < 4)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (lane_id < 8)  { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (lane_id < 16) { ai =  baseai - 1;  bi =  basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;} //barrier(CLK_LOCAL_MEM_FENCE);
}

/*inline
void scan_64_plus1(__local volatile int *s_scan,
                   const int      lane_id)
{
    int ai, bi;
    int baseai = 1 + 2 * lane_id;
    int basebi = baseai + 1;
    int temp;

    if (lane_id < 32) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    if (lane_id < 16) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (lane_id < 8)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (lane_id < 4)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (lane_id < 2)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (lane_id == 0) { s_scan[63] += s_scan[31]; s_scan[64] = s_scan[63]; s_scan[63] = 0; temp = s_scan[31]; s_scan[31] = 0; s_scan[63] += temp; }
    if (lane_id < 2)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (lane_id < 4)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (lane_id < 8)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (lane_id < 16) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (lane_id < 32) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}*/

inline
void scan_256_plus1(__local volatile int *s_scan,
              const int      lane_id)
{
    int ai, bi;
    int baseai = 1 + 2 * lane_id;
    int basebi = baseai + 1;
    int temp;

    if (lane_id < 128) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane_id < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; } barrier(CLK_LOCAL_MEM_FENCE);
    if (lane_id < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   s_scan[bi] += s_scan[ai]; } //barrier(CLK_LOCAL_MEM_FENCE);
    if (lane_id < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   s_scan[bi] += s_scan[ai]; } //barrier(CLK_LOCAL_MEM_FENCE);
    if (lane_id < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (lane_id < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (lane_id < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (lane_id == 0) { s_scan[255] += s_scan[127]; s_scan[256] = s_scan[255]; s_scan[255] = 0; temp = s_scan[127]; s_scan[127] = 0; s_scan[255] += temp; }
    if (lane_id < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (lane_id < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (lane_id < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (lane_id < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;} //barrier(CLK_LOCAL_MEM_FENCE);
    if (lane_id < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;} //barrier(CLK_LOCAL_MEM_FENCE);
    if (lane_id < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;} barrier(CLK_LOCAL_MEM_FENCE);
    if (lane_id < 128) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; } barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel
void warmup_kernel(__global int *d_scan)
{
    volatile __local int s_scan[ANONYMOUSLIB_CSR5_OMEGA+1];
    s_scan[get_local_id(0)] = 1;
    int local_id = get_local_id(0);
    scan_32_plus1(s_scan, local_id);
    if(!get_group_id(0))
        d_scan[get_local_id(0)] = s_scan[get_local_id(0)];
}

__kernel
void generate_partition_pointer_s1_kernel(__global const iT     *d_row_pointer,
                                          __global uiT          *d_partition_pointer,
                                          const int     sigma,
                                          const iT      p,
                                          const iT      m,
                                          const iT      nnz)
{
    // global thread id
    iT global_id = get_global_id(0);

    // compute partition boundaries by partition of size sigma * omega
    iT boundary = global_id * sigma * ANONYMOUSLIB_CSR5_OMEGA;

    // clamp partition boundaries to [0, nnz]
    boundary = boundary > nnz ? nnz : boundary;

    // binary search
    if (global_id <= p)
        d_partition_pointer[global_id] = binary_search_right_boundary_kernel(d_row_pointer, boundary, m + 1) - 1;
}

__kernel
void generate_partition_pointer_s2_kernel(__global const iT   *d_row_pointer,
                                          __global uiT        *d_partition_pointer)
{
    const iT group_id = get_group_id(0);
    const int local_id = get_local_id(0);
    const iT local_size = get_local_size(0);

    volatile __local int s_dirty[1];

    if (!local_id)
        s_dirty[0] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    uiT start = d_partition_pointer[group_id];
    uiT stop  = d_partition_pointer[group_id+1];

    start = (start << 1) >> 1;
    stop  = (stop << 1) >> 1;

    if(start == stop)
        return;

    uiT num_row_in_partition = stop + 1 - start;
    int loop = ceil((float)num_row_in_partition / (float)local_size);
    iT row_idx, row_off_l, row_off_r;

    for (int i = 0; i < loop; i++)
    {
        row_idx = i * local_size + start + local_id;

        if (row_idx < stop)
        {
            row_off_l = d_row_pointer[row_idx];
            row_off_r = d_row_pointer[row_idx+1];

            if (row_off_l == row_off_r)
                s_dirty[0] = 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (s_dirty[0])
            break;
    }

    if (s_dirty[0] && !local_id)
    {
        start |= sizeof(uiT) == 4 ? 0x80000000 : 0x8000000000000000;
        d_partition_pointer[group_id] = start;
    }
}

__kernel
void generate_partition_descriptor_s0_kernel(__global uiT         *d_partition_descriptor,
                                             const int    num_packet)
{
    const int local_id = get_local_id(0);

    for (int i = 0; i < num_packet; i++)
        d_partition_descriptor[get_group_id(0) * ANONYMOUSLIB_CSR5_OMEGA * num_packet + i * ANONYMOUSLIB_CSR5_OMEGA + local_id] = 0;
}

__kernel
void generate_partition_descriptor_s1_kernel(__global const iT    *d_row_pointer,
                                             __global uiT         *d_partition_descriptor,
                                             const iT     m,
                                             const int    sigma,
                                             const int    bit_all_offset,
                                             const int    num_packet)
{
    const iT global_id = get_global_id(0);

    if (global_id < m)
    {
        const iT row_offset = d_row_pointer[global_id];

        const iT  gx    = row_offset / sigma;

        const iT  lx    = gx % ANONYMOUSLIB_CSR5_OMEGA;
        const iT  pid   = gx / ANONYMOUSLIB_CSR5_OMEGA;

        const int glid  = row_offset % sigma + bit_all_offset;
        const int llid  = glid % 32;
        const int ly    = glid / 32;

        const uiT val = 0x1 << (31 - llid);

        const int location = pid * ANONYMOUSLIB_CSR5_OMEGA * num_packet + ly * ANONYMOUSLIB_CSR5_OMEGA + lx;

        atomic_or(&d_partition_descriptor[location], val);
    }
}

__kernel
void generate_partition_descriptor_s2_kernel(__global const uiT    *d_partition_pointer,
                                             __global uiT          *d_partition_descriptor,
                                             __global iT           *d_partition_descriptor_offset_pointer,
                                             const int     sigma,
                                             const int     num_packet,
                                             const int     bit_y_offset,
                                             const int     bit_scansum_offset,
                                             const int     p)
{
    const int local_id = get_local_id(0);

    const int lane_id = local_id % ANONYMOUSLIB_CSR5_OMEGA;
    const int bunch_id = local_id / ANONYMOUSLIB_CSR5_OMEGA;
    const int par_id = get_global_id(0) / ANONYMOUSLIB_CSR5_OMEGA;

    volatile __local uiT s_row_start_stop[ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1];

    if (local_id < ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1)
        s_row_start_stop[local_id] = d_partition_pointer[par_id + local_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    uiT row_start       = s_row_start_stop[bunch_id];
    bool with_empty_rows = (row_start >> 31) & 0x1;
    row_start          &= 0x7FFFFFFF; //( << 1) >> 1
    const iT row_stop   = s_row_start_stop[bunch_id + 1] & 0x7FFFFFFF;

    // if this is fast track partition, do not generate its partition_descriptor
    if (row_start == row_stop)
    {
        if (!lane_id)
            d_partition_descriptor_offset_pointer[par_id] = 0;
        return;
    }
    int y_offset = 0;
    int scansum_offset = 0;

    int start = 0, stop = 0, segn = 0;
    bool present = 0;
    uiT bitflag = 0;

    volatile __local int s_segn_scan[(ANONYMOUSLIB_CSR5_OMEGA + 1) * ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA];
    volatile __local int s_present[(ANONYMOUSLIB_CSR5_OMEGA + 1) * ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA];

    const int bit_all_offset = bit_y_offset + bit_scansum_offset;

    present |= !lane_id;

    // extract the first bit-flag packet
    int ly = 0;
    uiT first_packet = d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet + lane_id];
    bitflag = (first_packet << bit_all_offset) | ((uiT)present << 31);
    start = !((bitflag >> 31) & 0x1);
    present |= (bitflag >> 31) & 0x1;

    #pragma unroll
    for (int i = 1; i < ANONYMOUSLIB_CSR5_SIGMA; i++)
    {
        if ((!ly && i == 32 - bit_all_offset) || (ly && (i - (32 - bit_all_offset)) % 32 == 0))
        {
            ly++;
            bitflag = d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet + ly * ANONYMOUSLIB_CSR5_OMEGA + lane_id];
        }
        const int norm_i = !ly ? i : i - (32 - bit_all_offset);
        stop += (bitflag >> (31 - norm_i % 32) ) & 0x1;
        present |= (bitflag >> (31 - norm_i % 32)) & 0x1;
    }

    // compute y_offset for all partitions
    segn = stop - start + present;
    segn = segn > 0 ? segn : 0;

    s_segn_scan[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1) + lane_id] = segn;
    scan_32_plus1(&s_segn_scan[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1)], lane_id); // exclusive scan
    if (!lane_id && !with_empty_rows)
        d_partition_descriptor_offset_pointer[par_id] = 0;
    if (!lane_id && with_empty_rows)
    {
        d_partition_descriptor_offset_pointer[par_id] = s_segn_scan[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1) + ANONYMOUSLIB_CSR5_OMEGA]; // the total number of segments in this partition
        //d_partition_descriptor_offset_pointer[p] = 1;
    }
    y_offset = s_segn_scan[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1) + lane_id];

    // compute scansum_offset
    s_present[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1) + lane_id] = present;
    int next1 = lane_id + 1;
    if (present)
    {
        while (!s_present[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1) + next1] && next1 < ANONYMOUSLIB_CSR5_OMEGA)
        {
            scansum_offset++;
            next1++;
        }
    }

    y_offset = lane_id ? y_offset - 1 : 0;

    first_packet |= y_offset << (32 - bit_y_offset);
    first_packet |= scansum_offset << (32 - bit_all_offset);

    d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet + lane_id] = first_packet;
}

__kernel
void generate_partition_descriptor_s3_kernel(__global iT           *d_partition_descriptor_offset_pointer,
                                             const int     p)
{
    const int local_id = get_local_id(0);
    const int local_size = get_local_size(0);

    iT sum = 0;
    volatile __local iT s_partition_descriptor_offset_pointer[256 + 1];

    int loop = ceil((float)p / (float)local_size);

    for (int i = 0; i < loop; i++)
    {
        s_partition_descriptor_offset_pointer[local_id] = (local_id + i * local_size < p) ? d_partition_descriptor_offset_pointer[local_id + i * local_size] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        scan_256_plus1(s_partition_descriptor_offset_pointer, local_id);
        barrier(CLK_LOCAL_MEM_FENCE);

        s_partition_descriptor_offset_pointer[local_id] += sum;
        if (!local_id)
            s_partition_descriptor_offset_pointer[256] += sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        sum = s_partition_descriptor_offset_pointer[256];

        if (local_id + i * local_size < p + 1)
            d_partition_descriptor_offset_pointer[local_id + i * local_size] = s_partition_descriptor_offset_pointer[local_id];
    }
}

inline
void partition_normal_track_empty_Ologn(__global const iT             *d_row_pointer,
                                        __global const uiT            *d_partition_descriptor,
                                        __global const iT             *d_partition_descriptor_offset_pointer,
                                        __global iT                   *d_partition_descriptor_offset,
                                        const iT              par_id,
                                        const int             lane_id,
                                        const int             bit_y_offset,
                                        const int             bit_scansum_offset,
                                        iT                    row_start,
                                        const iT              row_stop,
                                        const int             c_sigma)
{
    bool local_bit;

    int offset_pointer = d_partition_descriptor_offset_pointer[par_id];

    uiT descriptor = d_partition_descriptor[lane_id];

    int y_offset = descriptor >> (32 - bit_y_offset);
    const int bit_bitflag = 32 - bit_y_offset - bit_scansum_offset;

    // step 1. thread-level seg sum
    // extract the first bit-flag packet
    int ly = 0;
    descriptor = descriptor << (bit_y_offset + bit_scansum_offset);
    descriptor = lane_id ? descriptor : descriptor | 0x80000000;

    local_bit = (descriptor >> 31) & 0x1;

    if (local_bit && lane_id)
    {
        const iT idx = par_id * ANONYMOUSLIB_CSR5_OMEGA * c_sigma + lane_id * c_sigma;
        const iT y_index = binary_search_right_boundary_kernel(&d_row_pointer[row_start+1], idx, row_stop - row_start) - 1;
        d_partition_descriptor_offset[offset_pointer + y_offset] = y_index;

        y_offset++;
    }

    #pragma unroll
    for (int i = 1; i < ANONYMOUSLIB_CSR5_SIGMA; i++)
    {
        if ((!ly && i == bit_bitflag) || (ly && !(31 & (i - bit_bitflag))))
        {
            ly++;
            descriptor = d_partition_descriptor[ly * ANONYMOUSLIB_CSR5_OMEGA + lane_id];
        }
        const int norm_i = 31 & (!ly ? i : i - bit_bitflag);

        local_bit = (descriptor >> (31 - norm_i)) & 0x1;

        if (local_bit)
        {
            const iT idx = par_id * ANONYMOUSLIB_CSR5_OMEGA * c_sigma + lane_id * c_sigma + i;
            const iT y_index = binary_search_right_boundary_kernel(&d_row_pointer[row_start+1], idx, row_stop - row_start) - 1;
            d_partition_descriptor_offset[offset_pointer + y_offset] = y_index;

            y_offset++;
        }
    }
}

inline
void generate_partition_descriptor_offset_partition(__global const iT           *d_row_pointer,
                                                    __global const uiT          *d_partition_pointer,
                                                    __global const uiT          *d_partition_descriptor,
                                                    __global const iT           *d_partition_descriptor_offset_pointer,
                                                    __global iT                 *d_partition_descriptor_offset,
                                                    const iT            par_id,
                                                    const int           lane_id,
                                                    const int           bunch_id,
                                                    const int           bit_y_offset,
                                                    const int           bit_scansum_offset,
                                                    const int           c_sigma,
                                                    volatile __local uiT *s_row_start_stop)
{
    const int local_id    = get_local_id(0);
    uiT row_start, row_stop;

    //volatile __local uiT s_row_start_stop[ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1];
    if (local_id < ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1)
        s_row_start_stop[local_id] = d_partition_pointer[par_id + local_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    row_start = s_row_start_stop[bunch_id];
    row_stop  = s_row_start_stop[bunch_id + 1] & 0x7FFFFFFF;

    if (row_start >> 31) // with empty rows
    {
        row_start &= 0x7FFFFFFF;     //( << 1) >> 1

        partition_normal_track_empty_Ologn
                (d_row_pointer,
                 d_partition_descriptor, d_partition_descriptor_offset_pointer, d_partition_descriptor_offset,
                 par_id, lane_id,
                 bit_y_offset, bit_scansum_offset, row_start, row_stop, c_sigma);
    }
    else // without empty rows
    {
        return;
    }
}

__kernel
void generate_partition_descriptor_offset_kernel(__global const iT           *d_row_pointer,
                                                 __global const uiT          *d_partition_pointer,
                                                 __global const uiT          *d_partition_descriptor,
                                                 __global const iT           *d_partition_descriptor_offset_pointer,
                                                 __global iT                 *d_partition_descriptor_offset,
                                                 const iT            p,
                                                 const int           num_packet,
                                                 const int           bit_y_offset,
                                                 const int           bit_scansum_offset,
                                                 const int           c_sigma)
{
    const int local_id    = get_local_id(0);

    // warp lane id
    const int lane_id = local_id % ANONYMOUSLIB_CSR5_OMEGA;
    // warp global id == par_id
    const iT  par_id =  get_global_id(0) / ANONYMOUSLIB_CSR5_OMEGA;
    const int bunch_id = local_id / ANONYMOUSLIB_CSR5_OMEGA;
    volatile __local uiT s_row_start_stop[ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1];
    if (par_id >= p - 1)
        return;

    generate_partition_descriptor_offset_partition
                (d_row_pointer, d_partition_pointer,
                 &d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet],
                 d_partition_descriptor_offset_pointer, d_partition_descriptor_offset,
                 par_id, lane_id, bunch_id, bit_y_offset, bit_scansum_offset, c_sigma, s_row_start_stop);
}

__kernel
void aosoa_transpose_kernel_smem_iT(__global iT         *d_data,
                                 __global const uiT *d_partition_pointer,
                                 const int R2C) // R2C==true means CSR->CSR5, otherwise CSR5->CSR
{
    __local uiT s_par[2];

    const int local_id = get_local_id(0);

    if (local_id < 2)
        s_par[local_id] = d_partition_pointer[get_group_id(0) + local_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    // if this is fast track partition, do not transpose it
    if (s_par[0] == s_par[1])
        return;

    __local iT s_data[ANONYMOUSLIB_CSR5_SIGMA * (ANONYMOUSLIB_CSR5_OMEGA + 1)];

    // load global data to shared mem
    int idx_y, idx_x;
    for (int idx = local_id; idx < ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA; idx += get_local_size(0))
    {
        if (R2C)
        {
            idx_y = idx % ANONYMOUSLIB_CSR5_SIGMA;
            idx_x = idx / ANONYMOUSLIB_CSR5_SIGMA;
        }
        else
        {
            idx_x = idx % ANONYMOUSLIB_CSR5_OMEGA;
            idx_y = idx / ANONYMOUSLIB_CSR5_OMEGA;
        }

        s_data[idx_y * (ANONYMOUSLIB_CSR5_OMEGA+1) + idx_x] = d_data[get_group_id(0) * ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA + idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // store transposed shared mem data to global
    for (int idx = local_id; idx < ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA; idx += get_local_size(0))
    {
        if (R2C)
        {
            idx_x = idx % ANONYMOUSLIB_CSR5_OMEGA;
            idx_y = idx / ANONYMOUSLIB_CSR5_OMEGA;
        }
        else
        {
            idx_y = idx % ANONYMOUSLIB_CSR5_SIGMA;
            idx_x = idx / ANONYMOUSLIB_CSR5_SIGMA;
        }

        d_data[get_group_id(0) * ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA + idx] = s_data[idx_y * (ANONYMOUSLIB_CSR5_OMEGA+1) + idx_x];
    }
}

__kernel
void aosoa_transpose_kernel_smem_vT(__global vT         *d_data,
                                 __global const uiT *d_partition_pointer,
                                 const int R2C) // R2C==true means CSR->CSR5, otherwise CSR5->CSR
{
    __local uiT s_par[2];

    const int local_id = get_local_id(0);

    if (local_id < 2)
        s_par[local_id] = d_partition_pointer[get_group_id(0) + local_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    // if this is fast track partition, do not transpose it
    if (s_par[0] == s_par[1])
        return;

    __local vT s_data[ANONYMOUSLIB_CSR5_SIGMA * (ANONYMOUSLIB_CSR5_OMEGA + 1)];

    // load global data to shared mem
    int idx_y, idx_x;
    for (int idx = local_id; idx < ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA; idx += get_local_size(0))
    {
        if (R2C)
        {
            idx_y = idx % ANONYMOUSLIB_CSR5_SIGMA;
            idx_x = idx / ANONYMOUSLIB_CSR5_SIGMA;
        }
        else
        {
            idx_x = idx % ANONYMOUSLIB_CSR5_OMEGA;
            idx_y = idx / ANONYMOUSLIB_CSR5_OMEGA;
        }

        s_data[idx_y * (ANONYMOUSLIB_CSR5_OMEGA+1) + idx_x] = d_data[get_group_id(0) * ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA + idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // store transposed shared mem data to global
    for (int idx = local_id; idx < ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA; idx += get_local_size(0))
    {
        if (R2C)
        {
            idx_x = idx % ANONYMOUSLIB_CSR5_OMEGA;
            idx_y = idx / ANONYMOUSLIB_CSR5_OMEGA;
        }
        else
        {
            idx_y = idx % ANONYMOUSLIB_CSR5_SIGMA;
            idx_x = idx / ANONYMOUSLIB_CSR5_SIGMA;
        }

        d_data[get_group_id(0) * ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA + idx] = s_data[idx_y * (ANONYMOUSLIB_CSR5_OMEGA+1) + idx_x];
    }
}
