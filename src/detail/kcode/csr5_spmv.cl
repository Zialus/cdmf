//#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define  ANONYMOUSLIB_CSR5_OMEGA _REPLACE_ANONYMOUSLIB_CSR5_OMEGA_SEGMENT_
#define  ANONYMOUSLIB_CSR5_SIGMA _REPLACE_ANONYMOUSLIB_CSR5_SIGMA_SEGMENT_
#define  ANONYMOUSLIB_THREAD_GROUP _REPLACE_ANONYMOUSLIB_THREAD_GROUP_SEGMENT_
#define  ANONYMOUSLIB_THREAD_BUNCH _REPLACE_ANONYMOUSLIB_THREAD_BUNCH_SEGMENT_
typedef _REPLACE_ANONYMOUSLIB_CSR5_INDEX_TYPE_SEGMENT_   iT;
typedef _REPLACE_ANONYMOUSLIB_CSR5_UNSIGNED_INDEX_TYPE_SEGMENT_   uiT;
typedef _REPLACE_ANONYMOUSLIB_CSR5_VALUE_TYPE_SEGMENT_   vT;

inline
void sum_32(__local volatile  vT *s_sum,
            const int    local_id)
{
    if (local_id < 16) { s_sum[local_id] += s_sum[local_id + 16];} //barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 8) { s_sum[local_id] += s_sum[local_id + 8]; }
    if (local_id < 4) { s_sum[local_id] += s_sum[local_id + 4]; }
    if (local_id < 2) { s_sum[local_id] += s_sum[local_id + 2]; }
    if (local_id < 1) { s_sum[local_id] += s_sum[local_id + 1]; }
}

/*inline
void sum_64(__local volatile  vT *s_sum,
            const int    local_id)
{
    //s_sum[local_id] += s_sum[local_id + 32];
    //s_sum[local_id] += s_sum[local_id + 16];
    //s_sum[local_id] += s_sum[local_id + 8];
    //s_sum[local_id] += s_sum[local_id + 4];
    //s_sum[local_id] += s_sum[local_id + 2];
    //s_sum[local_id] += s_sum[local_id + 1];
    vT sum = s_sum[local_id];
    if (local_id < 16) s_sum[local_id] = sum = sum +  s_sum[local_id + 16] + s_sum[local_id + 32] + s_sum[local_id + 48];
    if (local_id < 4)  s_sum[local_id] = sum = sum +  s_sum[local_id + 4] + s_sum[local_id + 8] + s_sum[local_id + 12];
    if (local_id < 1)  s_sum[local_id] = sum = sum +  s_sum[local_id + 1] + s_sum[local_id + 2] + s_sum[local_id + 3];
}*/

inline
void sum_256(__local volatile  vT *s_sum,
            const int    local_id)
{
    if (local_id < 128) s_sum[local_id] += s_sum[local_id + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 64) s_sum[local_id] += s_sum[local_id + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 32) s_sum[local_id] += s_sum[local_id + 32];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 16) s_sum[local_id] += s_sum[local_id + 16];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 8)
    {
        s_sum[local_id] += s_sum[local_id + 8];
        s_sum[local_id] += s_sum[local_id + 4];
        s_sum[local_id] += s_sum[local_id + 2];
        s_sum[local_id] += s_sum[local_id + 1];
    }
}

inline
void scan_32(__local volatile vT *s_scan,
                   const int      lane_id)
{
    int ai, bi;
    int baseai = 1 + 2 * lane_id;
    int basebi = baseai + 1;
    vT temp;

    if (lane_id < 16) { ai =  baseai - 1;  bi =  basebi - 1;   s_scan[bi] += s_scan[ai]; } //barrier(CLK_LOCAL_MEM_FENCE);
    if (lane_id < 8)  { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (lane_id < 4)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (lane_id < 2)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (lane_id == 0) { s_scan[31] += s_scan[15]; s_scan[31] = 0; temp = s_scan[15]; s_scan[15] = 0; s_scan[31] += temp; }
    if (lane_id < 2)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (lane_id < 4)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (lane_id < 8)  { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (lane_id < 16) { ai =  baseai - 1;  bi =  basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}  //barrier(CLK_LOCAL_MEM_FENCE);
}

/*inline
void scan_64(__local volatile vT *s_scan,
                   const int      lane_id)
{
    int ai, bi;
    int baseai = 1 + 2 * lane_id;
    int basebi = baseai + 1;
    vT temp;

    if (lane_id < 32) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    if (lane_id < 16) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (lane_id < 8)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (lane_id < 4)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (lane_id < 2)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (lane_id == 0) { s_scan[63] += s_scan[31]; s_scan[63] = 0; temp = s_scan[31]; s_scan[31] = 0; s_scan[63] += temp; }
    if (lane_id < 2)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (lane_id < 4)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (lane_id < 8)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (lane_id < 16) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (lane_id < 32) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}*/

inline
void atom_add_fp32(volatile __global float *val,
                   float delta)
{
    union { float f; unsigned int i; } old;
    union { float f; unsigned int i; } new;
    do
    {
        old.f = *val;
        new.f = old.f + delta;
    }
    while (atomic_cmpxchg((volatile __global unsigned int *)val, old.i, new.i) != old.i);
}

//inline
//void atom_add_fp64(volatile __global double *val,
  //                 double delta)
//{
    //union { double f; ulong i; } old;
    //union { double f; ulong i; } new;
    //do
    //{
      //  old.f = *val;
        //new.f = old.f + delta;
    //}
    //while (atom_cmpxchg((volatile __global ulong *)val, old.i, new.i) != old.i);
//}

inline
void candidate(__global const vT           *d_value_partition,
             __global const vT           *d_x,
             __global const iT           *d_column_index_partition,
             const iT            candidate_index,
             const vT            alpha,
             vT            *sum_t,
             vT            *sum_b)
{
    vT x = d_x[d_column_index_partition[candidate_index]];
    //return d_value_partition[candidate_index] * x * alpha;
    sum_t[0] = d_value_partition[candidate_index] * x;
    sum_b[0] = x * x;
}

inline
vT segmented_sum(vT             tmp_sum,
                 __local volatile vT   *s_sum,
                 const int      scansum_offset,
                 const int      lane_id)
{
    if (lane_id)
        s_sum[lane_id - 1] = tmp_sum;
    s_sum[lane_id] = lane_id == ANONYMOUSLIB_CSR5_OMEGA - 1 ? 0 : s_sum[lane_id];
    vT sum = tmp_sum = s_sum[lane_id];
    scan_32(s_sum, lane_id); // exclusive scan
    s_sum[lane_id] += tmp_sum; // inclusive scan = exclusive scan + original value
    tmp_sum = s_sum[lane_id + scansum_offset];
    tmp_sum = tmp_sum - s_sum[lane_id] + sum;

    return tmp_sum;
}

inline
void partition_fast_track(__global const vT           *d_value_partition,
                          __global const vT           *d_x,
                          __global const iT           *d_column_index_partition,
                          __global vT                 *d_calibrator_t,
                          __global vT                 *d_calibrator_b,
                          __local volatile vT        *s_sum_t,
                          __local volatile vT        *s_sum_b,
                          const int           lane_id,
                          const iT            par_id,
                          const vT            alpha)
{
    vT sum_t = 0;
    vT sum_b = 0;
    vT sum_t_ = 0;
    vT sum_b_ = 0;

    #pragma unroll
    for (int i = 0; i < ANONYMOUSLIB_CSR5_SIGMA; i++){
        candidate(d_value_partition, d_x, d_column_index_partition, i * ANONYMOUSLIB_CSR5_OMEGA + lane_id, alpha, &sum_t_, &sum_b_);
        sum_t += sum_t_, sum_b += sum_b_;}

    s_sum_t[lane_id] = sum_t;
    s_sum_b[lane_id] = sum_b;
    sum_32(s_sum_t, lane_id);
    sum_32(s_sum_b, lane_id);
    if (!lane_id)
        d_calibrator_t[par_id] = s_sum_t[0];
        d_calibrator_b[par_id] = s_sum_b[0];
}

inline
void partition_normal_track(__global const iT           *d_column_index_partition,
                            __global const vT           *d_value_partition,
                            __global const vT           *d_x,
                            __global const uiT          *d_partition_descriptor,
                            __global const iT           *d_partition_descriptor_offset_pointer,
                            __global const iT           *d_partition_descriptor_offset,
                            __global vT                 *d_calibrator_t,
                            __global vT                 *d_calibrator_b,
                            __global vT                 *d_y_t,
                            __global vT                 *d_y_b,
                            __local volatile vT         *s_sum_t,
                            __local volatile vT         *s_sum_b,
                            __local volatile int        *s_scan,
                            const iT            par_id,
                            const int           lane_id,
                            const int           bit_y_offset,
                            const int           bit_scansum_offset,
                            iT                  row_start,
                            const bool          empty_rows,
                            const vT            alpha)
{
    int start = 0;
    int stop = 0;

    bool local_bit;
    vT sum_t = 0;
    vT sum_b = 0;
    vT sum_t_ = 0;
    vT sum_b_ = 0;

    int offset_pointer = empty_rows ? d_partition_descriptor_offset_pointer[par_id] : 0;

    uiT descriptor = d_partition_descriptor[lane_id];

    int y_offset = descriptor >> (32 - bit_y_offset);
    const int scansum_offset = (descriptor << bit_y_offset) >> (32 - bit_scansum_offset);
    const int bit_bitflag = 32 - bit_y_offset - bit_scansum_offset;

    bool direct = false;

    vT first_sum_t, last_sum_t;
    vT first_sum_b, last_sum_b;

    // step 1. thread-level seg sum
#if ANONYMOUSLIB_CSR5_SIGMA > 16
    int ly = 0;
#endif

    // extract the first bit-flag packet
    descriptor = descriptor << (bit_y_offset + bit_scansum_offset);
    descriptor = lane_id ? descriptor : descriptor | 0x80000000;

    local_bit = (descriptor >> 31) & 0x1;
    start = !local_bit;
    direct = local_bit & (bool)lane_id;

    candidate(d_value_partition, d_x, d_column_index_partition, lane_id, alpha, &sum_t, &sum_b);

    #pragma unroll
    for (int i = 1; i < ANONYMOUSLIB_CSR5_SIGMA; i++)
    {
#if ANONYMOUSLIB_CSR5_SIGMA > 16
        int norm_i = i - bit_bitflag;

        if (!(ly || norm_i) || (ly && !(31 & norm_i)))
        {
            ly++;
            descriptor = d_partition_descriptor[ly * ANONYMOUSLIB_CSR5_OMEGA + lane_id];
        }
        norm_i = !ly ? 31 & i : 31 & norm_i;
        norm_i = 31 - norm_i;

        local_bit = (descriptor >> norm_i) & 0x1;
#else
        local_bit = (descriptor >> (31-i)) & 0x1;
#endif
        if (local_bit)
        {
            if (direct){
                d_y_t[empty_rows ? d_partition_descriptor_offset[offset_pointer + y_offset] : y_offset] = sum_t;
                d_y_b[empty_rows ? d_partition_descriptor_offset[offset_pointer + y_offset] : y_offset] = sum_b;}
            else{
                first_sum_t = sum_t;
                first_sum_b = sum_b;}
        }

        y_offset += local_bit & direct;

        direct |= local_bit;
        sum_t = local_bit ? 0 : sum_t;
        sum_b = local_bit ? 0 : sum_b;
        stop += local_bit;

        candidate(d_value_partition, d_x, d_column_index_partition, i * ANONYMOUSLIB_CSR5_OMEGA + lane_id, alpha, &sum_t_, &sum_b_);
        sum_t += sum_t_, sum_b += sum_b_;
    }

    first_sum_t = direct ? first_sum_t : sum_t;
    first_sum_b = direct ? first_sum_b : sum_b;
    last_sum_t = sum_t;
    last_sum_b = sum_b;

    // step 2. segmented sum
    sum_t = start ? first_sum_t : 0;
    sum_b = start ? first_sum_b : 0;

    sum_t = segmented_sum(sum_t, s_sum_t, scansum_offset, lane_id);
    sum_b = segmented_sum(sum_b, s_sum_b, scansum_offset, lane_id);

    // step 3-1. add s_sum to position stop
    last_sum_t += (start <= stop) ? sum_t : 0;
    last_sum_b += (start <= stop) ? sum_b : 0;

    // step 3-2. write sums to result array
    if (direct){
        d_y_t[empty_rows ? d_partition_descriptor_offset[offset_pointer + y_offset] : y_offset] = last_sum_t;
        d_y_b[empty_rows ? d_partition_descriptor_offset[offset_pointer + y_offset] : y_offset] = last_sum_b;}

    // the first/last value of the first thread goes to calibration
    if (!lane_id){
        d_calibrator_t[par_id] = direct ? first_sum_t : last_sum_t;
        d_calibrator_b[par_id] = direct ? first_sum_b : last_sum_b;}
}

inline
void spmv_partition(__global const iT           *d_column_index_partition,
                    __global const vT           *d_value_partition,
                    __global const iT           *d_row_pointer,
                    __global const vT           *d_x,
                    __global const uiT          *d_partition_pointer,
                    __global const uiT          *d_partition_descriptor,
                    __global const iT           *d_partition_descriptor_offset_pointer,
                    __global const iT           *d_partition_descriptor_offset,
                    __global vT                 *d_calibrator_t,
                    __global vT                 *d_calibrator_b,
                    __global vT                 *d_y_t,
                    __global vT                 *d_y_b,
                    const iT            par_id,
                    const int           lane_id,
                    const int           bunch_id,
                    const int           bit_y_offset,
                    const int           bit_scansum_offset,
                    const vT            alpha,
                    volatile __local vT  *s_sum_t,
                    volatile __local vT  *s_sum_b,
                    volatile __local int *s_scan,
                    volatile __local uiT *s_row_start_stop)
{
    //volatile __local vT s_y[ANONYMOUSLIB_THREAD_GROUP];

    //volatile __local vT  s_sum[ANONYMOUSLIB_THREAD_GROUP + ANONYMOUSLIB_CSR5_OMEGA / 2];
    //volatile __local int s_scan[(ANONYMOUSLIB_CSR5_OMEGA + 1) * (ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA)];

    uiT row_start, row_stop;

    //volatile __local uiT s_row_start_stop[ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1];
    if (get_local_id(0) < ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1)
        s_row_start_stop[get_local_id(0)] = d_partition_pointer[par_id + get_local_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    row_start = s_row_start_stop[bunch_id];
    row_stop  = s_row_start_stop[bunch_id + 1] & 0x7FFFFFFF;

    if (row_start == row_stop) // fast track through reduction
    {
        partition_fast_track
                (d_value_partition, d_x, d_column_index_partition,
                 d_calibrator_t,
                 d_calibrator_b,
                 &s_sum_t[bunch_id * ANONYMOUSLIB_CSR5_OMEGA],
                 &s_sum_b[bunch_id * ANONYMOUSLIB_CSR5_OMEGA],
                 lane_id, par_id, alpha);
    }
    else
    {
        const bool empty_rows = (row_start >> 31) & 0x1;
        row_start &= 0x7FFFFFFF;

        d_y_t = &d_y_t[row_start+1];
        d_y_b = &d_y_b[row_start+1];

        partition_normal_track
                (d_column_index_partition, d_value_partition, d_x,
                 d_partition_descriptor, d_partition_descriptor_offset_pointer, d_partition_descriptor_offset,
                 d_calibrator_t, d_calibrator_b, d_y_t, d_y_b,
                 &s_sum_t[bunch_id * ANONYMOUSLIB_CSR5_OMEGA],
                 &s_sum_b[bunch_id * ANONYMOUSLIB_CSR5_OMEGA],
                 &s_scan[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1)],
                 par_id, lane_id,
                 bit_y_offset, bit_scansum_offset, row_start, empty_rows, alpha);
    }
}

__kernel
void spmv_csr5_compute_kernel(__global const iT           *d_column_index,
                              __global const vT           *d_value,
                              __global const iT           *d_row_pointer,
                              __global const vT           *d_x,
                              __global const uiT          *d_partition_pointer,
                              __global const uiT          *d_partition_descriptor,
                              __global const iT           *d_partition_descriptor_offset_pointer,
                              __global const iT           *d_partition_descriptor_offset,
                              __global vT                 *d_calibrator_t,
                              __global vT                 *d_calibrator_b,
                              __global vT                 *d_y_t,
                              __global vT                 *d_y_b,
                              const iT            p,
                              const int           num_packet,
                              const int           bit_y_offset,
                              const int           bit_scansum_offset,
                              const vT            alpha)
{
    // warp lane id
    const int lane_id = get_local_id(0) % ANONYMOUSLIB_CSR5_OMEGA;
    // warp global id == par_id
    const iT  par_id = get_global_id(0) / ANONYMOUSLIB_CSR5_OMEGA;
    const int bunch_id = get_local_id(0) / ANONYMOUSLIB_CSR5_OMEGA;

    if (par_id >= p - 1)
        return;
    volatile __local vT  s_sum_t[ANONYMOUSLIB_THREAD_GROUP + ANONYMOUSLIB_CSR5_OMEGA / 2];
    volatile __local vT  s_sum_b[ANONYMOUSLIB_THREAD_GROUP + ANONYMOUSLIB_CSR5_OMEGA / 2];
    volatile __local int s_scan[(ANONYMOUSLIB_CSR5_OMEGA + 1) * (ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA)];
    volatile __local uiT s_row_start_stop[ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1];

    spmv_partition(&d_column_index[par_id * ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA],
                 &d_value[par_id * ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA],
                 d_row_pointer, d_x, d_partition_pointer,
                 &d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet],
                 d_partition_descriptor_offset_pointer, d_partition_descriptor_offset,
                 d_calibrator_t, d_calibrator_b, d_y_t, d_y_b,
                 par_id, lane_id, bunch_id, bit_y_offset, bit_scansum_offset, alpha, s_sum_t, s_sum_b, s_scan, s_row_start_stop);
}

__kernel
void spmv_csr5_calibrate_kernel(__global const uiT *d_partition_pointer,
                                __global const vT  *d_calibrator_t,
                                __global const vT  *d_calibrator_b,
                                __global vT        *d_y_t,
                                __global vT        *d_y_b,
                                const iT   p)
{
    const int lane_id  = get_local_id(0) % ANONYMOUSLIB_THREAD_BUNCH;
    const int bunch_id = get_local_id(0) / ANONYMOUSLIB_THREAD_BUNCH;
    const int local_id = get_local_id(0);
    const iT global_id = get_global_id(0);

    vT sum_t;
    vT sum_b;

    volatile __local iT s_partition_pointer[ANONYMOUSLIB_THREAD_GROUP+1];
    volatile __local vT  s_calibrator_t[ANONYMOUSLIB_THREAD_GROUP];
    volatile __local vT  s_calibrator_b[ANONYMOUSLIB_THREAD_GROUP];
    //volatile __local vT  s_sum[ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_THREAD_BUNCH];

    s_partition_pointer[local_id] = global_id < p-1 ? d_partition_pointer[global_id] & 0x7FFFFFFF : -1;
    s_calibrator_t[local_id] = sum_t = global_id < p-1 ? d_calibrator_t[global_id] : 0;
    s_calibrator_b[local_id] = sum_b = global_id < p-1 ? d_calibrator_b[global_id] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // do a fast track if all s_partition_pointer are the same
    if (s_partition_pointer[0] == s_partition_pointer[ANONYMOUSLIB_THREAD_GROUP-1])
    {
        // sum all calibrators
        sum_256(s_calibrator_t, local_id);
        sum_256(s_calibrator_b, local_id);
        //d_y[s_partition_pointer[0]] += sum;
        if (!local_id)
        {
            if (sizeof(vT) == 8){
              //atom_add_fp64(&d_y_t[s_partition_pointer[0]], s_calibrator_t[0]);
              //atom_add_fp64(&d_y_b[s_partition_pointer[0]], s_calibrator_b[0]);
              }
            else{
              atom_add_fp32(&d_y_t[s_partition_pointer[0]], s_calibrator_t[0]);
              atom_add_fp32(&d_y_b[s_partition_pointer[0]], s_calibrator_b[0]);}
        }
        return;
    }

    int local_par_id = local_id;
    iT row_start_current, row_start_target, row_start_previous;
    sum_t = 0;
    sum_b = 0;

    // use (p - 1), due to the tail partition is dealt with CSR-vector method
    if (global_id < p - 1)
    {
        row_start_previous = local_id ? s_partition_pointer[local_id-1] : -1;
        row_start_current = s_partition_pointer[local_id];

        if (row_start_previous != row_start_current)
        {
            row_start_target = row_start_current;

            while (row_start_target == row_start_current && local_par_id < get_local_size(0))
            {
                sum_t +=  s_calibrator_t[local_par_id];
                sum_b +=  s_calibrator_b[local_par_id];
                local_par_id++;
                row_start_current = s_partition_pointer[local_par_id];
            }

            if (row_start_target == s_partition_pointer[0] || row_start_target == s_partition_pointer[ANONYMOUSLIB_THREAD_GROUP-1])
            {
                if (sizeof(vT) == 8){
                    //atom_add_fp64(&d_y_t[row_start_target], sum_t);
                    //atom_add_fp64(&d_y_b[row_start_target], sum_b);}
                    }
                else{
                    atom_add_fp32(&d_y_t[row_start_target], sum_t);
                    atom_add_fp32(&d_y_b[row_start_target], sum_b);}
            }
            else{
                d_y_t[row_start_target] += sum_t;
                d_y_b[row_start_target] += sum_b;}
        }
    }
}

__kernel
void spmv_csr5_tail_partition_kernel(__global const iT           *d_row_pointer,
                                     __global const iT           *d_column_index,
                                     __global const vT           *d_value,
                                     __global const vT           *d_x,
                                     __global vT                 *d_y_t,
                                     __global vT                 *d_y_b,
                                     const iT            tail_partition_start,
                                     const iT            p,
                                     const int           sigma,
                                     const vT            alpha)
{
    const int local_id = get_local_id(0);

    const iT row_id    = tail_partition_start + get_group_id(0);
    const iT row_start = !get_group_id(0) ? (p - 1) * ANONYMOUSLIB_CSR5_OMEGA * sigma : d_row_pointer[row_id];
    const iT row_stop  = d_row_pointer[row_id + 1];

    vT sum_t = 0;
    vT sum_b = 0;
    vT sum_t_ = 0;
    vT sum_b_ = 0;

    for (iT idx = local_id + row_start; idx < row_stop; idx += ANONYMOUSLIB_CSR5_OMEGA){
        candidate(d_value, d_x, d_column_index, idx, alpha, &sum_t_, &sum_b_);
        sum_t += sum_t_, sum_b += sum_b_;}

    volatile __local vT s_sum_t[ANONYMOUSLIB_CSR5_OMEGA + ANONYMOUSLIB_CSR5_OMEGA / 2];
    volatile __local vT s_sum_b[ANONYMOUSLIB_CSR5_OMEGA + ANONYMOUSLIB_CSR5_OMEGA / 2];
    s_sum_t[local_id] = sum_t;
    s_sum_b[local_id] = sum_b;
    sum_32(s_sum_t, local_id);
    sum_32(s_sum_b, local_id);
    sum_t = s_sum_t[local_id];
    sum_b = s_sum_b[local_id];

    if (!local_id){
        d_y_t[row_id] = !get_group_id(0) ? d_y_t[row_id] + sum_t : sum_t;
        d_y_b[row_id] = !get_group_id(0) ? d_y_b[row_id] + sum_b : sum_b;}
}
