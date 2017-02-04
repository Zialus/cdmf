#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "anonymouslib_phi.h"
#include "mmio.h"
using namespace std;

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#ifndef NUM_RUN
#define NUM_RUN 1000
#endif


	template <typename T>
inline std::string to_string(T value)
{
	std::ostringstream os ;
	os << value ;
	return os.str() ;
}

int call_anonymouslib(int m, int n, int nnzA,
                  int *csrRowPtrA, int *csrColIdxA, VALUE_TYPE *csrValA,
                  VALUE_TYPE *x, VALUE_TYPE *y, VALUE_TYPE alpha)
{
    int err = 0;

    memset(y, 0, sizeof(VALUE_TYPE) * m);

    double gb = getB<int, VALUE_TYPE>(m, nnzA);
    double gflop = getFLOP<int>(nnzA);

    anonymouslib_timer asCSR5_timer;
    anonymouslib_timer CSR5Spmv_timer;
    anonymouslib_timer ref_timer;

#pragma offload mandatory target(mic:0) in(m, n, nnzA, gb, gflop) \
    in(csrRowPtrA:length(m+1)) \
    in(csrColIdxA:length(nnzA)) \
    in(csrValA:length(nnzA)) \
    in(x:length(n)) \
    inout(y:length(m)) \
    in(asCSR5_timer) \
    in(CSR5Spmv_timer) \
    in(ref_timer)
{
    printf("omp_get_max_threads = %i\n", omp_get_max_threads());

    int        *d_csrRowPtrA = (int *)_mm_malloc((m+1) * sizeof(int), ANONYMOUSLIB_X86_CACHELINE);
    int        *d_csrColIdxA = (int *)_mm_malloc(nnzA * sizeof(int), ANONYMOUSLIB_X86_CACHELINE);
    VALUE_TYPE *d_csrValA = (VALUE_TYPE *)_mm_malloc(nnzA * sizeof(VALUE_TYPE), ANONYMOUSLIB_X86_CACHELINE);
    VALUE_TYPE *d_x = (VALUE_TYPE *)_mm_malloc(n * sizeof(VALUE_TYPE), ANONYMOUSLIB_X86_CACHELINE);
    VALUE_TYPE *d_y = (VALUE_TYPE *)_mm_malloc(m * sizeof(VALUE_TYPE), ANONYMOUSLIB_X86_CACHELINE);
    VALUE_TYPE *d_y_bench = (VALUE_TYPE *)_mm_malloc(m * sizeof(VALUE_TYPE), ANONYMOUSLIB_X86_CACHELINE);

    memcpy(d_csrRowPtrA, csrRowPtrA, (m+1) * sizeof(int));
    memcpy(d_csrColIdxA, csrColIdxA, nnzA * sizeof(int));
    memcpy(d_csrValA, csrValA, nnzA * sizeof(VALUE_TYPE));
    memcpy(d_x, x, n * sizeof(VALUE_TYPE));
    memset(d_y, 0, m * sizeof(VALUE_TYPE));
    memset(d_y_bench, 0, m * sizeof(VALUE_TYPE));

    /*
    ref_timer.start();

    int ref_iter = 1000;
    #pragma omp parallel for
    for (int iter = 0; iter < ref_iter; iter++)
    {
        for (int i = 0; i < m; i++)
        {
            VALUE_TYPE sum = 0;
            for (int j = d_csrRowPtrA[i]; j < d_csrRowPtrA[i+1]; j++)
                sum += d_x[d_csrColIdxA[j]] * d_csrValA[j];
            d_y_bench[i] = sum;
        }
    }

    double ref_time = ref_timer.stop() / (double)ref_iter;

    printf("CSR-based SpMV MIC-OMP time = %f ms. Bandwidth = %f GB/s. GFlops = %f GFlops.\n\n",
               ref_time, gb/(1.0e+6 * ref_time), gflop/(1.0e+6 * ref_time));
    */ 
    anonymouslibHandle<int, unsigned int, VALUE_TYPE> A(m, n);
    err = A.inputCSR(nnzA, d_csrRowPtrA, d_csrColIdxA, d_csrValA);
    //cout << "inputCSR err = " << err << endl;

    err = A.setX(d_x); // you only need to do it once!
    //cout << "setX err = " << err << endl;

    int sigma = ANONYMOUSLIB_CSR5_SIGMA; //nnzA/(8*ANONYMOUSLIB_CSR5_OMEGA);
    A.setSigma(sigma);

    A.asCSR5();
    A.asCSR();

    // record a correct CSR->CSR5 time without PCIe overhead
    asCSR5_timer.start();
    err = A.asCSR5();
    printf("CSR->CSR5 time = %f ms.\n", asCSR5_timer.stop());
    //cout << "asCSR5 err = " << err << endl;
   
    // check correctness by running 1 time
    err = A.spmv(alpha, d_y);
    memcpy(y, d_y, m * sizeof(VALUE_TYPE));
    //cout << "spmv err = " << err << endl;

    // warm up by running 50 times
    if (NUM_RUN)
    {
        for (int i = 0; i < 50; i++)
            err = A.spmv(alpha, d_y_bench);

        CSR5Spmv_timer.start();

        // time spmv by running NUM_RUN times
        for (int i = 0; i < NUM_RUN; i++)
            err = A.spmv(alpha, d_y_bench);

        double CSR5Spmv_time = CSR5Spmv_timer.stop() / (double)NUM_RUN;

        printf("CSR5-based SpMV MIC time = %f ms. Bandwidth = %f GB/s. GFlops = %f GFlops.\n",
               CSR5Spmv_time, gb/(1.0e+6 * CSR5Spmv_time), gflop/(1.0e+6 * CSR5Spmv_time));
    }

    _mm_free(d_csrRowPtrA);
    _mm_free(d_csrColIdxA);
    _mm_free(d_csrValA);
    _mm_free(d_x);
    _mm_free(d_y);  
    _mm_free(d_y_bench);
}

    return err;
}


int main(int argc, char** argv){
	char input_file_name[1024];
	char filename[1024] = {"./kcode/ccd01.cl"};
	parameter param = parse_command_line(argc, argv, input_file_name, NULL, filename);
	// reading rating matrix
	smat_t R;	// val: csc, val_t: csr
	load(input_file_name, R, false, false);
	unsigned int m = R.rows;
	unsigned int n = R.cols;
	unsigned int *ptr = R.row_ptr;
	unsigned int *idx = R.col_idx;
	unsigned int nnz = R.nnz;
	VALUE_TYPE * value = R.val_t;
	cout << " ( " << m << ", " << n << " ) nnz = " << nnz << endl;

	// native spmv
	VALUE_TYPE *x = (VALUE_TYPE *)malloc(n * sizeof(VALUE_TYPE));
	for(unsigned int i = 0; i < n; i++)
		x[i] = (VALUE_TYPE)i * 0.1;
	VALUE_TYPE *y = (VALUE_TYPE *)malloc(m * sizeof(VALUE_TYPE));
	VALUE_TYPE *y_ref = (VALUE_TYPE *)malloc(m * sizeof(VALUE_TYPE));

	// for debugging
	srand(time(NULL));
	for(unsigned int i=0; i<nnz; i++)
	{
		value[i] = rand() % 10;
	}

	// compute reference results on a cpu core
	VALUE_TYPE alpha = 1.0;
	double gb = getB<int, VALUE_TYPE>(m, nnz);
	double gflop = getFLOP<int>(nnz);
	double t = gettime();
	int ref_iter = 1;
	for (int iter = 0; iter < ref_iter; iter++){
		for (int i = 0; i < m; i++){
			VALUE_TYPE sum = 0;
			for (int j = ptr[i]; j < ptr[i+1]; j++)
				sum += x[idx[j]] * value[j] * alpha;
			y_ref[i] = sum;
		}
	}

	double ref_time = (gettime() - t) / (double)ref_iter;
	cout << "cpu sequential time = " << ref_time
		<< " ms. Bandwidth = " << gb/(1.0e+6 * ref_time)
		<< " GB/s. GFlops = " << gflop/(1.0e+6 * ref_time)  << " GFlops." << endl << endl;
	// compute spmv on the ocl device
	call_anonymouslib(m, n, nnz, (int *)ptr, (int *)idx, value, x, y, alpha);

	// compare reference and anonymouslib results
	int error_count = 0;
	for (int i = 0; i < m; i++)
		if (abs(y_ref[i] - y[i]) > 0.01 * abs(y_ref[i]))
		{
			error_count++;
			cout << y_ref[i] << "," << y[i] << "\t";
		}
	if (error_count == 0)
		cout << "Check... PASS!" << endl;
	else
		cout << "Check... NO PASS! #Error = " << error_count << " out of " << m << " entries." << endl;
	cout << "------------------------------------------------------" << endl;

	if(x!=NULL) free(x);
	if(y!=NULL) free(y);
	if(y_ref!=NULL) free(y_ref);
	return 0;
}


