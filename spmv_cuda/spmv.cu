#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "anonymouslib_cuda.h"
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
    cudaError_t err_cuda = cudaSuccess;

    // set device
    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    cout << "Device [" <<  device_id << "] " << deviceProp.name << ", " << " @ " << deviceProp.clockRate * 1e-3f << "MHz. " << endl;

    double gb = getB<int, VALUE_TYPE>(m, nnzA);
    double gflop = getFLOP<int>(nnzA);

    // Define pointers of matrix A, vector x and y
    int *d_csrRowPtrA;
    int *d_csrColIdxA;
    VALUE_TYPE *d_csrValA;
    VALUE_TYPE *d_x;
    VALUE_TYPE *d_y;

    // Matrix A
    checkCudaErrors(cudaMalloc((void **)&d_csrRowPtrA, (m+1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_csrColIdxA, nnzA  * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_csrValA,    nnzA  * sizeof(VALUE_TYPE)));

    checkCudaErrors(cudaMemcpy(d_csrRowPtrA, csrRowPtrA, (m+1) * sizeof(int),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrColIdxA, csrColIdxA, nnzA  * sizeof(int),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrValA,    csrValA,    nnzA  * sizeof(VALUE_TYPE),   cudaMemcpyHostToDevice));

    // Vector x
    checkCudaErrors(cudaMalloc((void **)&d_x, n * sizeof(VALUE_TYPE)));
    checkCudaErrors(cudaMemcpy(d_x, x, n * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));

    // Vector y
    checkCudaErrors(cudaMalloc((void **)&d_y, m  * sizeof(VALUE_TYPE)));
    checkCudaErrors(cudaMemset(d_y, 0, m * sizeof(VALUE_TYPE)));

    anonymouslibHandle<int, unsigned int, VALUE_TYPE> A(m, n);
    err = A.inputCSR(nnzA, d_csrRowPtrA, d_csrColIdxA, d_csrValA);
    //cout << "inputCSR err = " << err << endl;

    err = A.setX(d_x); // you only need to do it once!
    //cout << "setX err = " << err << endl;

    A.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA);

    // warmup device
    A.warmup();

    anonymouslib_timer asCSR5_timer;
    asCSR5_timer.start();

    err = A.asCSR5();

    cout << "CSR->CSR5 time = " << asCSR5_timer.stop() << " ms." << endl;
    //cout << "asCSR5 err = " << err << endl;

    // check correctness by running 1 time
    err = A.spmv(alpha, d_y);
    //cout << "spmv err = " << err << endl;
    checkCudaErrors(cudaMemcpy(y, d_y, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));

    // warm up by running 50 times
    if (NUM_RUN)
    {
        for (int i = 0; i < 50; i++)
            err = A.spmv(alpha, d_y);
    }

    err_cuda = cudaDeviceSynchronize();

    anonymouslib_timer CSR5Spmv_timer;
    CSR5Spmv_timer.start();

    // time spmv by running NUM_RUN times
    for (int i = 0; i < NUM_RUN; i++)
        err = A.spmv(alpha, d_y);
    err_cuda = cudaDeviceSynchronize();

    double CSR5Spmv_time = CSR5Spmv_timer.stop() / (double)NUM_RUN;

    if (NUM_RUN)
        cout << "CSR5-based SpMV time = " << CSR5Spmv_time
             << " ms. Bandwidth = " << gb/(1.0e+6 * CSR5Spmv_time)
             << " GB/s. GFlops = " << gflop/(1.0e+6 * CSR5Spmv_time)  << " GFlops." << endl;

    A.destroy();

    checkCudaErrors(cudaFree(d_csrRowPtrA));
    checkCudaErrors(cudaFree(d_csrColIdxA));
    checkCudaErrors(cudaFree(d_csrValA));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));

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


