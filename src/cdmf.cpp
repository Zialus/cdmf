#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <string>
#include <fstream>

#include "util.h"

using namespace std;

void cdmf_ref(smat_t &R, mat_t &W, mat_t &H, parameter &param);
void cdmf_ocl(smat_t &R, mat_t &W, mat_t &H, parameter &param, const char* srcdir);
void cdmf_csr5(smat_t &R, mat_t &W, mat_t &H, parameter &param);
void cdmf_native(smat_t &R, mat_t &W, mat_t &H, parameter &param, const char* srcdir);

int main(int argc, char** argv){
	char input_file_name[1024];
	char filename[1024] = {"../kcode/ccd01.cl"};
	parameter param = parse_command_line(argc, argv, input_file_name, nullptr, filename);

	// reading rating matrix
	smat_t R;	// val: csc, val_t: csr
	mat_t W;
	mat_t W_ref;
	mat_t H;
	mat_t H_ref;

	cout << "[info] load rating data." << endl;
    double t1 = gettime();
    load(input_file_name, R, false, false);
    double t2 = gettime();
    double deltaT = t2 - t1;
    printf("[info] - loading time: %lf s\n",  deltaT);

	// W, H  here are k*m, k*n
	cout << "[info] initializ W and H matrix." << endl;
	initial_col(W, param.k, R.rows);
	initial_col(W_ref, param.k, R.rows);
	initial_col(H, param.k, R.cols);
	initial_col(H_ref, param.k, R.cols);

	// compute cdmf on the ocl device
	cout << "------------------------------------------------------" << endl;
	cout << "[info] compute cdmf on the selected ocl device." << endl;

#ifdef V1
	cdmf_native(R, W, H, param, input_file_name);
#endif
#ifdef V2
	cdmf_ocl(R, W, H, param, input_file_name);
#endif
#ifdef V3
	cdmf_csr5(R, W, H, param);
#endif

	cout << "------------------------------------------------------" << endl;
	cout << "[info] now computing cdmf reference results on a cpu core." << endl;
	cdmf_ref(R, W_ref, H_ref, param);

	// compare reference and anonymouslib results
	cout << "[info] validate the results." << endl;
	golden_compare(W, W_ref, param.k, R.rows);
	golden_compare(H, H_ref, param.k, R.cols);

	cout << "------------------------------------------------------" << endl;

	return 0;
}
