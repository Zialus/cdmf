#ifndef CDMF_H
#define CDMF_H

void cdmf_ref(SparseMatrix& R, MatData& W, MatData& H, TestData& T, parameter& param);

void cdmf_ocl_02(SparseMatrix& R, MatData& W_c, MatData& H_c, TestData& T, parameter& param, char filename[]);

void cdmf_ocl_01(SparseMatrix& R, MatData& W_c, MatData& H_c, TestData& T, parameter& param, char filename[]);

void cdmf_csr5(SparseMatrix& R, MatData& W, MatData& H, parameter& param, char filename[]);

#endif // CDMF_H
