#ifndef CDMF_H
#define CDMF_H

void cdmf_ref(smat_t& R, mat_t& W, mat_t& H, parameter& param);

void cdmf_ocl(smat_t& R, mat_t& W, mat_t& H, parameter& param, char filename[]);

void cdmf_csr5(smat_t& R, mat_t& W, mat_t& H, parameter& param, char filename[]);

#endif // CDMF_H
