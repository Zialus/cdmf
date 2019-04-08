#ifndef PMF_UTIL_H
#define PMF_UTIL_H

#include "util.h"
#include <memory>

using VecData = std::vector<VALUE_TYPE>;
using MatData = std::vector<VecData>;
using VecInt = std::vector<int>;
using MatInt = std::vector<VecInt>;

// Sparse matrix format CSC & CSR
class SparseMatrix {
public:
    long rows, cols, nnz, max_row_nnz, max_col_nnz;

    void read_binary_file(long rows_, long cols_, long nnz_,
//                                    std::string fname_data, std::string fname_row, std::string fname_col,
                          const std::string& fname_csr_row_ptr, const std::string& fname_csr_col_indx,
                          const std::string& fname_csr_val,
                          const std::string& fname_csc_col_ptr, const std::string& fname_csc_row_indx,
                          const std::string& fname_csc_val) {
        this->rows = rows_;
        this->cols = cols_;
        this->nnz = nnz_;

        /// read csr
        this->read_compressed(fname_csr_row_ptr, fname_csr_col_indx, fname_csr_val,
                              this->csr_row_ptr_, this->csr_col_indx_, this->csr_val_, rows + 1,
                              this->max_row_nnz);

        /// read csc
        this->read_compressed(fname_csc_col_ptr, fname_csc_row_indx, fname_csc_val,
                              this->csc_col_ptr_, this->csc_row_indx_, this->csc_val_, cols + 1,
                              this->max_col_nnz);
    }

    SparseMatrix get_shallow_transpose() {
        SparseMatrix shallow_transpose;
        shallow_transpose.cols = rows;
        shallow_transpose.rows = cols;
        shallow_transpose.nnz = nnz;
        shallow_transpose.csc_val_ = csr_val_;
        shallow_transpose.csr_val_ = csc_val_;
        shallow_transpose.csc_col_ptr_ = csr_row_ptr_;
        shallow_transpose.csr_row_ptr_ = csc_col_ptr_;
        shallow_transpose.csr_col_indx_ = csc_row_indx_;
        shallow_transpose.csc_row_indx_ = csr_col_indx_;
        shallow_transpose.max_col_nnz = max_row_nnz;
        shallow_transpose.max_row_nnz = max_col_nnz;

        return shallow_transpose;
    }

    unsigned* get_csc_col_ptr() const {
        return csc_col_ptr_.get();
    }

    unsigned* get_csc_row_indx() const {
        return csc_row_indx_.get();
    }

    VALUE_TYPE* get_csc_val() const {
        return csc_val_.get();
    }

    unsigned* get_csr_col_indx() const {
        return csr_col_indx_.get();
    }

    unsigned* get_csr_row_ptr() const {
        return csr_row_ptr_.get();
    }

    VALUE_TYPE* get_csr_val() const {
        return csr_val_.get();
    }

private:
    void read_compressed(const std::string& fname_cs_ptr, const std::string& fname_cs_indx, const std::string& fname_cs_val,
                         std::shared_ptr<unsigned>& cs_ptr, std::shared_ptr<unsigned>& cs_indx, std::shared_ptr<VALUE_TYPE>& cs_val,
                         long num_elems_in_cs_ptr, long& max_nnz_in_one_dim) {

        cs_ptr = std::shared_ptr<unsigned>(new unsigned[num_elems_in_cs_ptr], std::default_delete<unsigned[]>());
        cs_indx = std::shared_ptr<unsigned>(new unsigned[this->nnz], std::default_delete<unsigned[]>());
        cs_val = std::shared_ptr<VALUE_TYPE>(new VALUE_TYPE[this->nnz], std::default_delete<VALUE_TYPE[]>());

        std::ifstream f_indx(fname_cs_indx, std::ios::binary);
        std::ifstream f_val(fname_cs_val, std::ios::binary);

        for (long i = 0; i < this->nnz; i++) {
            f_indx.read((char*) &cs_indx.get()[i], sizeof(unsigned));
            f_val.read((char*) &cs_val.get()[i], sizeof(float));
        }

        std::ifstream f_ptr(fname_cs_ptr, std::ios::binary);
        max_nnz_in_one_dim = std::numeric_limits<long>::min();

        int cur = 0;
        for (long i = 0; i < num_elems_in_cs_ptr; i++) {
            int prev = cur;
            f_ptr.read((char*) &cur, sizeof(int));
            cs_ptr.get()[i] = cur;

            if (i > 0) { max_nnz_in_one_dim = std::max<long>(max_nnz_in_one_dim, cur - prev); }
        }
    }

    std::shared_ptr<unsigned> csc_col_ptr_, csr_row_ptr_, col_nnz_, row_nnz_;
    std::shared_ptr<VALUE_TYPE> csr_val_, csc_val_;
    std::shared_ptr<unsigned> csc_row_indx_, csr_col_indx_;
};

// Test set in COO format
class TestData {
public:
    long rows, cols, nnz;

    void read(long rows_, long cols_, long nnz_, const std::string& filename) {
        this->rows = rows_;
        this->cols = cols_;
        this->nnz = nnz_;

        test_row = std::unique_ptr<unsigned[]>(new unsigned[nnz_]);
        test_col = std::unique_ptr<unsigned[]>(new unsigned[nnz_]);
        test_val = std::unique_ptr<VALUE_TYPE[]>(new VALUE_TYPE[nnz_]);

        std::ifstream fp(filename);
        for (long idx = 0; idx < nnz_; ++idx) {
            fp >> test_row[idx] >> test_col[idx] >> test_val[idx];
        }
    }

    unsigned* getTestCol() const {
        return test_col.get();
    }

    unsigned* getTestRow() const {
        return test_row.get();
    }

    VALUE_TYPE* getTestVal() const {
        return test_val.get();
    }

private:
    std::unique_ptr<unsigned[]> test_row, test_col;
    std::unique_ptr<VALUE_TYPE[]> test_val;
};

#endif //PMF_UTIL_H
