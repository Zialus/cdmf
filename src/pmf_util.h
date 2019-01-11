#ifndef PMF_UTIL_H
#define PMF_UTIL_H

#include "util.h"

typedef std::vector<VALUE_TYPE> vec_t;
typedef std::vector<vec_t> mat_t;

class rate_t {
public:
    unsigned i;
    unsigned j;
    VALUE_TYPE v;
};

// Comparator for sorting rates into row/column compression storage
class SparseComp {
public:
    const unsigned* row_idx;
    const unsigned* col_idx;

    SparseComp(const unsigned* row_idx_, const unsigned* col_idx_, bool isRCS_ = true) {
        row_idx = (isRCS_) ? row_idx_ : col_idx_;
        col_idx = (isRCS_) ? col_idx_ : row_idx_;
    }

    bool operator()(size_t x, size_t y) const {
        return (row_idx[x] < row_idx[y]) || ((row_idx[x] == row_idx[y]) && (col_idx[x] <= col_idx[y]));
    }
};

// Sparse matrix format CCS & RCS
// Access column format only when you use it..
class smat_t {
public:
    unsigned rows;
    unsigned cols;
    unsigned nnz;
    unsigned max_row_nnz;
    unsigned max_col_nnz;
    VALUE_TYPE* val;
    VALUE_TYPE* val_t;
    size_t nbits_val;
    size_t nbits_val_t;
    unsigned* col_ptr;
    unsigned* row_ptr;
    size_t nbits_col_ptr;
    size_t nbits_row_ptr;
    unsigned* col_nnz;
    unsigned* row_nnz;
    size_t nbits_col_nnz;
    size_t nbits_row_nnz;
    unsigned* row_idx;
    unsigned* col_idx;
    size_t nbits_row_idx;
    size_t nbits_col_idx;
    unsigned* colMajored_sparse_idx;
    size_t nbits_colMajored_sparse_idx;
    bool mem_alloc_by_me;

    smat_t() : mem_alloc_by_me(false) {}

    smat_t(const smat_t& m) {
        *this = m;
        mem_alloc_by_me = false;
    }

    rate_t read_next_line(FILE* fp) {
        unsigned i;
        unsigned j;
        VALUE_TYPE v;
        if (sizeof(VALUE_TYPE) == 8) {
            CHECK_FSCAN(fscanf(fp, "%u %u %lf", &i, &j, &v), 3);
        } else {
            CHECK_FSCAN(fscanf(fp, "%u %u %f", &i, &j, &v), 3);
        };
        return {i - 1, j - 1, v};
    }

    void load(unsigned _rows, unsigned _cols, unsigned _nnz, const char* filename, bool ifALS) {
        FILE* fp = fopen(filename, "r");
        load_from_file(_rows, _cols, _nnz, ifALS, fp);
        fclose(fp);
    }

    void load_from_file(unsigned _rows, unsigned _cols, unsigned _nnz, bool ifALS, FILE* fp) {
        rows = _rows;
        cols = _cols;
        nnz = _nnz;
        mem_alloc_by_me = true;
        val = MALLOC(VALUE_TYPE, nnz);
        val_t = MALLOC(VALUE_TYPE, nnz);
        nbits_val = SIZEBITS(VALUE_TYPE, nnz);
        nbits_val_t = SIZEBITS(VALUE_TYPE, nnz);
        row_idx = MALLOC(unsigned, nnz);
        col_idx = MALLOC(unsigned, nnz);
        nbits_row_idx = SIZEBITS(unsigned, nnz);
        nbits_col_idx = SIZEBITS(unsigned, nnz);
        row_ptr = MALLOC(unsigned, rows + 1);
        col_ptr = MALLOC(unsigned, cols + 1);
        nbits_row_ptr = SIZEBITS(unsigned, rows + 1);
        nbits_col_ptr = SIZEBITS(unsigned, cols + 1);
        memset(row_ptr, 0, sizeof(unsigned) * (rows + 1));
        memset(col_ptr, 0, sizeof(unsigned) * (cols + 1));
        if (ifALS) {
            colMajored_sparse_idx = MALLOC(unsigned, nnz);
            nbits_colMajored_sparse_idx = SIZEBITS(unsigned, nnz);
        }

        // a trick here to utilize the space the have been allocated
        std::vector<size_t> perm(_nnz);
        unsigned* tmp_row_idx = col_idx;
        unsigned* tmp_col_idx = row_idx;
        VALUE_TYPE* tmp_val = val;
        for (unsigned idx = 0; idx < _nnz; idx++) {
            rate_t rate = read_next_line(fp);
            row_ptr[rate.i + 1]++;
            col_ptr[rate.j + 1]++;
            tmp_row_idx[idx] = rate.i;
            tmp_col_idx[idx] = rate.j;
            tmp_val[idx] = rate.v;
            perm[idx] = idx;
        }
        // sort entries into row-majored ordering
        sort(perm.begin(), perm.end(), SparseComp(tmp_row_idx, tmp_col_idx, true));

        // Generate CRS format
        for (unsigned idx = 0; idx < _nnz; idx++) {
            val_t[idx] = tmp_val[perm[idx]];
            col_idx[idx] = tmp_col_idx[perm[idx]];
        }

        // Calculate nnz for each row and col
        max_row_nnz = max_col_nnz = 0;
        for (unsigned r = 1; r <= rows; ++r) {
            max_row_nnz = std::max(max_row_nnz, row_ptr[r]);
            row_ptr[r] += row_ptr[r - 1];
        }
        for (unsigned c = 1; c <= cols; ++c) {
            max_col_nnz = std::max(max_col_nnz, col_ptr[c]);
            col_ptr[c] += col_ptr[c - 1];
        }

        // Transpose CRS into CCS matrix
        for (unsigned r = 0; r < rows; ++r) {
            for (unsigned i = row_ptr[r]; i < row_ptr[r + 1]; ++i) {
                unsigned c = col_idx[i];
                row_idx[col_ptr[c]] = r;
                val[col_ptr[c]] = val_t[i];
                col_ptr[c]++;
            }
        }
        for (unsigned c = cols; c > 0; --c) { col_ptr[c] = col_ptr[c - 1]; }
        col_ptr[0] = 0;

        if (ifALS) {
            unsigned* mapIDX;
            mapIDX = MALLOC(unsigned, rows);
            for (unsigned r = 0; r < rows; ++r) {
                mapIDX[r] = row_ptr[r];
            }

            for (unsigned r = 0; r < nnz; ++r) {
                colMajored_sparse_idx[mapIDX[row_idx[r]]] = r;
                ++mapIDX[row_idx[r]];
            }
            free(mapIDX);
        }
    }

    unsigned nnz_of_row(int i) const { return (row_ptr[i + 1] - row_ptr[i]); }

    unsigned nnz_of_col(int i) const { return (col_ptr[i + 1] - col_ptr[i]); }

    VALUE_TYPE get_global_mean() {
        VALUE_TYPE sum = 0;
        for (unsigned i = 0; i < nnz; ++i) { sum += val[i]; }
        return sum / nnz;
    }

    void remove_bias(VALUE_TYPE bias) {
        for (unsigned i = 0; i < nnz; ++i) { val[i] -= bias; }
        for (unsigned i = 0; i < nnz; ++i) { val_t[i] -= bias; }
    }

    ~smat_t() {
        if (mem_alloc_by_me) {
            //puts("Warning: Somebody just freed me.");
            free(val);
            free(val_t);
            free(row_ptr);
            free(row_idx);
            free(col_ptr);
            free(col_idx);
        }
    }

    void clear_space() {
        free(val);
        free(val_t);
        free(row_ptr);
        free(row_idx);
        free(col_ptr);
        free(col_idx);
        mem_alloc_by_me = false;
    }

    smat_t transpose() {
        smat_t mt;
        mt.cols = rows;
        mt.rows = cols;
        mt.nnz = nnz;
        mt.val = val_t;
        mt.val_t = val;
        mt.nbits_val = nbits_val_t;
        mt.nbits_val_t = nbits_val;

        mt.col_ptr = row_ptr;
        mt.row_ptr = col_ptr;
        mt.nbits_col_ptr = nbits_row_ptr;
        mt.nbits_row_ptr = nbits_col_ptr;
        mt.col_idx = row_idx;
        mt.row_idx = col_idx;
        mt.nbits_col_idx = nbits_row_idx;
        mt.nbits_row_idx = nbits_col_idx;
        mt.max_col_nnz = max_row_nnz;
        mt.max_row_nnz = max_col_nnz;
        return mt;
    }
};

#endif //PMF_UTIL_H
