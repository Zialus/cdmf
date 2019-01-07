#ifndef PMF_UTIL_H
#define PMF_UTIL_H

#include "util.h"

typedef std::vector<VALUE_TYPE> vec_t;
typedef std::vector<vec_t> mat_t;

class rate_t {
public:
    unsigned i, j;
    VALUE_TYPE v, weight;

    rate_t(unsigned ii = 0, unsigned jj = 0, VALUE_TYPE vv = 0, VALUE_TYPE ww = 1.0) : i(ii), j(jj), v(vv), weight(ww) {}
};

class entry_iterator_t {
private:
    FILE* fp;
    char buf[1000]{};
public:
    bool with_weights;
    size_t nnz;

    entry_iterator_t(size_t nnz_, const char* filename, bool with_weights_ = false) {
        nnz = nnz_;
        fp = fopen(filename, "r");
        with_weights = with_weights_;
    }

    virtual rate_t next() {
        unsigned i = 1, j = 1;
        VALUE_TYPE v = 0, w = 1.0;
        if (nnz > 0) {
            CHECK_FGETS(fgets(buf, 1000, fp));
            if (with_weights) {
                (sizeof(VALUE_TYPE) == 8) ? (sscanf(buf, "%u %u %lf %lf", &i, &j, &v, &w)) : (sscanf(buf, "%u %u %f %f", &i, &j, &v, &w));
            } else {
                (sizeof(VALUE_TYPE) == 8) ? (sscanf(buf, "%u %u %lf", &i, &j, &v)) : (sscanf(buf, "%u %u %f", &i, &j, &v));
            }
            --nnz;
        } else {
            fprintf(stderr, "Error: no more entry to iterate !!\n");
        }
        return {i - 1, j - 1, v, w};
    }

    virtual ~entry_iterator_t() {
        if (fp) { fclose(fp); }
    }
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
// Access column fomat only when you use it..
class smat_t {
public:
    unsigned rows, cols;
    unsigned nnz, max_row_nnz, max_col_nnz;
    VALUE_TYPE* val, * val_t;
    size_t nbits_val, nbits_val_t;
    VALUE_TYPE* weight, * weight_t;
    size_t nbits_weight, nbits_weight_t;
    unsigned* col_ptr, * row_ptr;
    size_t nbits_col_ptr, nbits_row_ptr;
    unsigned* col_nnz, * row_nnz;
    size_t nbits_col_nnz, nbits_row_nnz;
    unsigned* row_idx, * col_idx;    // condensed
    size_t nbits_row_idx, nbits_col_idx;
    unsigned* colMajored_sparse_idx;
    size_t nbits_colMajored_sparse_idx;
    bool mem_alloc_by_me, with_weights;

    smat_t() : mem_alloc_by_me(false), with_weights(false) {}

    smat_t(const smat_t& m) {
        *this = m;
        mem_alloc_by_me = false;
    }

    void load(unsigned _rows, unsigned _cols, unsigned _nnz, const char* filename, bool ifALS, bool use_weights = false) {
        entry_iterator_t entry_it(_nnz, filename, use_weights);
        load_from_iterator(_rows, _cols, _nnz, &entry_it, ifALS);
    }

    void load_from_iterator(unsigned _rows, unsigned _cols, unsigned _nnz, entry_iterator_t* entry_it, bool ifALS) {
        rows = _rows;
        cols = _cols;
        nnz = _nnz;
        mem_alloc_by_me = true;
        with_weights = entry_it->with_weights;
        val = MALLOC(VALUE_TYPE, nnz);
        val_t = MALLOC(VALUE_TYPE, nnz);
        nbits_val = SIZEBITS(VALUE_TYPE, nnz);
        nbits_val_t = SIZEBITS(VALUE_TYPE, nnz);
        if (with_weights) {
            weight = MALLOC(VALUE_TYPE, nnz);
            weight_t = MALLOC(VALUE_TYPE, nnz);
            nbits_weight = SIZEBITS(VALUE_TYPE, nnz);
            nbits_weight_t = SIZEBITS(VALUE_TYPE, nnz);
        }
        row_idx = MALLOC(unsigned, nnz);
        col_idx = MALLOC(unsigned, nnz);  // switch to this for memory
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
        VALUE_TYPE* tmp_weight = weight;
        for (unsigned idx = 0; idx < _nnz; idx++) {
            rate_t rate = entry_it->next();
            row_ptr[rate.i + 1]++;
            col_ptr[rate.j + 1]++;
            tmp_row_idx[idx] = rate.i;
            tmp_col_idx[idx] = rate.j;
            tmp_val[idx] = rate.v;
            if (with_weights) {
                tmp_weight[idx] = rate.weight;
            }
            perm[idx] = idx;
        }
        // sort entries into row-majored ordering
        sort(perm.begin(), perm.end(), SparseComp(tmp_row_idx, tmp_col_idx, true));

        // Generate CRS format
        for (unsigned idx = 0; idx < _nnz; idx++) {
            val_t[idx] = tmp_val[perm[idx]];
            col_idx[idx] = tmp_col_idx[perm[idx]];
            if (with_weights) {
                weight_t[idx] = tmp_weight[idx];
            }
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
                if (with_weights) { weight[col_ptr[c]] = weight_t[i]; }
                col_ptr[c]++;
            }
        }
        for (unsigned c = cols; c > 0; --c) { col_ptr[c] = col_ptr[c - 1]; }
        col_ptr[0] = 0;

        if (ifALS) {
            unsigned * mapIDX;
            mapIDX = MALLOC(unsigned , rows);
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

    void remove_bias(VALUE_TYPE bias = 0) {
        if (bias) {
            for (unsigned i = 0; i < nnz; ++i) { val[i] -= bias; }
            for (unsigned i = 0; i < nnz; ++i) { val_t[i] -= bias; }
        }
    }

    void free(void* ptr) { if (ptr) { ::free(ptr); }}

    ~smat_t() {
        if (mem_alloc_by_me) {
            //puts("Warning: Somebody just freed me.");
            free(val);
            free(val_t);
            free(row_ptr);
            free(row_idx);
            free(col_ptr);
            free(col_idx);
            if (with_weights) {
                free(weight);
                free(weight_t);
            }
        }
    }

    void clear_space() {
        free(val);
        free(val_t);
        free(row_ptr);
        free(row_idx);
        free(col_ptr);
        free(col_idx);
        if (with_weights) {
            free(weight);
            free(weight_t);
        }
        mem_alloc_by_me = false;
        with_weights = false;
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
        mt.with_weights = with_weights;

        mt.weight = weight_t;
        mt.weight_t = weight;
        mt.nbits_weight = nbits_weight_t;
        mt.nbits_weight_t = nbits_weight;
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
