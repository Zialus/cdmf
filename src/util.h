#ifndef UTIL_H
#define UTIL_H

#include <cmath>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <iostream>
#include <algorithm>
#include <utility>
#include <map>
#include <queue>
#include <set>
#include <vector>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#include <sys/time.h>

//typedef float VALUE_TYPE;
//typedef double VALUE_TYPE;

using namespace std;

#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))
#define SIZEBITS(type, size) sizeof(type)*(size)
#define CL_CHECK(API) if((err=API)!=CL_SUCCESS){\
    printf("[err] id: %d, msg: %s, line: %d\n", err, get_error_string(err), __LINE__);\
    exit(-1); \
}

#define CHECK_ERROR(err)    if(err != CL_SUCCESS){ \
    printf ("[err] %s\n", get_error_string (err)); \
    exit(-1); \
}

enum {ROWMAJOR, COLMAJOR};
enum {CCDR1};
enum {BOLDDRIVER, EXPDECAY};

class smat_t;
class parameter;

typedef vector<VALUE_TYPE> vec_t;
typedef vector<vec_t> mat_t;

/* interface declaration */
void load(const char* srcdir, smat_t &R, bool ifALS, bool with_weights);
void initial_col(mat_t &X, unsigned int k, unsigned int n);

/** convert the kernel file into a string */
int convertToString(const char *filename,string& s);

/** Getting platforms and choose an available one.*/
int getPlatform(cl_platform_id &platform, int id);

/** Query the platform and choose the first GPU device if has one.*/
cl_device_id *getCl_device_id(cl_platform_id &platform, char* device_type);

/** parsing the commandline arguments **/
parameter parse_command_line(int argc, char **argv, char *input_file_name, char *kernel_code);

void exit_with_help();

const char* get_error_string(cl_int err);

inline double gettime() {
    struct timeval t;
    gettimeofday(&t, nullptr);
    return t.tv_sec + t.tv_usec * 1e-6;
}

inline char* getT(unsigned sz) {
    if (sz == 8) { return (char*) "double"; }
    if (sz == 4) { return (char*) "float"; }
    return (char*) "float";
}

template<typename iT, typename vT>
double getB(const iT m, const iT nnz) {
    return (double) ((m + 1 + nnz) * sizeof(iT) + (2 * nnz + m) * sizeof(vT));
}

template<typename iT>
double getFLOP(const iT nnz) {
    return (double) (2 * nnz);
}

template<typename T>
inline std::string to_string(T value) {
    std::ostringstream os;
    os << value;
    return os.str();
}

class parameter {
public:
    int version;
    unsigned int k;
    int threads;
    int maxinneriter;
    int maxiter;
    VALUE_TYPE lambda;
    VALUE_TYPE eps; // for the fundec stop-cond in ccdr1
    int do_predict;
    int verbose;
    int platform_id;
    int do_nmf;  // non-negative matrix factorization
    int nBlocks;
    int nThreadsPerBlock;
    int do_ref;

    parameter() {
        version = 1;
        k = 10;
        maxiter = 5;
        maxinneriter = 5;
        lambda = 0.1f;
        threads = 4;
        eps = 1e-3f;
        do_predict = 0;
        platform_id = 0;
        verbose = 0;
        do_ref = 0;
        do_nmf = 0;
        nBlocks = 16;
        nThreadsPerBlock = 32;
    }
};

class rate_t
{
    public:
        int i, j; VALUE_TYPE v, weight;
        rate_t(int ii=0, int jj=0, VALUE_TYPE vv=0, VALUE_TYPE ww=1.0): i(ii), j(jj), v(vv), weight(ww){}
};

class entry_iterator_t
{
    private:
        FILE *fp;
        char buf[1000];
    public:
        unsigned int with_weights;
        size_t nnz;
        entry_iterator_t():nnz(0),fp(nullptr), with_weights(false){}
        entry_iterator_t(size_t nnz_, const char* filename, unsigned int with_weights_=false) 
        {
            nnz = nnz_;
            fp = fopen(filename,"r");
            with_weights = with_weights_;
        }
        size_t size() {return nnz;}
        virtual rate_t next() 
        {
            int i = 1, j = 1;
            VALUE_TYPE v = 0, w = 1.0;
            if (nnz > 0)    
            {
                fgets(buf, 1000, fp);
                if (with_weights)
                    (sizeof(VALUE_TYPE)==8)?(sscanf(buf, "%d %d %lf %lf", &i, &j, &v, &w)):(sscanf(buf, "%d %d %f %f", &i, &j, &v, &w));
                else
                    (sizeof(VALUE_TYPE)==8)?(sscanf(buf, "%d %d %lf", &i, &j, &v)):(sscanf(buf, "%d %d %f", &i, &j, &v));
                --nnz;
            } 
            else 
            {
                fprintf(stderr,"Error: no more entry to iterate !!\n");
            }
            return rate_t(i-1,j-1,v,w);
        }
        virtual ~entry_iterator_t()
        {
            if (fp) fclose(fp);
        }
};

class SparseComp 
{
    public:
        const unsigned *row_idx;
        const unsigned *col_idx;
        SparseComp(const unsigned *row_idx_, const unsigned *col_idx_, unsigned int isRCS_=true) 
        {
            row_idx = (isRCS_)? row_idx_: col_idx_;
            col_idx = (isRCS_)? col_idx_: row_idx_;
        }
        unsigned int operator()(size_t x, size_t y) const {
            return  (row_idx[x] < row_idx[y]) || ((row_idx[x] == row_idx[y]) && (col_idx[x]<= col_idx[y]));
        }
};

// Sparse matrix format CCS & RCS
// Access column fomat only when you use it..
class smat_t
{
    public:
        unsigned int rows, cols;
        unsigned int nnz, max_row_nnz, max_col_nnz;
        VALUE_TYPE *val, *val_t;
        size_t nbits_val, nbits_val_t;
        VALUE_TYPE *weight, *weight_t;
        size_t nbits_weight, nbits_weight_t;
        unsigned int *col_ptr, *row_ptr;
        size_t nbits_col_ptr, nbits_row_ptr;
        unsigned int *col_nnz, *row_nnz;
        size_t nbits_col_nnz, nbits_row_nnz;
        unsigned *row_idx, *col_idx;    // condensed
        size_t nbits_row_idx, nbits_col_idx;
        unsigned *colMajored_sparse_idx;
        size_t nbits_colMajored_sparse_idx;
        //unsigned unsigned int *row_idx, *col_idx; // for matlab
        bool mem_alloc_by_me;
        bool with_weights;
        smat_t():mem_alloc_by_me(false), with_weights(false){ }
        smat_t(const smat_t& m){ *this = m; mem_alloc_by_me = false;}

        // For matlab (Almost deprecated)
        smat_t(unsigned int m, unsigned int n, unsigned *ir, unsigned int *jc, VALUE_TYPE *v, unsigned *ir_t, unsigned int *jc_t, VALUE_TYPE *v_t):
            rows(m), cols(n), mem_alloc_by_me(false),
            row_idx(ir), col_ptr(jc), val(v), col_idx(ir_t), row_ptr(jc_t), val_t(v_t) 
        {
            if(col_ptr[n] != row_ptr[m])
                fprintf(stderr,"Error occurs! two nnz do not match (%d, %d)\n", col_ptr[n], row_ptr[n]);
            nnz = col_ptr[n];
            max_row_nnz = max_col_nnz = 0;
            for(unsigned int r=1; r<=rows; ++r)
                max_row_nnz = max(max_row_nnz, row_ptr[r]);
            for(unsigned int c=1; c<=cols; ++c)
                max_col_nnz = max(max_col_nnz, col_ptr[c]);
        }
        void load(unsigned int _rows, unsigned int _cols, unsigned int _nnz, const char* filename, unsigned int ifALS, unsigned int with_weights = false)
        {
            entry_iterator_t entry_it(_nnz, filename, with_weights);
            load_from_iterator(_rows, _cols, _nnz, &entry_it, ifALS);
        }
        void load_from_iterator(unsigned int _rows, unsigned int _cols, unsigned int _nnz, entry_iterator_t* entry_it, unsigned int ifALS) 
        {
            rows =_rows,cols=_cols,nnz=_nnz;
            mem_alloc_by_me = true;
            with_weights = entry_it->with_weights;
            val = MALLOC(VALUE_TYPE, nnz); val_t = MALLOC(VALUE_TYPE, nnz);
            nbits_val = SIZEBITS(VALUE_TYPE, nnz); nbits_val_t = SIZEBITS(VALUE_TYPE, nnz);
            if(with_weights) 
            {
                weight = MALLOC(VALUE_TYPE, nnz);
                weight_t = MALLOC(VALUE_TYPE, nnz);
                nbits_weight = SIZEBITS(VALUE_TYPE, nnz);
                nbits_weight_t = SIZEBITS(VALUE_TYPE, nnz);
            }
            row_idx = MALLOC(unsigned, nnz); col_idx = MALLOC(unsigned, nnz);  // switch to this for memory
            nbits_row_idx = SIZEBITS(unsigned, nnz); nbits_col_idx = SIZEBITS(unsigned, nnz);
            row_ptr = MALLOC(unsigned int, rows+1); col_ptr = MALLOC(unsigned int, cols+1);
            nbits_row_ptr = SIZEBITS(unsigned int, rows + 1); nbits_col_ptr = SIZEBITS(unsigned int, cols + 1);
            memset(row_ptr,0,sizeof(unsigned int)*(rows+1));
            memset(col_ptr,0,sizeof(unsigned int)*(cols+1));
            if (ifALS){
                colMajored_sparse_idx = MALLOC(unsigned, nnz);
                nbits_colMajored_sparse_idx = SIZEBITS(unsigned, nnz);
            }

            // a trick here to utilize the space the have been allocated
            vector<size_t> perm(_nnz);
            unsigned *tmp_row_idx = col_idx;
            unsigned *tmp_col_idx = row_idx;
            VALUE_TYPE *tmp_val = val;
            VALUE_TYPE *tmp_weight = weight;
            for(size_t idx = 0; idx < _nnz; idx++)
            {
                rate_t rate = entry_it->next();
                row_ptr[rate.i+1]++;
                col_ptr[rate.j+1]++;
                tmp_row_idx[idx] = rate.i;
                tmp_col_idx[idx] = rate.j;
                tmp_val[idx] = rate.v;
                if(with_weights)
                    tmp_weight[idx] = rate.weight;
                perm[idx] = idx;
            }
            // sort entries into row-majored ordering
            sort(perm.begin(), perm.end(), SparseComp(tmp_row_idx, tmp_col_idx, true));
            
            // Generate CRS format
            for(size_t idx = 0; idx < _nnz; idx++) {
                val_t[idx] = tmp_val[perm[idx]];
                col_idx[idx] = tmp_col_idx[perm[idx]];
                if(with_weights)
                    weight_t[idx] = tmp_weight[idx];
            }
    
            // Calculate nnz for each row and col
            max_row_nnz = max_col_nnz = 0;
            for(unsigned int r=1; r<=rows; ++r) {
                max_row_nnz = max(max_row_nnz, row_ptr[r]);
                row_ptr[r] += row_ptr[r-1];
            }
            for(unsigned int c=1; c<=cols; ++c) {
                max_col_nnz = max(max_col_nnz, col_ptr[c]);
                col_ptr[c] += col_ptr[c-1];
            }
    
            // Transpose CRS into CCS matrix
            for(unsigned int r=0; r<rows; ++r){
                for(unsigned int i = row_ptr[r]; i < row_ptr[r+1]; ++i){
                    unsigned int c = col_idx[i];
                    row_idx[col_ptr[c]] = r;
                    val[col_ptr[c]] = val_t[i];
                    if(with_weights) weight[col_ptr[c]] = weight_t[i];
                    col_ptr[c]++;
                }
            }
            for(unsigned int c=cols; c>0; --c) col_ptr[c] = col_ptr[c-1];
            col_ptr[0] = 0;

            if (ifALS)
            {
                unsigned *mapIDX = MALLOC(unsigned, rows);
                for (int r = 0; r < rows; ++r)
                {
                    mapIDX[r] = row_ptr[r];
                }

                for (int r = 0; r < nnz; ++r)
                {
                    colMajored_sparse_idx[mapIDX[row_idx[r]]] = r;
                    ++mapIDX[row_idx[r]];
                }
                free(mapIDX);
            }
        }
        unsigned int nnz_of_row(int i) const {return (row_ptr[i+1]-row_ptr[i]);}
        unsigned int nnz_of_col(int i) const {return (col_ptr[i+1]-col_ptr[i]);}
        VALUE_TYPE get_global_mean()
        {
            VALUE_TYPE sum=0;
            for(unsigned int i=0;i<nnz;++i) sum+=val[i];
            return sum/nnz;
        }
        void remove_bias(VALUE_TYPE bias=0)
        {
            if(bias) 
            {
                for(unsigned int i=0;i<nnz;++i) val[i]-=bias;
                for(unsigned int i=0;i<nnz;++i) val_t[i]-=bias;
            }
        }
        void free(void *ptr) {if(ptr) ::free(ptr);}

        ~smat_t(){
            if(mem_alloc_by_me) 
            {
                //puts("Warnning: Somebody just free me.");
                free(val); free(val_t);
                free(row_ptr);free(row_idx);
                free(col_ptr);free(col_idx);
                if(with_weights) { free(weight); free(weight_t);}
            }
        }
        void clear_space() 
        {
            free(val); free(val_t);
            free(row_ptr);free(row_idx);
            free(col_ptr);free(col_idx);
            if(with_weights) { free(weight); free(weight_t);}
            mem_alloc_by_me = false;
            with_weights = false;

        }
        smat_t transpose()
        {
            smat_t mt;
            mt.cols = rows; mt.rows = cols; mt.nnz = nnz;
            mt.val = val_t; mt.val_t = val;
            mt.nbits_val = nbits_val_t; mt.nbits_val_t = nbits_val;
            mt.with_weights = with_weights;

            mt.weight = weight_t; mt.weight_t = weight;
            mt.nbits_weight = nbits_weight_t; mt.nbits_weight_t = nbits_weight;
            mt.col_ptr = row_ptr; mt.row_ptr = col_ptr;
            mt.nbits_col_ptr = nbits_row_ptr; mt.nbits_row_ptr = nbits_col_ptr;
            mt.col_idx = row_idx; mt.row_idx = col_idx;
            mt.nbits_col_idx = nbits_row_idx; mt.nbits_row_idx = nbits_col_idx;
            mt.max_col_nnz=max_row_nnz; mt.max_row_nnz=max_col_nnz;
            return mt;
        }
};

// Test set format
class testset_t
{
    public:
        unsigned rows, cols, nnz;
        vector<rate_t> T;
        testset_t(): rows(0), cols(0), nnz(0){}
        inline rate_t& operator[](const unsigned &idx) {return T[idx];}
        void load(unsigned _rows, unsigned _cols, unsigned _nnz, const char *filename) 
        {
            int r, c; 
            VALUE_TYPE v;
            rows = _rows; cols = _cols; nnz = _nnz;
            T = vector<rate_t>(nnz);
            FILE *fp = fopen(filename, "r");
            for(unsigned idx = 0; idx < nnz; ++idx){
                fscanf(fp, "%d %d %lg", &r, &c, &v); 
                T[idx] = rate_t(r-1,c-1,v);
            }
            fclose(fp);
        }
        void load_from_iterator(unsigned _rows, unsigned _cols, unsigned _nnz, entry_iterator_t* entry_it)
        { 
            rows =_rows,cols=_cols,nnz=_nnz;
            T = vector<rate_t>(nnz);
            for(size_t idx=0; idx < nnz; ++idx) 
                T[idx] = entry_it->next();
        }
        VALUE_TYPE get_global_mean()
        {
            VALUE_TYPE sum=0;
            for(unsigned i=0; i<nnz; ++i) sum+=T[i].v;
            return sum/nnz;
        }
        void remove_bias(VALUE_TYPE bias=0)
        {
            if(bias) for(unsigned i=0; i<nnz; ++i) T[i].v-=bias;
        }
};

VALUE_TYPE dot(const vec_t &a, const vec_t &b);
VALUE_TYPE dot(const mat_t &W, const int i, const mat_t &H, const int j);
VALUE_TYPE dot(const mat_t &W, const int i, const vec_t &H_j);
VALUE_TYPE calrmse(testset_t &testset, const mat_t &W, const mat_t &H, bool iscol=false);
VALUE_TYPE calrmse_r1(testset_t &testset, vec_t &Wt, vec_t &H_t);
VALUE_TYPE calrmse_r1(testset_t &testset, vec_t &Wt, vec_t &Ht, vec_t &oldWt, vec_t &oldHt);
void golden_compare(mat_t W, mat_t W_ref, unsigned k, unsigned m);

#endif // UTIL_H

