#include <cstring>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <string>
#include <fstream>

#include "util.h"

using namespace std;

// load utility for CCS RCS
void load(const char* srcdir, smat_t &R, bool ifALS, bool with_weights)
{
    char filename[1024], buf[1024];
    sprintf(filename,"%s/meta",srcdir);
    FILE *fp = fopen(filename,"r");
    unsigned m, n, nnz;
    fscanf(fp, "%u %u", &m, &n);
    fscanf(fp, "%u %s", &nnz, buf);
    sprintf(filename,"%s/%s", srcdir, buf);
    R.load(m, n, nnz, filename, ifALS, with_weights);
    fclose(fp);
    return ;
}


int convertToString(const char *filename,string &s)
{
    size_t size;
    char* str;
    fstream f(filename,(fstream::in|fstream::binary));

    if(f.is_open())
    {
        size_t fileSize;
        f.seekg(0,fstream::end);
        size=fileSize=(size_t)f.tellg();
        f.seekg(0,fstream::beg);
        str=new char[size+1];
        if(!str)
        {
            f.close();
            return 0;
        }
        f.read(str,fileSize);
        f.close();
        str[size]='\0';
        s=str;
        delete[] str;
        return 0;
    }
    cout<<"Error:failed to open file:"<<filename<<"\n";
    return -1;
}

int getPlatform(cl_platform_id &platform, int id)
{
    platform= nullptr;
    cl_uint numPlatforms;
    cl_int status=clGetPlatformIDs(0, nullptr,&numPlatforms);
    if(status!=CL_SUCCESS)
    {
        cout<<"ERROR:Getting platforms!\n";
        return -1;
    }
    if(numPlatforms>0)
    {
        cl_platform_id* platforms=(cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
        status=clGetPlatformIDs(numPlatforms,platforms, nullptr);
        if(status!=CL_SUCCESS)
        {
            cout<<"ERROR:Getting platform IDs!\n";
            return -1;
        }
        platform=platforms[id];
        free(platforms);
        return 0;
    }
    else
        return -1;
}

cl_device_id *getCl_device_id(cl_platform_id &platform, char* device_type)
{
    cl_uint numDevices=0;
    cl_int status = 0;
    cl_device_id *devices= nullptr;

    if((device_type[0]=='m')&&(device_type[1]=='i')&&(device_type[2]=='c'))
        status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_ACCELERATOR,0, nullptr,&numDevices);
    else if((device_type[0]=='c')&&(device_type[1]=='p')&&(device_type[2]=='u'))
        status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_CPU,0, nullptr,&numDevices);
    else if((device_type[0]=='g')&&(device_type[1]=='p')&&(device_type[2]=='u'))
        status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,0, nullptr,&numDevices);

    if(status!=CL_SUCCESS)
    {
        cout<<"ERROR:Getting numDevices from clGetDeviceIDs!\n";
        exit(1);
    }

    if(numDevices>0)
    {
        devices=(cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
        if((device_type[0]=='c')&&(device_type[1]=='p')&&(device_type[2]=='u'))
            status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_CPU,numDevices,devices, nullptr);
        else if((device_type[0]=='m')&&(device_type[1]=='i')&&(device_type[2]=='c'))
            status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_ACCELERATOR,numDevices,devices, nullptr);
        else if((device_type[0]=='g')&&(device_type[1]=='p')&&(device_type[2]=='u'))
            status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,numDevices,devices, nullptr);
    }

    if(status!=CL_SUCCESS)
    {
        cout<<"ERROR:Getting devices from clGetDeviceIDs!\n";
        exit(1);
    }
    return devices;
}

parameter parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name, char *kernel_code)
{
    parameter param;   // default values have been set by the constructor   
    int i;

    // parse options
    for(i=1;i<argc;i++)
    {
        if (argv[i][0] != '-'){
            break;
        }
        if (++i >= argc){
            exit_with_help();
        }
        if (strcmp(argv[i - 1], "-nBlocks") == 0){
            param.nBlocks = atoi(argv[i]);
        }
        else if (strcmp(argv[i - 1], "-nThreadsPerBlock") == 0){
            param.nThreadsPerBlock = atoi(argv[i]);
        }
        else if (strcmp(argv[i - 1], "-Cuda") == 0){
            param.enable_cuda = true;
            --i;
        }
        else if (strcmp(argv[i - 1], "-runOriginal") == 0){
            param.solver_type = 1;
            --i;
        }else if (strcmp(argv[i - 1], "-ALS") == 0){
            param.solver_type = 2;
            --i;
        }
        else{
            switch (argv[i - 1][1])
            {
                //case 's':
                //  param.solver_type = atoi(argv[i]);
                //  if (param.solver_type == 0){
                //      param.solver_type = CCDR1;
                //  }
                //  break;

                case 'c':
                    //param.k = atoi(argv[i]);
                    sprintf(kernel_code,"%s", argv[i]);
                    break;

                case 'k':
                    param.k = atoi(argv[i]);
                    break;

                case 'n':
                    param.threads = atoi(argv[i]);
                    break;

                case 'l':
                    param.lambda = atof(argv[i]);
                    break;

                case 'r':
                    param.rho = atof(argv[i]);
                    break;

                case 't':
                    param.maxiter = atoi(argv[i]);
                    break;

                case 'T':
                    param.maxinneriter = atoi(argv[i]);
                    break;

                case 'e':
                    param.eps = atof(argv[i]);
                    param.eta0 = atof(argv[i]);
                    break;

                case 'B':
                    param.num_blocks = atoi(argv[i]);
                    break;

                case 'm':
                    param.lrate_method = atoi(argv[i]);
                    break;

                case 'u':
                    param.betaup = atof(argv[i]);
                    break;

                case 'd':
                    param.betadown = atof(argv[i]);
                    break;

                case 'P':
                    param.platform_id = atoi(argv[i]);
                    //param.do_predict = atoi(argv[i]);
                    break;
                case 'p':
                    //param.platform_id = atoi(argv[i]);
                    param.do_predict = atoi(argv[i]);
                    break;

                case 'q':
                    param.verbose = atoi(argv[i]);
                    break;

                case 'V':
                    param.version = atoi(argv[i]);
                    break;

                case 'N':
                    param.do_nmf = atoi(argv[i]) == 1;
                    break;

                    //case 'C':
                    //  param.enable_cuda = atoi(argv[i]) == 1 ? true : false;
                    //  break;


                default:
                    fprintf(stderr, "unknown option: -%c\n", argv[i - 1][1]);
                    exit_with_help();
                    break;
            }
        }

    }

    //if (param.do_predict!=0) 
    //  param.verbose = 1;

    // determine filenames
    if(i>=argc)
        exit_with_help();

    int toCut = 0;//begin remove exe____ Andre
    for (int index=strlen(argv[0])-1;index>0;index--){
        toCut++;
        if (argv[0][index]=='\\'|| argv[0][index]=='/'){
            index = 0;
        }
    }
    char src[5120], dest[5120];
    strcpy(src,  argv[0]);
    src[strlen(argv[0]) - toCut+1] = '\0';
    argv[0] = src;
    sprintf(input_file_name, "%s",argv[i]);
    return param;
}

void exit_with_help(){
    printf(
            "Usage: omp-pmf-train [options] data_dir [model_filename]\n"
            "options:\n"
            "    -s type : set type of solver (default 0)\n"
            "    -c : full path to the kernel code (default x)\n"
            "        0 -- CCDR1 with fundec stopping condition\n"
            "    -k rank : set the rank (default 10)\n"
            "    -n threads : set the number of threads (default 4)\n"
            "    -l lambda : set the regularization parameter lambda (default 0.1)\n"
            "    -t max_iter: set the number of iterations (default 5)\n"
            "    -T max_iter: set the number of inner iterations used in CCDR1 (default 5)\n"
            "    -e epsilon : set inner termination criterion epsilon of CCDR1 (default 1e-3)\n"
            "    -p platform_id: select a platform (default 0)\n"
            "    -q verbose: show information or not (default 0)\n"
            "    -N do_nmf: do nmf (default 0)\n"
            "    -runOriginal: Flag to run libpmf original implementation\n"
            "    -Cuda: Flag to enable cuda\n"
            "    -nBlocks: Number of blocks on cuda (default 16)\n"
            "    -nThreadsPerBlock: Number of threads per block on cuda (default 32)\n"
            "    -ALS: Flag to enable ALS algorithm, if not present CCD++ is used\n"

            );
    exit(1);
}

const char* get_error_string(cl_int err)
{
    switch(err){
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}



VALUE_TYPE dot(const vec_t &a, const vec_t &b)
{
    VALUE_TYPE ret = 0;
#pragma omp parallel for 
    for(int i = a.size()-1; i >=0; --i)
        ret+=a[i]*b[i];
    return ret;
}
VALUE_TYPE dot(const mat_t &W, const int i, const mat_t &H, const int j)
{
    int k = W.size();
    VALUE_TYPE ret = 0;
    for(int t = 0; t < k; ++t)
        ret+=W[t][i]*H[t][j];
    return ret;
}
VALUE_TYPE dot(const mat_t &W, const int i, const vec_t &H_j)
{
    int k = H_j.size();
    VALUE_TYPE ret = 0;
    for(int t = 0; t < k; ++t)
        ret+=W[t][i]*H_j[t];
    return ret;
}
VALUE_TYPE calrmse(testset_t &testset, const mat_t &W, const mat_t &H, bool iscol)
{
    size_t nnz = testset.nnz;
    VALUE_TYPE rmse = 0, err;
    for(size_t idx = 0; idx < nnz; ++idx)
    {
        err = -testset[idx].v;
        if(iscol)
            err += dot(W, testset[idx].i, H, testset[idx].j);
        else 
            err += dot(W[testset[idx].i], H[testset[idx].j]);
        rmse += err*err;
    }
    return sqrt(rmse/nnz);
}

VALUE_TYPE calrmse_r1(testset_t &testset, vec_t &Wt, vec_t &Ht)
{
    size_t nnz = testset.nnz;
    VALUE_TYPE rmse = 0, err;
#pragma omp parallel for reduction(+:rmse)
    for(size_t idx = 0; idx < nnz; ++idx)
    {
        testset[idx].v -= Wt[testset[idx].i]*Ht[testset[idx].j];
        rmse += testset[idx].v*testset[idx].v;
    }
    return sqrt(rmse/nnz);
}

VALUE_TYPE calrmse_r1(testset_t &testset, vec_t &Wt, vec_t &Ht, vec_t &oldWt, vec_t &oldHt)
{
    size_t nnz = testset.nnz;
    VALUE_TYPE rmse = 0, err;
#pragma omp parallel for reduction(+:rmse)
    for(size_t idx = 0; idx < nnz; ++idx)
    {
        testset[idx].v -= Wt[testset[idx].i]*Ht[testset[idx].j] - oldWt[testset[idx].i]*oldHt[testset[idx].j];
        rmse += testset[idx].v*testset[idx].v;
    }
    return sqrt(rmse/nnz);
}

void initial_col(mat_t &X, unsigned k, unsigned n){
    X = mat_t(k, vec_t(n));
    srand(0L);
    for(unsigned i = 0; i < n; ++i)
        for(unsigned j = 0; j < k; ++j)
            X[j][i] = 0.1*(VALUE_TYPE(rand()) / RAND_MAX)+0.001;
//            X[j][i] = 0.01;
}

void golden_compare(mat_t W, mat_t W_ref, unsigned k, unsigned m)
{
    int error_count = 0;
    for (unsigned i = 0; i < k; i++)
    {
        for(unsigned j = 0; j < m; j++)
            if (fabs((double)W[i][j] - (double)W_ref[i][j]) > 0.1 * fabs((double)W_ref[i][j]))
            {
//                cout << i << "|" << j << "\t";
//                cout << W[i][j] << "," << W_ref[i][j] << "\t";
                error_count++;
            }
//        cout << endl;
    }
//    cout << endl;
    if (error_count == 0)
        cout << "Check... PASS!" << endl;
    else
        cout << "Check... NO PASS! #Error = " << error_count << " out of " << k*m << " entries." << endl;
}
