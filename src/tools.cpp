#include "tools.h"
#include "util.h"

// load utility for CCS RCS
void load(const char* srcdir, smat_t& R, bool ifALS, bool with_weights) {
    char filename[2048], buf[1024];
    sprintf(filename, "%s/meta", srcdir);
    FILE* fp = fopen(filename, "r");
    if (fp == nullptr) {
        printf("Can't open input file.\n");
        exit(1);
    }
    unsigned m, n, nnz;
    CHECK_FSCAN(fscanf(fp, "%u %u", &m, &n), 2);
    CHECK_FSCAN(fscanf(fp, "%u %1023s", &nnz, buf), 2);
    sprintf(filename, "%s/%s", srcdir, buf);
    R.load(m, n, nnz, filename, ifALS, with_weights);
    fclose(fp);
}

void convertToString(const char* filename, std::string& s) {
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));

    if (f.is_open()) {
        size_t fileSize;
        size_t size;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t) f.tellg();
        f.seekg(0, std::fstream::beg);
        char* str = new char[size + 1];
        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        s = str;
        delete[] str;
    } else {
        std::cout << "Error:failed to open file:" << filename << "\n";
        exit(EXIT_FAILURE);
    }
}

void print_all_the_info() {
    // get platform count
    cl_uint platformCount;
    clGetPlatformIDs(0, nullptr, &platformCount);
    // get all platforms
    cl_platform_id* platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, nullptr);

    for (unsigned i = 0; i < platformCount; i++) {
        printf("%u. Platform \n", i + 1);

        print_platform_info(platforms, i);

        // get device count
        cl_uint deviceCount;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
        // get all devices
        cl_device_id* devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, nullptr);

        // for each device print critical attributes
        for (unsigned j = 0; j < deviceCount; j++) {
            print_device_info(devices, j);
        }

        printf("\n");
        free(devices);
    }

    free(platforms);
}

void print_device_info(cl_device_id* devices, unsigned int j) {
    char* value;
    size_t valueSize;
    cl_uint maxComputeUnits;

    // print device name
    clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, nullptr, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, nullptr);
    printf("%u. Device: %s\n", j + 1, value);
    free(value);

    // print hardware device version
    clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, nullptr, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, nullptr);
    printf(" %u.%d Hardware version: %s\n", j + 1, 1, value);
    free(value);

    // print software driver version
    clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, nullptr, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, nullptr);
    printf(" %u.%d Software version: %s\n", j + 1, 2, value);
    free(value);

    // print c version supported by compiler for device
    clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, nullptr, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, nullptr);
    printf(" %u.%d OpenCL C version: %s\n", j + 1, 3, value);
    free(value);

    // print parallel compute units
    clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits,
                    nullptr);
    printf(" %u.%d Parallel compute units: %u\n", j + 1, 4, maxComputeUnits);

    clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 0, nullptr, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, valueSize, value, nullptr);
    printf(" %u.%d Vendor: %s\n", j + 1, 5, value);
    free(value);

    clGetDeviceInfo(devices[j], CL_DEVICE_PROFILE, 0, nullptr, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(devices[j], CL_DEVICE_PROFILE, valueSize, value, nullptr);
    printf(" %u.%d Profile: %s\n", j + 1, 6, value);
    free(value);

    clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, 0, nullptr, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, valueSize, value, nullptr);
    printf(" %u.%d Extension: %s\n", j + 1, 7, value);
    free(value);
}

void print_platform_info(cl_platform_id* platforms, unsigned int id) {
    const char* attributeNames[5] = {"Name", "Vendor",
                                     "Version", "Profile", "Extensions"};
    const cl_platform_info attributeTypes[5] = {CL_PLATFORM_NAME, CL_PLATFORM_VENDOR,
                                                CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS};
    const int attributeCount = sizeof(attributeNames) / sizeof(attributeNames[0]);

    for (unsigned j = 0; j < attributeCount; j++) {
        // get platform attribute value size
        size_t infoSize;
        clGetPlatformInfo(platforms[id], attributeTypes[j], 0, nullptr, &infoSize);
        char* info = (char*) malloc(infoSize);

        // get platform attribute value
        clGetPlatformInfo(platforms[id], attributeTypes[j], infoSize, info, nullptr);

        printf(" %u.%u %-11s: %s\n", id + 1, j + 1, attributeNames[j], info);
        free(info);
    }

    printf("\n");
}

int report_device(cl_device_id device_id) {
    cl_char device_name[1024] = {'\0'};

    int err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to retrieve device info! %s\n", get_error_string(err));
        return -1;
    }
    printf("Connecting to %s...\n", device_name);
    return 0;
}

cl_platform_id getPlatform(int id) {
    cl_int status;
    cl_uint numPlatforms;

    status = clGetPlatformIDs(0, nullptr, &numPlatforms);
    CL_CHECK(status);

    assert(numPlatforms > 0);
    cl_platform_id* platforms = (cl_platform_id*) malloc(numPlatforms * sizeof(cl_platform_id));

    status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
    CL_CHECK(status);

    cl_platform_id platform = platforms[id];
    free(platforms);
    return platform;
}

cl_device_id* getDevices(cl_platform_id& platform, char* device_type) {
    cl_int status = 0;
    cl_uint numDevices = 0;

    if (strcmp(device_type, "mic") == 0) {
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 0, nullptr, &numDevices);
    } else if (strcmp(device_type, "cpu") == 0) {
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &numDevices);
    } else if (strcmp(device_type, "gpu") == 0) {
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    }
    CL_CHECK(status);

    assert(numDevices > 0);
    cl_device_id* devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));

    if (strcmp(device_type, "mic") == 0) {
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, numDevices, devices, nullptr);
    } else if (strcmp(device_type, "cpu") == 0) {
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, nullptr);
    } else if (strcmp(device_type, "gpu") == 0) {
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, nullptr);
    }
    CL_CHECK(status);

    return devices;
}

parameter parse_command_line(int argc, char** argv, char* input_file_name, char* kernel_code) {
    // default values have been set by the constructor
    parameter param;
    // parse options
    int i;
    for (i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            break;
        }
        if (++i >= argc) {
            exit_with_help();
        }
        if (strcmp(argv[i - 1], "-nBlocks") == 0) {
            param.nBlocks = atoi(argv[i]);
        } else if (strcmp(argv[i - 1], "-nThreadsPerBlock") == 0) {
            param.nThreadsPerBlock = atoi(argv[i]);
        } else {
            switch (argv[i - 1][1]) {
                case 'c':
                    sprintf(kernel_code, "%s", argv[i]);
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
                case 't':
                    param.maxiter = atoi(argv[i]);
                    break;
                case 'T':
                    param.maxinneriter = atoi(argv[i]);
                    break;
                case 'e':
                    param.eps = atof(argv[i]);
                    break;
                case 'P':
                    param.platform_id = atoi(argv[i]);
                    break;
                case 'p':
                    param.do_predict = atoi(argv[i]);
                    break;
                case 'r':
                    param.do_ref = atoi(argv[i]);
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
                default:
                    fprintf(stderr, "unknown option: -%c\n", argv[i - 1][1]);
                    exit_with_help();
                    break;
            }
        }

    }

    if (i >= argc) {
        exit_with_help();
    }

    sprintf(input_file_name, "%s", argv[i]);
    return param;
}

void exit_with_help() {
    printf(
            "Usage: cdmf [options] data_dir\n"
            "options:\n"
            "    -c : full path to the kernel code (default x)\n"
            "    -k rank : set the rank (default 10)\n"
            "    -n threads : set the number of threads (default 4)\n"
            "    -l lambda : set the regularization parameter lambda (default 0.1)\n"
            "    -t max_iter: set the number of iterations (default 5)\n"
            "    -T max_iter: set the number of inner iterations used in CCDR1 (default 5)\n"
            "    -e epsilon : set inner termination criterion epsilon of CCDR1 (default 1e-3)\n"
            "    -P platform_id: select a platform (default 0)\n"
            "    -q verbose: show information or not (default 0)\n"
            "    -nBlocks: Number of blocks on cuda (default 16)\n"
            "    -nThreadsPerBlock: Number of threads per block on cuda (default 32)\n"
    );
    exit(1);
}

const char* get_error_string(cl_int err) {
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

void initial_col(mat_t& X, unsigned k, unsigned n) {
    X = mat_t(k, vec_t(n));
    srand(0L);
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = 0; j < k; ++j) {
            X[j][i] = 0.1 * (VALUE_TYPE(rand()) / RAND_MAX) + 0.001;
        }
    }
}

void golden_compare(mat_t W, mat_t W_ref, unsigned k, unsigned m) {
    int error_count = 0;
    for (unsigned i = 0; i < k; i++) {
        for (unsigned j = 0; j < m; j++) {
            if (fabs((double) W[i][j] - (double) W_ref[i][j]) > 0.1 * fabs((double) W_ref[i][j])) {
//                std::cout << i << "|" << j << "\t";
//                std::cout << W[i][j] << "," << W_ref[i][j] << "\t";
                error_count++;
            }
        }
//        std::cout << std::endl;
    }
//    std::cout << std::endl;
    if (error_count == 0) {
        std::cout << "Check... PASS!" << std::endl;
    } else {
        std::cout << "Check... NO PASS! #Error = " << error_count << " out of " << k * m << " entries." << std::endl;
    }
}

void calculate_rmse_ocl(const mat_t& W_c, const mat_t& H_c, const int k, const char* srcdir) {
    auto t5 = std::chrono::high_resolution_clock::now();
    int i, j;
    VALUE_TYPE vv, rmse = 0;
    int num_insts = 0;
    int nans_count = 0;

    char meta_filename[1024];
    sprintf(meta_filename, "%s/meta", srcdir);
    FILE* fp = fopen(meta_filename, "r");
    if (fp == nullptr) {
        printf("Can't open meta input file.\n");
        exit(1);
    }

    char buf_train[1024], buf_test[1024], test_file_name[2048], train_file_name[2048];
    unsigned m, n, nnz, nnz_test;
    CHECK_FSCAN(fscanf(fp, "%u %u", &m, &n),2);
    CHECK_FSCAN(fscanf(fp, "%u %1023s", &nnz, buf_train),2);
    CHECK_FSCAN(fscanf(fp, "%u %1023s", &nnz_test, buf_test),2);
    sprintf(test_file_name, "%s/%s", srcdir, buf_test);
    sprintf(train_file_name, "%s/%s", srcdir, buf_train);
    fclose(fp);

    FILE* test_fp = fopen(test_file_name, "r");
    if (test_fp == nullptr) {
        printf("Can't open test file.\n");
        exit(1);
    }

    while ((sizeof(VALUE_TYPE) == 8) ? (fscanf(test_fp, "%d %d %lf", &i, &j, &vv) != EOF) : (fscanf(test_fp, "%d %d %f", &i, &j, &vv) != EOF)) {
        VALUE_TYPE pred_v = 0;
        for (int t = 0; t < k; t++) {
            pred_v += W_c[t][i - 1] * H_c[t][j - 1];
        }
        num_insts++;
        VALUE_TYPE tmp = (pred_v - vv) * (pred_v - vv);
        if (tmp == tmp) {
            rmse += tmp;
        } else {
            nans_count++;
        }
//            printf("%d - %d,%d,%lf,%lf,%lf\n", num_insts-1,i,j, tmp, vv, pred_v);
    }
    fclose(test_fp);

    double nans_percentage = (double) nans_count / (double) num_insts;
    printf("NaNs percentage: %lf, NaNs Count: %d, Total Insts: %d\n", nans_percentage, nans_count, num_insts);
    rmse = sqrt(rmse / num_insts);
    printf("[info] test RMSE = %lf\n", rmse);
    auto t6 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT2 = t6 - t5;;
    printf("[info] Predict time: %lf s\n", deltaT2.count());

}

void print_matrix(mat_t M, unsigned k, unsigned n) {
    printf("-----------------------------------------\n");
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = 0; j < k; ++j) {
            printf("|%f", M[j][i]);
        }
        printf("\n-----------------------------------------\n");
    }
}
