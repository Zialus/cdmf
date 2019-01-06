#ifndef UTIL_H
#define UTIL_H

#include <cmath>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>

#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))
#define SIZEBITS(type, size) sizeof(type)*(size)

#define CL_CHECK(res) \
    {if (res != CL_SUCCESS) {fprintf(stderr,"Error \"%s\" (%d) in file %s on line %d\n", \
        get_error_string(res), res, __FILE__,__LINE__); abort();}}

#define CHECK_FSCAN(err, num)    if(err != num){ \
    perror("FSCANF"); \
    exit(EXIT_FAILURE); \
}

#define CHECK_FGETS(err)    if(err == nullptr){ \
    perror("FGETS"); \
    exit(EXIT_FAILURE); \
}

#endif // UTIL_H
