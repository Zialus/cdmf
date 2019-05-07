#ifndef UTIL_H
#define UTIL_H

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))
#define SIZEBITS(type, size) sizeof(type)*(size)

#define CL_CHECK(res) \
    {if (res != CL_SUCCESS) {fprintf(stderr,"Error \"%s\" (%d) in file %s on line %d\n", \
        get_error_string(res), res, __FILE__,__LINE__); abort();}}

#define CHECK_FSCAN(err, num)    if(err != num){ \
    fprintf(stderr,"FSCANF read %d, needed %d, in file %s on line %d\n", err, num,__FILE__,__LINE__); \
    abort(); \
}

#define CHECK_FREAD(err, num)    if(err != num){ \
    fprintf(stderr,"FREAD read %zu, needed %d, in file %s on line %d\n", err, num,__FILE__,__LINE__); \
    abort(); \
}


#endif //UTIL_H
