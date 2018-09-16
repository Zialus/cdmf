#ifndef UTILS_H
#define UTILS_H

#include <sys/time.h>
#include <sys/types.h>
#include <dirent.h>

#include "common.h"

/*template<typename iT, typename vT>
double getB(const iT m, const iT nnz)
{
    return (double)((m + 1 + nnz) * sizeof(iT) + (2 * nnz + m) * sizeof(vT));
}

template<typename iT>
double getFLOP(const iT nnz)
{
    return (double)(2 * nnz);
}*/

template<typename T>
void print_tile_t(T *input, int m, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int local_id = 0; local_id < m; local_id++)
        {
            std::cout << input[local_id * n + i] << ", ";
        }
        std::cout << std::endl;
    }
}

template<typename T>
void print_tile(T *input, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int local_id = 0; local_id < n; local_id++)
        {
            std::cout << input[i * n + local_id] << ", ";
        }
        std::cout << std::endl;
    }
}

template<typename T>
void print_1darray(T *input, int l)
{
    for (int i = 0; i < l; i++)
        std::cout << input[i] << ", ";
    std::cout << std::endl;
}

#endif // UTILS_H
