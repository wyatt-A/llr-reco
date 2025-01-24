#ifndef KERNELS_H
#define KERNELS_H

#include <cuComplex.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void diag_mm_batched(int m, int n, int batch_size, cuComplex* d_matrix, float* d_diag);

#ifdef __cplusplus
}
#endif

#endif // KERNELS_H