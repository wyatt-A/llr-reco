#ifndef HOST_H
#define HOST_H

#include <cuda_runtime.h>
#include <cuComplex.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t diag_mm_batched_exec(int m, int n, int batch_size, cuComplex* d_matrix, float* d_diag)

#ifdef __cplusplus
}
#endif

#endif // HOST_H