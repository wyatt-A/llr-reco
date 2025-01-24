#include <cuComplex.h>
#include <cuda_runtime.h>
#include "kernels.h"

// host function to configure kernel launch
cudaError_t diag_mm_batched_exec(int m, int n, int batch_size, cuComplex* d_matrix, float* d_diag) {

    // 16 x 16 x 16 (4096) threads per block
    dim3 blockDim(16, 16, 16);
    // ceiling div to calculate total number of blocks to execute (grid size)
    dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y, (batch_size + blockDim.z - 1) / blockDim.z);

    myKernel<<<numBlocks, threadsPerBlock>>>(data, width, height, depth);
    diag_mm_batched<<<gridDim,blockDim>>>(m, n, batch_size, d_matrix, d_diag);

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();

    // ensure all device operations are complete
    cudaDeviceSynchronize();

    // return the status
    return err;
}