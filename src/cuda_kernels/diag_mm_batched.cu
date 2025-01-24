#include <cuComplex.h>
#include "kernels.h"

// Simple CUDA kernel to compute a right-handed in-place diagonal matrix multiply (A = A * D) in a batched configuration.
// The memory layout of A is assumed to be column-major with the batch dimension last. The assumed layout of D is
__global__ void diag_mm_batched(int m, int n, int batch_size, cuComplex* d_matrix, float* d_diag) {

    // query the thread indices
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;

    // guard against index overrun from kernel launch config
    if (batch_idx < batch_size && row_idx < m && col_idx < n) {

        // number of entries from the start of one matrix to the next
        int matrix_stride = m * n;

        // assuming a right-hand diag multiplication
        int diag_stride = n;

        // address to the start of this batch
        int matrix_batch_offset = batch_idx * matrix_stride;
        int diag_batch_offset = batch_idx * diag_stride;

        // compute the index of the matrix entry assuming col-major layout
        int matrix_elem_idx = matrix_batch_offset + col_idx * m + row_idx;
        // compute the index of the diag entry
        int diag_elem_idx = diag_batch_offset + col_idx;

        float diag_entry = d_diag[diag_elem_idx];
        cuComplex matrix_entry = d_matrix[matrix_elem_idx];
        cuComplex result = make_cuComplex(matrix_entry.x * diag_entry, matrix_entry.y * diag_entry);

        d_matrix[matrix_elem_idx] = result;

    }
}