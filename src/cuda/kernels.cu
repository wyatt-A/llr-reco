#include <cuComplex.h>
#include <cuda_runtime.h>

/*
CUDA kernel to perform batched right-handed diagonal matrix multiply in-place (A = A * D) where A is complex-valued and
D is real-valued
The data layout for A is column-major (m,n,batch_size)
The data layout for D is also column-major (n,batch_size)
*/
__global__ void _rhsCdgmmBatched(int m, int n, int batch_size, cuComplex* d_matrix, float* d_diag) {

    // query the thread indices
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
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

extern "C" {

    // host function to configure kernel launch
    cudaError_t rhsCdgmmBatched(int m, int n, int batch_size, cuComplex* d_matrix, float* d_diag) {

        // 16 x 8 x 8 (1024) threads per block
        dim3 blockDim(16, 8, 8);

        // ceiling div to calculate total number of blocks to execute (grid size)
        dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y, (batch_size + blockDim.z - 1) / blockDim.z);

        _rhsCdgmmBatched<<<gridDim,blockDim>>>(m, n, batch_size, d_matrix, d_diag);

        // ensure all device operations are complete
        cudaDeviceSynchronize();

        // Check for errors in kernel launch
        cudaError_t status = cudaGetLastError();

        // return the status
        return status;
    }

}