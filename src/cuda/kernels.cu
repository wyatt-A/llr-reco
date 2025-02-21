#include <cuComplex.h>
#include <cuda_runtime.h>
#include <stdio.h>

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

__device__ int euclidean_mod(int x, int m) {
    int r = x % m;
    return (r < 0) ? r + m : r;
}

__device__ unsigned int compute_new_idx(int z, int shift, int nz) {
    return (unsigned int)euclidean_mod(z + shift, nz);
}

__device__ unsigned int shift_address(unsigned int address, unsigned int *vol_size, int *shift) {
    unsigned int nz = vol_size[0];
    unsigned int ny = vol_size[1];
    unsigned int nx = vol_size[2];

    unsigned int z = address % nz;
    unsigned int remainder = address / nz;
    unsigned int y = remainder % ny;
    unsigned int x = remainder / ny;

    unsigned int new_z = compute_new_idx((int)z,shift[0],(int)nz);
    unsigned int new_y = compute_new_idx((int)y,shift[1],(int)nz);
    unsigned int new_x = compute_new_idx((int)x,shift[2],(int)nz);

    return new_z + nz * (new_y + ny * new_x);
}

__device__ unsigned int symmetric_boundary_index(int index, int n) {
    if (index < 0) {
        return (unsigned int)abs(index + 1);
    }else if (index >= n) {
        return (unsigned int)(2 * n - index - 1);
    }else {
        return (unsigned int)index;
    }
}

__device__ unsigned int lane_head(unsigned int lane_idx, unsigned int lane_axis, unsigned int *vol_size) {

    unsigned int n_lanes = 0;
    if (lane_axis == 0) {
        n_lanes = vol_size[1] * vol_size[2];
    }else if (lane_axis == 1) {
        n_lanes = vol_size[0] * vol_size[2];
    }else {
        n_lanes = vol_size[0] * vol_size[1];
    }

    if (lane_axis == 0) {
        return lane_idx * vol_size[0];
    }else if (lane_axis == 1) {
        unsigned int x = lane_idx % vol_size[0];
        unsigned int z = lane_idx / vol_size[0];
        return x + (z * vol_size[0]) * vol_size[1];
    }else {
        return lane_idx;
    }

}

__device__ unsigned int lane_stride(unsigned int *vol_size, unsigned int axis) {
    if (axis == 0) {
        return 1;
    } else if (axis == 1) {
        return vol_size[0];
    } else {
        return vol_size[0] * vol_size[1];
    }
}

__host__ __device__ unsigned int num_lanes(unsigned int *vol_size, unsigned int axis) {
    if (axis == 0) {
        return vol_size[1] * vol_size[2];
    } else if (axis == 1) {
        return vol_size[0] * vol_size[2];
    } else {
        return vol_size[0] * vol_size[1];
    }
}

__host__ __device__ int sub_band_size(unsigned int signal_length, unsigned int filter_length) {
    return ((int)signal_length + (int)filter_length - 1) / 2;
}

// invoked with lane index as x, and coefficient index as y
__global__ void _dwt3_axis(cuComplex *vol_data, cuComplex *decomp, unsigned int *vol_size, unsigned int *decomp_size, unsigned int axis, unsigned int filter_len, float *lo_d, float *hi_d, int *rand_shift) {

    // query thread index
    unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    int coeff = blockIdx.y * blockDim.y + threadIdx.y;

    // get the problem size
    unsigned int n_lanes = num_lanes(vol_size, axis);
    int n_coeffs = sub_band_size(vol_size[axis], filter_len);

    // guard from out-of-bounds thread invocation
    if ((coeff >= n_coeffs) || (lane >= n_lanes)) {
        return;
    }

    int f_len = (int)filter_len;
    // signal padding size for both sides of lane
    int signal_extension_len = f_len - 1;
    int sig_len = (int)vol_size[axis];

    // this is the jump to get from one signal element to the next across an axis
    unsigned int signal_stride = lane_stride(vol_size, axis);
    unsigned int decomp_stride = lane_stride(decomp_size, axis);

    unsigned int signal_lane_head = lane_head(lane, axis, vol_size);
    unsigned int result_lane_head = lane_head(lane, axis, decomp_size);

    cuComplex a = make_cuComplex(0.,0.);
    cuComplex d = make_cuComplex(0.,0.);

    for(int j=0; j<f_len; j++) {

        int virtual_idx = 2 * coeff - signal_extension_len + j + 1;

        unsigned int sample_index_lane = symmetric_boundary_index(virtual_idx,sig_len);
        int filter_idx = (f_len - j - 1);

        unsigned int sample_address_vol = sample_index_lane * signal_stride + signal_lane_head;

        sample_address_vol = shift_address(sample_address_vol, vol_size, rand_shift);

        cuComplex sample = vol_data[sample_address_vol];

        cuComplex filter_c_a = make_cuComplex(lo_d[filter_idx],0.);
        cuComplex filter_c_d = make_cuComplex(hi_d[filter_idx],0.);

        a = cuCfmaf(filter_c_a,vol_data[sample_address_vol],a);
        d = cuCfmaf(filter_c_d,vol_data[sample_address_vol],d);

    }

    unsigned int result_index_a = (unsigned int)coeff * decomp_stride + result_lane_head;
    unsigned int result_index_d = (unsigned int)(coeff + n_coeffs) * decomp_stride + result_lane_head;

    decomp[result_index_a] = a;
    decomp[result_index_d] = d;

}

// invoked with lane index as x, and sample index as y
__global__ void _idwt3_axis(cuComplex *vol_data, cuComplex *decomp, unsigned int *vol_size, unsigned int *decomp_size, unsigned int axis, unsigned int filter_len, float *lo_r, float *hi_r, int *rand_shift) {

    // query thread index
    unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    int sample = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int n_lanes = num_lanes(decomp_size, axis);
    int n_samples = (int)vol_size[axis];

    if ((lane >= n_lanes) || (sample >= n_samples)) {
        return;
    }

    int f_len = (int)filter_len;

    int n_coeffs = (int)(decomp_size[axis] / 2);

    int full_len = 2 * n_coeffs + f_len - 1;
    int keep_len = 2 * n_coeffs - f_len + 2;
    int start = (full_len - keep_len) / 2;

    unsigned int decomp_stride = lane_stride(decomp_size, axis);
    unsigned int signal_stride = lane_stride(vol_size, axis);

    unsigned int decomp_lane_head = lane_head(lane, axis, decomp_size);
    unsigned int signal_lane_head = lane_head(lane, axis, vol_size);

    int c = start + sample;
    cuComplex r = make_cuComplex(0.,0.);
    for(int j=0;j<f_len;j++){
        int idx = c - j;
        if (idx >= 0 && idx < (2 * n_coeffs)) {
            if (idx % 2 == 0) {
                unsigned int approx_idx = (unsigned int)idx/2;
                unsigned int detail_idx = (unsigned int)((idx/2) + n_coeffs);
                unsigned int approx_idx_actual = approx_idx * decomp_stride + decomp_lane_head;
                unsigned int detail_idx_actual = detail_idx * decomp_stride + decomp_lane_head;
                cuComplex filter_c_a = make_cuComplex(lo_r[j],0.);
                cuComplex filter_c_d = make_cuComplex(hi_r[j],0.);
                cuComplex tmp = cuCaddf(
                    cuCmulf(filter_c_a, decomp[approx_idx_actual]),
                    cuCmulf(filter_c_d, decomp[detail_idx_actual])
                );
                r = cuCaddf(tmp,r);
            }
        }
    }

    unsigned int sample_address_vol = (unsigned int)sample * signal_stride + signal_lane_head;
    sample_address_vol = shift_address(sample_address_vol, vol_size, rand_shift);

    vol_data[sample_address_vol] = r;

}

extern "C" {

    // host function to configure kernel launch
    cudaError_t dwt3_axis(cuComplex *vol_data, cuComplex *decomp, unsigned int *vol_size, unsigned int *decomp_size, unsigned int axis, unsigned int filter_len, float *lo_d, float *hi_d, int *rand_shift) {

        // copy auxiliary data to gpu
        unsigned int *d_vol_size;
        cudaMalloc((void **)&d_vol_size, 3 * sizeof(unsigned int));
        cudaMemcpy(d_vol_size, vol_size, 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);

        unsigned int *d_decomp_size;
        cudaMalloc((void **)&d_decomp_size, 3 * sizeof(unsigned int));
        cudaMemcpy(d_decomp_size, decomp_size, 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);

        unsigned int n_lanes = num_lanes(vol_size, axis);
        int n_coeffs = sub_band_size(vol_size[axis], filter_len);

        // 32 x 32 x 1 (1024) threads per block (max threads per block)
        unsigned int block_dim_x = 32;
        unsigned int block_dim_y = 32;
        dim3 blockDim(block_dim_x, block_dim_y, 1);

        unsigned int nx = (n_lanes + blockDim.x - 1) / blockDim.x;
        unsigned int ny = (n_coeffs + blockDim.y - 1) / blockDim.y;

        dim3 gridDim(nx, ny, 1);

        _dwt3_axis<<<gridDim,blockDim>>>(vol_data, decomp, d_vol_size, d_decomp_size, axis, filter_len, lo_d, hi_d, rand_shift);

        // ensure all device operations are complete
        cudaDeviceSynchronize();

        cudaFree(d_decomp_size);
        cudaFree(d_vol_size);

        // Check for errors in kernel launch
        cudaError_t status = cudaGetLastError();

        // return the status
        return status;
    }

    cudaError_t idwt3_axis(cuComplex *vol_data, cuComplex *decomp, unsigned int *vol_size, unsigned int *decomp_size, unsigned int axis, unsigned int filter_len, float *lo_d, float *hi_d, int *rand_shift) {

        // copy auxiliary data to gpu
        unsigned int *d_vol_size;
        cudaMalloc((void **)&d_vol_size, 3 * sizeof(unsigned int));
        cudaMemcpy(d_vol_size, vol_size, 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);

        unsigned int *d_decomp_size;
        cudaMalloc((void **)&d_decomp_size, 3 * sizeof(unsigned int));
        cudaMemcpy(d_decomp_size, decomp_size, 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);

        unsigned int n_lanes = num_lanes(decomp_size, axis);
        unsigned int n_samples = vol_size[axis];

        // 32 x 32 x 1 (1024) threads per block (max threads per block)
        unsigned int block_dim_x = 32;
        unsigned int block_dim_y = 32;
        dim3 blockDim(block_dim_x, block_dim_y, 1);

        unsigned int nx = (n_lanes + blockDim.x - 1) / blockDim.x;
        unsigned int ny = (n_samples + blockDim.y - 1) / blockDim.y;

        dim3 gridDim(nx, ny, 1);
        _idwt3_axis<<<gridDim,blockDim>>>(vol_data, decomp, d_vol_size, d_decomp_size, axis, filter_len, lo_d, hi_d, rand_shift);

        // ensure all device operations are complete
        cudaDeviceSynchronize();

        cudaFree(d_decomp_size);
        cudaFree(d_vol_size);

        // Check for errors in kernel launch
        cudaError_t status = cudaGetLastError();

        // return the status
        return status;
    }

}