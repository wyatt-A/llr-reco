use crate::cuda::bindings::{cuComplex, cublasCgemmStridedBatched, cublasCreate_v2, cublasHandle_t, cublasOperation_t_CUBLAS_OP_C, cublasOperation_t_CUBLAS_OP_N, cublasStatus_t, cublasStatus_t_CUBLAS_STATUS_SUCCESS, cudaError_t};
use cfl::num_complex::Complex32;
use std::ffi::{c_int, c_longlong};
use std::ptr::null_mut;

/// Performs a general complex-valued matrix multiply of the form: C = alpha(A * B) + beta*C
///
/// Sizes are (m x n) = (m x k) * (k x n)
///
/// Matrices A, B and C are assumed to be already in device memory and in column-major layout.
/// The batch dimension is assumed to be last. Internally, this calls cublasCgemmStridedBatched
pub fn cublas_cgemm_strided_batched_device(
    m: usize,
    k: usize,
    n: usize,
    batch_size: usize, // number of matrices to process
    broadcast_a: bool, // if we only have a single matrix b to multiply. This sets the b-stride to 0
    alpha: Complex32,
    beta: Complex32,
    transpose_a: bool,
    transpose_b: bool,
    d_a: *const cuComplex,
    d_b: *const cuComplex,
    d_c: *mut cuComplex,
) -> Result<(), cublasStatus_t> {
    let mut cublas_handle: cublasHandle_t = null_mut();

    let status = unsafe { cublasCreate_v2(&mut cublas_handle) };

    if status != cublasStatus_t_CUBLAS_STATUS_SUCCESS {
        println!("cublasCreate_v2 failed with code: {}", status);
        return Err(status);
    }

    let m = m as c_int;
    let k = k as c_int;
    let n = n as c_int;

    let (trans_a, lda) = if transpose_a {
        (
            cublasOperation_t_CUBLAS_OP_C, // A^H
            k // this has to be the physical layout before transpose
        )
    } else {
        (
            cublasOperation_t_CUBLAS_OP_N,
            m // this has to be the physical layout before transpose
        )
    };

    let (trans_b, ldb) = if transpose_b {
        (
            cublasOperation_t_CUBLAS_OP_C, // A^H
            n // this has to be the physical layout before transpose
        )
    } else {
        (
            cublasOperation_t_CUBLAS_OP_N,
            k // this has to be the physical layout before transpose
        )
    };

    let stride_a = if broadcast_a {
        0 as c_longlong // if b is broadcast, we don't want to increment with batch index
    } else {
        m as c_longlong * k as c_longlong
    };

    let stride_b = k as c_longlong * n as c_longlong;
    
    let ldc = m;
    let stride_c = m as c_longlong * n as c_longlong;
    let alpha_ptr = &alpha as *const Complex32 as *const cuComplex;
    let beta_ptr = &beta as *const Complex32 as *const cuComplex;
    let batch_size = batch_size as c_int;

    let status = unsafe {
        cublasCgemmStridedBatched(
            cublas_handle,
            trans_a,
            trans_b,
            m,
            n,
            k,
            alpha_ptr,
            d_a,
            lda,
            stride_a,
            d_b,
            ldb,
            stride_b,
            beta_ptr,
            d_c,
            ldc,
            stride_c,
            batch_size,
        )
    };

    if status != cublasStatus_t_CUBLAS_STATUS_SUCCESS {
        println!("cublasCgemmStridedBatched failed with code: {}", status);
        return Err(status);
    }
    Ok(())
}

extern "C" {
    /// Calculates the batched right-handed diagonal matrix-matrix multiplication A = A * D where A is complex-valued
    /// and D is real-valued f32 diagonal matrix. This function assumes A has a column-major layout in device memory, and D
    /// is stored as a flat list of diagonal entries. The batch dimension is last and varies the slowest. This is the raw
    /// unsafe c function.
    fn rhsCdgmmBatched(m: c_int, n: c_int, batch_size: c_int, d_matrix: *mut cuComplex, d_diag: *mut f32) -> cudaError_t;
}

/// Calculates the batched right-handed diagonal matrix-matrix multiplication A = A * D where A is complex-valued
/// and D is real-valued f32 diagonal matrix. This function assumes A has a column-major layout in device memory, and D
/// is stored as a flat list of diagonal entries. The batch dimension is last and varies the slowest. This is a non-standard
/// implementation
pub fn rhs_cdgmm_batched_device(m: usize, n: usize, batch_size: usize, d_matrix: *mut cuComplex, d_diag: *mut f32) -> Result<(), cudaError_t> {
    let stat = unsafe {
        rhsCdgmmBatched(m as c_int, n as c_int, batch_size as c_int, d_matrix, d_diag)
    };
    if stat != 0 {
        return Err(stat);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::cuda_api::{copy_to_device, copy_to_host, cuda_free};
    use cfl::ndarray::{Array2, Array3, ShapeBuilder};
    use cfl::num_complex::Complex32;
    use std::time::Instant;

    #[cfg(feature = "cuda")]
    #[test]
    fn test_diag_mm_batched_exec() {
        let m = 27000; // (30 x 30 x 30 block)
        let n = 67; // (10 singular values - rank 10)
        let batch_size = 500; // number matrices to process

        println!("building test arrays ...");
        println!("m = {m}, n = {n}, batch_size = {batch_size}");
        println!("data size: {} MB", m * n * batch_size * size_of::<Complex32>() / 2usize.pow(20));
        // input complex matrix contains real = imaginary = 1.0
        // .f() enforces column-major layout (i.e. fortran)
        let mut matrix = Array3::<Complex32>::from_elem((m, n, batch_size).f(), Complex32::new(1.0, 1.0));

        // diag matrix is real-valued and increments by 1 per column
        /*
            1 0 0 0 ...
            0 2 0 0
            0 0 3 0
            0 0 0 4
         */
        let diag = Array2::<f32>::from_shape_fn((n, batch_size).f(), |(i, _)| (i + 1) as f32);

        // expected result such that each column is multiplied by its column index + 1
        let expected_result = Array3::<Complex32>::from_shape_fn((m, n, batch_size).f(), |(_, j, _)| Complex32::new((j + 1) as f32, (j + 1) as f32));

        println!("transferring data to gpu ...");
        // copy data to gpu ...
        let d_matrix = copy_to_device(matrix.as_slice_memory_order().unwrap());
        let d_diag = copy_to_device(diag.as_slice_memory_order().unwrap());

        println!("running cuda kernel ...");
        // invoke gpu kernel
        let now = Instant::now();
        rhs_cdgmm_batched_device(m, n, batch_size, d_matrix as *mut cuComplex, d_diag as *mut f32)
            .expect("diag matrix multiplication failed");
        let dur = now.elapsed().as_millis();

        println!("retrieving data from gpu ...");
        // copy data back and free up gpu memory
        copy_to_host(matrix.as_slice_memory_order_mut().unwrap(), d_matrix);
        cuda_free(d_matrix);
        cuda_free(d_diag);

        assert_eq!(matrix, expected_result, "unexpected result");
        println!("kernel exec took {dur} ms");
    }
}