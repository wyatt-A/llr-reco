use crate::bindings_cuda::{cuComplex, cudaError_t};
use std::ffi::c_int;

extern "C" {
    /*
        CUDA kernel to perform batched right-handed diagonal matrix multiply in-place (A = A * D) where A is complex-valued and
        D is real-valued
        The data layout for A is column-major (m,n,batch_size)
        The data layout for D is also column-major (n,batch_size)
        matrices are assumed to already be on device. No memory allocations on host or device will
        occur
    */
    fn diag_mm_batched_exec(m: c_int, n: c_int, batch_size: c_int, d_matrix: *mut cuComplex, d_diag: *mut f32) -> cudaError_t;
    //fn replace_nan_with_zero_exec(n: c_longlong, data: *mut cuComplex) -> cudaError_t;
}


pub fn diag_mm_batched(m: usize, n: usize, batch_size: usize, d_matrix: *mut cuComplex, d_diag: *mut f32) {
    let stat = unsafe {
        diag_mm_batched_exec(m as c_int, n as c_int, batch_size as c_int, d_matrix, d_diag)
    };
    if stat != 0 {
        panic!("diag_mm_batched_exec returned an error: {}", stat);
    }
}

// pub fn replace_nan_with_zero(n: usize, d_data: *mut cuComplex) {
//     let stat = unsafe {
//         replace_nan_with_zero_exec(n as c_longlong, d_data)
//     };
//     if stat != 0 {
//         panic!("diag_mm_batched_exec returned an error: {}", stat);
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_api::{copy_to_device, copy_to_host, cuda_free};
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
        let status = unsafe {
            diag_mm_batched_exec(m as c_int, n as c_int, batch_size as c_int, d_matrix as *mut cuComplex, d_diag as *mut f32)
        };
        let dur = now.elapsed().as_millis();

        println!("retrieving data from gpu ...");
        // copy data back and free up gpu memory
        copy_to_host(matrix.as_slice_memory_order_mut().unwrap(), d_matrix);
        cuda_free(d_matrix);
        cuda_free(d_diag);

        // report errors, assert correctness, report run time
        println!("status: {:?}", status);
        assert_eq!(matrix, expected_result, "unexpected result");
        println!("kernel exec took {dur} ms");
    }
}