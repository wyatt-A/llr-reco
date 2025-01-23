use crate::bindings_cuda::{cuComplex, cudaMalloc, cusolverDnCgesvd, cusolverDnCgesvd_bufferSize, cusolverDnCreate, cusolverDnHandle_t};
use crate::cuda_api::copy_to_device;
use cfl::num_complex::Complex32;
use std::ffi::{c_int, c_void};
use std::ptr;

pub fn cu_svd(m: usize, n: usize, matrix: &[Complex32], u: &mut [Complex32], s: &mut [f32], vt: &mut [Complex32]) {
    u.fill(Complex32::ZERO);
    s.fill(0.0);
    vt.fill(Complex32::ZERO);

    let matrix_device = copy_to_device(matrix);
    let u_device = copy_to_device(u);
    let s_device = copy_to_device(s);
    let vt_device = copy_to_device(vt);

    // set up context and solver handle
    let mut handle: cusolverDnHandle_t = ptr::null_mut();
    unsafe { cusolverDnCreate(&mut handle) };
    let mut lwork: c_int = 0;
    unsafe {
        cusolverDnCgesvd_bufferSize(handle, m as c_int, n as c_int, &mut lwork);
    }
    let d_work = unsafe {
        let mut d_ptr: *mut c_void = ptr::null_mut();
        cudaMalloc(&mut d_ptr, lwork as usize);
        d_ptr
    };

    let d_info = unsafe {
        let mut d_ptr: *mut i32 = ptr::null_mut();
        cudaMalloc(&mut d_ptr as *mut *mut c_void, size_of::<i32>());
        d_ptr
    };

    // Execute SVD
    let jobu = b'A'; // Compute all left singular vectors
    let jobvt = b'A'; // Compute all right singular vectors

    let lda = m as c_int;

    // unsafe {
    //     cusolverDnCgesvdjBatched
    // }

    // NEED TO DOUBLE-CHECK THESE ARGUMENTS WITH THE DOCS TO SEE THAT THEY ARE CORRECT
    unsafe {
        cusolverDnCgesvd(
            handle,
            jobu as i8,
            jobvt as i8,
            m as c_int,
            n as c_int,
            matrix_device as *mut cuComplex,
            lda,
            s_device as *mut f32,
            u_device as *mut cuComplex,
            m as c_int,
            vt_device as *mut cuComplex,
            n as c_int,
            d_work,
            lwork,
            ptr::null_mut(),
            d_info,
        );
    }
}