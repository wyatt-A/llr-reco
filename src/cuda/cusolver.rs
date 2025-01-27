use crate::cuda::bindings::{cuComplex, cudaMalloc, cusolverDnCgesvdaStridedBatched, cusolverDnCgesvdaStridedBatched_bufferSize, cusolverDnCreate, cusolverDnDestroy, cusolverDnHandle_t, cusolverEigMode_t_CUSOLVER_EIG_MODE_VECTOR};
use crate::cuda::cuda_api::{copy_to_device, copy_to_host, cuda_free};
use cfl::ndarray::parallel::prelude::{IntoParallelRefIterator, ParallelIterator};
use cfl::num_complex::Complex32;
use std::ffi::{c_float, c_int, c_longlong, c_void};
use std::ptr::null_mut;

/// executes the cusolver approximate svd (gesvda) on a complex batch of (m x n x batch_size) matrices.
/// Memory layout is assumed to be column-major.
/// The decomposition performed is
///
/// A = U * S * V
///
/// with only the first rank r singular values and right/left singular vectors calculated:
///
/// (m x n) = (m x rank) * (rank x rank) * (n x rank)^H
///
/// U has size (m x r x batch_size), S is diagonal and has size (r x batch_size), V has size (n x r x batch_size)
/// all arrays are assumed to be already on-device and have a column-major layout
pub fn cu_svd_device(
    m: usize,
    n: usize,
    rank: usize,
    batch_size: usize,
    d_a: *mut cuComplex,
    d_u: *mut cuComplex,
    d_s: *mut f32,
    d_v: *mut cuComplex,
) {
    let mut err_info = vec![0 as c_int; batch_size];
    let d_error_info = copy_to_device(&err_info);


    let mut cusolver_h: cusolverDnHandle_t = null_mut();
    //println!("cusolver_h before init: {:?}", cusolver_h);
    //let now = Instant::now();
    unsafe {
        let stat = cusolverDnCreate(&mut cusolver_h);
        if stat != 0 {
            panic!("cusolver DN create error");
        }
    }
    //let dur = now.elapsed().as_millis();
    //println!("cusolver_h after init: {:?}", cusolver_h);
    //println!("took {} ms to init", dur);

    let n_u = m * rank * batch_size;
    let n_v = n * rank * batch_size;

    let rank = rank as c_int;
    let job_z = cusolverEigMode_t_CUSOLVER_EIG_MODE_VECTOR;
    let m = m as c_int;
    let n = n as c_int;
    let lda = m;
    let ldu = m;
    let ldv = n;
    let stride_a = (lda * n) as c_longlong;
    let stride_u = (ldu * rank) as c_longlong;
    let stride_s = rank as c_longlong;
    let stride_v = (ldv * rank) as c_longlong;
    let batch_size = batch_size as c_int;
    let mut lwork = 0 as c_int;

    let mut d_work = null_mut() as *mut c_void;
    //let d_work_ptr = &mut d_work as *mut *mut c_void;

    unsafe {
        let stat = cusolverDnCgesvdaStridedBatched_bufferSize(
            cusolver_h,
            job_z, /* CUSOLVER_EIG_MODE_VECTOR */
            /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
            rank, /* number of singular values */
            m, /* number of rows of Aj, 0 <= m */
            n, /* number of columns of Aj, 0 <= n  */
            d_a as *const cuComplex, /* Aj is m-by-n */
            lda, /* leading dimension of Aj */
            stride_a, /* >= lda*n */
            d_s as *mut c_float, /* Sj is rank-by-1, singular values in descending order */
            stride_s, /* >= rank */
            d_u as *mut cuComplex, /* Uj is m-by-rank */
            ldu, /* leading dimension of Uj, ldu >= max(1,m) */
            stride_u, /* >= ldu*rank */
            d_v as *mut cuComplex, /* Vj is n-by-rank */
            ldv, /* leading dimension of Vj, ldv >= max(1,n) */
            stride_v, /* >= ldv*rank */
            &mut lwork as *mut c_int,
            batch_size, /* number of matrices */
        );
        if stat != 0 {
            panic!("cusolverDnCgesvdaStridedBatched_bufferSize error: {}", stat);
        }

        let stat = cudaMalloc(
            &mut d_work as *mut *mut c_void,
            size_of::<Complex32>() * lwork as usize,
        );
        if stat != 0 {
            panic!("cudaMalloc error");
        }

        //let now = Instant::now();
        cusolverDnCgesvdaStridedBatched(
            cusolver_h,
            job_z, /* CUSOLVER_EIG_MODE_VECTOR */
            /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
            rank, /* number of singular values */
            m, /* number of rows of Aj, 0 <= m */
            n, /* number of columns of Aj, 0 <= n  */
            d_a as *const cuComplex, /* Aj is m-by-n */
            lda, /* leading dimension of Aj */
            stride_a, /* >= lda*n */
            d_s as *mut c_float, /* Sj is rank-by-1 */
            /* the singular values in descending order */
            stride_s, /* >= rank */
            d_u as *mut cuComplex, /* Uj is m-by-rank */
            ldu, /* leading dimension of Uj, ldu >= max(1,m) */
            stride_u, /* >= ldu*rank */
            d_v as *mut cuComplex, /* Vj is n-by-rank */
            ldv, /* leading dimension of Vj, ldv >= max(1,n) */
            stride_v, /* >= ldv*rank */
            d_work as *mut cuComplex,
            lwork,
            d_error_info as *mut c_int,
            null_mut(), /* we don't care about precision at the moment */
            batch_size, /* number of matrices */
        );
        if stat != 0 {
            panic!("cusolverDnCgesvdaStridedBatched error: {}", stat);
        }
        //let dur = now.elapsed().as_millis();
        //println!("svd calc took: {} ms", dur);
    }

    copy_to_host(&mut err_info, d_error_info);
    // check for convergence errors
    let n_errors = err_info.par_iter().map(|&e| e as usize).sum::<usize>();
    if n_errors != 0 {
        panic!("svd failed for {} matrices!", n_errors);
    }

    cuda_free(d_work);
    cuda_free(d_error_info);

    let stat = unsafe { cusolverDnDestroy(cusolver_h) };
    if stat != 0 {
        panic!("cusolver DN destroy error");
    }

    //replace_nan_with_zero(n_u, d_u);
    //replace_nan_with_zero(n_v, d_v);
}

pub fn cu_svd(
    m: usize,
    n: usize,
    rank: usize,
    batch_size: usize,
    matrix: &[Complex32],
    u: &mut [Complex32],
    s: &mut [f32],
    v: &mut [Complex32],
) {
    // check for appropriate lengths of input arrays
    let matrix_len = m * n * batch_size;
    let u_len = m * rank * batch_size;
    let s_len = rank * batch_size;
    let v_len = n * rank * batch_size;

    assert_eq!(
        matrix_len,
        matrix.len(),
        "incorrect number of elements for matrix"
    );
    assert_eq!(u_len, u.len(), "incorrect number of elements for u");
    assert_eq!(s_len, s.len(), "incorrect number of elements for s");
    assert_eq!(v_len, v.len(), "incorrect number of elements for vt");

    u.fill(Complex32::ZERO);
    s.fill(0.0);
    v.fill(Complex32::ZERO);

    let d_a = copy_to_device(matrix);
    let d_u = copy_to_device(u);
    let d_s = copy_to_device(s);
    let d_v = copy_to_device(v);

    cu_svd_device(
        m,
        n,
        rank,
        batch_size,
        d_a as *mut cuComplex,
        d_u as *mut cuComplex,
        d_s as *mut f32,
        d_v as *mut cuComplex,
    );

    copy_to_host(u, d_u);
    copy_to_host(s, d_s);
    copy_to_host(v, d_v);

    cuda_free(d_a);
    cuda_free(d_u);
    cuda_free(d_s);
    cuda_free(d_v);
}