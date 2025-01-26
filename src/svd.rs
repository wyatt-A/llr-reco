use crate::bindings_cuda::{cuComplex, cublasCgemmStridedBatched, cublasCreate_v2, cublasHandle_t, cublasOperation_t_CUBLAS_OP_C, cublasOperation_t_CUBLAS_OP_N, cublasStatus_t_CUBLAS_STATUS_SUCCESS, cudaMalloc, cusolverDnCgesvdaStridedBatched, cusolverDnCgesvdaStridedBatched_bufferSize, cusolverDnCreate, cusolverDnDestroy, cusolverDnHandle_t, cusolverEigMode_t_CUSOLVER_EIG_MODE_VECTOR};
use crate::cuda_api::{copy_to_device, copy_to_host, cuda_free, cuda_malloc_memset};
use crate::kernel_bindings::diag_mm_batched;
use cfl::ndarray::parallel::prelude::{IntoParallelRefIterator, ParallelIterator};
use cfl::ndarray::{Array2, ShapeBuilder};
use cfl::num_complex::Complex32;
use std::ffi::{c_float, c_int, c_longlong, c_void};
use std::ptr::null_mut;
use std::time::Instant;

/// executes the cusolver approximate svd (gesvda) on a complex batch of (m x n x batch_size) matrices.
/// Memory layout is assumed to be column-major.
/// The decomposition performed is
///
/// A = U * S * V^H
///
/// with only the first rank r singular values and right/left singular vectors calculated:
///
/// (m x n) = (m x rank) * (rank x rank) * (rank x n)
///
/// U has size (m x r x batch_size), S is diagonal and has size (r x batch_size), V^H has size (r x n x batch_size)
/// all arrays are assumed to be already on-device and have a column-major layout
fn cu_svd_exec(
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
    let n_errors = err_info.par_iter().sum::<c_int>();
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

    cu_svd_exec(
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

/// Performs a general complex-valued matrix multiply of the form: C = alpha(A * B) + beta*C
///
/// Sizes are (m x n) = (m x k) * (k x n)
///
/// Matrices A, B and C are assumed to be already in device memory and in column-major layout.
/// The batch dimension is assumed to be last. Internally, this calls cublasCgemmStridedBatched
fn cu_mat_mult_exec(
    m: usize,
    k: usize,
    n: usize,
    batch_size: usize,
    alpha: Complex32,
    beta: Complex32,
    transpose_a: bool,
    transpose_b: bool,
    d_a: *const cuComplex,
    d_b: *const cuComplex,
    d_c: *mut cuComplex,
) {
    let mut cublas_handle: cublasHandle_t = null_mut();

    let status = unsafe { cublasCreate_v2(&mut cublas_handle) };

    if status != cublasStatus_t_CUBLAS_STATUS_SUCCESS {
        panic!("cublasCreate_v2 failed: {}", status);
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

    let stride_a = m as c_longlong * k as c_longlong;
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
        panic!("cublasCgemmStridedBatched failed: {}", status);
    }
}

fn cu_low_rank_approx_batch(m: usize, n: usize, rank: usize, batch_size: usize, matrix_data: &mut [Complex32]) {
    assert_eq!(m * n * batch_size, matrix_data.len(), "incorrect number of elements for matrix data");

    let now = Instant::now();
    // copy matrix data to gpu
    let d_a = copy_to_device(matrix_data);
    let dur = now.elapsed();
    println!("data copy took {} ms", dur.as_millis());

    let now = Instant::now();
    // allocate temp device arrays for svd calculation
    let d_u = cuda_malloc_memset::<Complex32>(m * rank * batch_size);
    let d_s = cuda_malloc_memset::<f32>(rank * batch_size);
    let d_v = cuda_malloc_memset::<Complex32>(rank * n * batch_size);
    let dur = now.elapsed();
    println!("temp alloc took {} ms", dur.as_millis());

    let now = Instant::now();
    // perform svd
    cu_svd_exec(
        m,
        n,
        rank,
        batch_size,
        d_a as *mut cuComplex,
        d_u as *mut cuComplex,
        d_s as *mut f32,
        d_v as *mut cuComplex,
    );
    let dur = now.elapsed();
    println!("svd took {} ms", dur.as_millis());

    let now = Instant::now();
    // reconstruct U * S
    diag_mm_batched(m, rank, batch_size, d_u as *mut cuComplex, d_s as *mut f32);
    let dur = now.elapsed();
    println!("diag mul took {} ms", dur.as_millis());

    let alpha = Complex32::ONE;
    let beta = Complex32::ZERO;
    let now = Instant::now();
    // reconstruct (U * S) * V^H, and write the result back into the original matrix data array on device
    cu_mat_mult_exec(m, rank, n, batch_size, alpha, beta, false, true, d_u as *mut cuComplex, d_v as *mut cuComplex, d_a as *mut cuComplex);
    let dur = now.elapsed();
    println!("mat mul took {} ms", dur.as_millis());

    let now = Instant::now();
    copy_to_host(matrix_data, d_a);
    let dur = now.elapsed();
    println!("data copy to host took {} ms", dur.as_millis());

    cuda_free(d_a);
    cuda_free(d_u);
    cuda_free(d_s);
    cuda_free(d_v);
}

#[cfg(all(test, feature = "cuda"))]
#[cfg(test)]
mod tests {
    use super::*;
    use cfl::ndarray::{s, Array1, Array2, Array3, ShapeBuilder};
    use cfl::ndarray_linalg::{JobSvd, SVDDC};
    use std::time::Instant;

    #[test]
    fn test_cuda_svd() {
        let rank = 5;
        let a = gen_test_matrix();
        let (m, n): (usize, usize) = a.dim().into();

        let mut s = Array1::<f32>::zeros(rank.f());
        let mut u = Array2::from_elem((m, rank).f(), Complex32::ZERO);
        let mut v = Array2::from_elem((n, rank).f(), Complex32::ZERO);

        cu_svd(
            m,
            n,
            rank,
            1,
            a.as_slice_memory_order().unwrap(),
            u.as_slice_memory_order_mut().unwrap(),
            s.as_slice_memory_order_mut().unwrap(),
            v.as_slice_memory_order_mut().unwrap(),
        );

        println!("cuda singular values: {:?}", s);
        println!("u: {:?}", u);
        println!("v: {:?}", v);
    }

    #[test]
    fn test_cu_mat_mult_exec() {
        let m = 4;
        let k = 3;
        let n = 4;
        let batch_size = 2;
        let alpha = Complex32::ONE;
        let beta = Complex32::ZERO;

        let a = Array3::from_shape_fn((m, k, batch_size).f(), |(i, j, _)| {
            Complex32::new(i as f32, j as f32)
        });

        let b = Array3::from_shape_fn((k, n, batch_size).f(), |(i, j, _)| {
            Complex32::new(j as f32, i as f32)
        });

        let mut c = Array3::from_elem((m, n, batch_size).f(), Complex32::ZERO);

        let d_a = copy_to_device(a.as_slice_memory_order().unwrap());
        let d_b = copy_to_device(b.as_slice_memory_order().unwrap());
        let d_c = copy_to_device(c.as_slice_memory_order().unwrap());

        cu_mat_mult_exec(
            m,
            k,
            n,
            batch_size,
            alpha,
            beta,
            false,
            false,
            d_a as *const cuComplex,
            d_b as *const cuComplex,
            d_c as *mut cuComplex,
        );

        copy_to_host(c.as_slice_memory_order_mut().unwrap(), d_c);
        cuda_free(d_a);
        cuda_free(d_b);
        cuda_free(d_c);

        /* expected result
            -5.0000 + 0.0000i  -5.0000 + 3.0000i  -5.0000 + 6.0000i  -5.0000 + 9.0000i
            -5.0000 + 3.0000i  -2.0000 + 6.0000i   1.0000 + 9.0000i   4.0000 +12.0000i
            -5.0000 + 6.0000i   1.0000 + 9.0000i   7.0000 +12.0000i  13.0000 +15.0000i
            -5.0000 + 9.0000i   4.0000 +12.0000i  13.0000 +15.0000i  22.0000 +18.0000i
        */

        let result = vec![
            Complex32::new(-5., 0.),
            Complex32::new(-5., 3.),
            Complex32::new(-5., 6.),
            Complex32::new(-5., 9.),
            Complex32::new(-5., 3.),
            Complex32::new(-2., 6.),
            Complex32::new(1., 9.),
            Complex32::new(4., 12.),
            Complex32::new(-5., 6.),
            Complex32::new(1., 9.),
            Complex32::new(7., 12.),
            Complex32::new(13., 15.),
            Complex32::new(-5., 9.),
            Complex32::new(4., 12.),
            Complex32::new(13., 15.),
            Complex32::new(22., 18.),
        ];

        let expected_result = Array2::from_shape_vec((m, n).f(), result).unwrap();
        // assert correct single-matrix result
        assert_eq!(expected_result.view(), c.slice(s![.., .., 0]));
        // assert that first and last matrix is the same (no batch idx dependence)
        assert_eq!(c.slice(s![.., .., 0]), c.slice(s![.., .., batch_size - 1]));
    }

    #[test]
    fn test_cuda_svd_batch() {
        let rank = 6;
        let a = gen_test_matrix_batch();
        let (m, n, batch_size): (usize, usize, usize) = a.dim().into();

        let mut s = Array2::<f32>::zeros((rank, batch_size).f());
        let mut u = Array3::from_elem((m, rank, batch_size).f(), Complex32::ZERO);
        let mut vt = Array3::from_elem((rank, n, batch_size).f(), Complex32::ZERO);

        let now = Instant::now();
        for i in 0..1 {
            // loop through multiple batches to simulate very large data sets (~100 GB)
            println!("iter {} ...", i + 1);
            cu_svd(
                m,
                n,
                rank,
                batch_size,
                a.as_slice_memory_order().unwrap(),
                u.as_slice_memory_order_mut().unwrap(),
                s.as_slice_memory_order_mut().unwrap(),
                vt.as_slice_memory_order_mut().unwrap(),
            );
        }
        let dur = now.elapsed().as_millis();

        let sv_start = &s.as_slice_memory_order().unwrap()[0..rank];
        let sv_end = &s.as_slice_memory_order().unwrap()[(batch_size - 1) * rank..];

        println!("cuda singular values start: {:?}", sv_start);
        println!("cuda singular values end: {:?}", sv_end);
        //assert_eq!(sv_start, sv_end, "error between starting and trailing batch singular values");
        println!("cuda svd took {} ms", dur);
    }

    #[test]
    fn test_cuda_lowrank_batch() {
        let rank = 10;
        println!("generating test host data ...");
        let mut a = gen_test_matrix_batch();
        let (m, n, batch_size): (usize, usize, usize) = a.dim().into();

        //println!("{:?}", a.slice(s![.., .., 0]));

        let outer = Instant::now();
        for i in 1..=2 {
            println!("iter {i} ...");
            println!("performing low rank approximation ...");
            let now = Instant::now();
            cu_low_rank_approx_batch(m, n, rank, batch_size, a.as_slice_memory_order_mut().unwrap());
            let dur = now.elapsed().as_millis();
            println!("cuda low-rank took: {dur} ms");
        }
        let total_dur = outer.elapsed().as_secs_f32();

        println!("{:?}", a.slice(s![.., .., 0]));

        println!("total time: {total_dur} sec");
    }

    #[test]
    fn test_lapack_svd() {
        let rank = 10;
        let a = gen_test_matrix();
        let now = Instant::now();
        let (u, s, vt) = a.svddc(JobSvd::Some).unwrap();
        let dur = now.elapsed().as_millis();
        println!(
            "lapack singular values: {:?}",
            &s.as_slice_memory_order().unwrap()[0..rank]
        );
        println!("lapack svd took {} ms", dur);
    }

    fn gen_test_matrix() -> Array2<Complex32> {
        let m = 20;
        let n = 10;
        Array2::<Complex32>::from_shape_fn((m, n).f(), |(i, j)| {
            Complex32::new(1.0, 0.)
            //Complex32::new(i as f32 + j as f32, 0.)
        })
    }

    fn gen_test_matrix_batch() -> Array3<Complex32> {
        let batch_size = 500;
        let m = 27000;
        let n = 67;
        println!(
            "generating {} MB of data...",
            batch_size * m * n * size_of::<Complex32>() / 2usize.pow(20)
        );
        Array3::<Complex32>::from_shape_fn((m, n, batch_size).f(), |(i, j, _)| {
            //Complex32::new(i as f32 + j as f32, 0.)
            Complex32::new(1., 0.)
        })
    }
}
