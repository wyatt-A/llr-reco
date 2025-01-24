use crate::bindings_cuda::{cuComplex, cudaMalloc, cusolverDnCgesvdaStridedBatched, cusolverDnCgesvdaStridedBatched_bufferSize, cusolverDnCreate, cusolverDnDestroy, cusolverDnHandle_t, cusolverEigMode_t_CUSOLVER_EIG_MODE_VECTOR};
use crate::cuda_api::{copy_to_device, copy_to_host, cuda_free};
use cfl::num_complex::Complex32;
use std::ffi::{c_float, c_int, c_longlong, c_void};
use std::ptr::null_mut;
use std::time::Instant;

pub fn cu_svd(m: usize, n: usize, rank: usize, batch_size: usize, matrix: &[Complex32], u: &mut [Complex32], s: &mut [f32], vt: &mut [Complex32]) {
    u.fill(Complex32::ZERO);
    s.fill(0.0);
    vt.fill(Complex32::ZERO);

    let mut err_info = vec![0 as c_int; batch_size];

    let d_a = copy_to_device(matrix);
    let d_u = copy_to_device(u);
    let d_s = copy_to_device(s);
    let d_v = copy_to_device(vt);
    let d_error_info = copy_to_device(&err_info);

    let rank = rank as c_int;

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

        let stat = cudaMalloc(&mut d_work as *mut *mut c_void, size_of::<Complex32>() * lwork as usize);
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

    //copy data back to host
    copy_to_host(u, d_u);
    copy_to_host(s, d_s);
    copy_to_host(vt, d_v);
    copy_to_host(&mut err_info, d_error_info);

    cuda_free(d_a);
    cuda_free(d_u);
    cuda_free(d_s);
    cuda_free(d_v);
    cuda_free(d_work);
    cuda_free(d_error_info);

    let stat = unsafe { cusolverDnDestroy(cusolver_h) };
    if stat != 0 {
        panic!("cusolver DN destroy error");
    }

    //CUDA_CHECK(cudaStreamDestroy(stream));

    // let stat = unsafe { cudaDeviceReset() };
    // if stat != 0 {
    //     panic!("cudaDeviceReset error!");
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfl::ndarray::{Array1, Array2, Array3, ShapeBuilder};
    use cfl::ndarray_linalg::{JobSvd, SVDDC};

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_svd() {
        let rank = 10;
        let a = gen_test_matrix();
        let (m, n): (usize, usize) = a.dim().into();

        let mut s = Array1::<f32>::zeros(rank.f());
        let mut u = Array2::from_elem((m, rank).f(), Complex32::ZERO);
        let mut vt = Array2::from_elem((rank, m).f(), Complex32::ZERO);

        cu_svd(
            m,
            n,
            rank,
            1,
            a.as_slice_memory_order().unwrap(),
            u.as_slice_memory_order_mut().unwrap(),
            s.as_slice_memory_order_mut().unwrap(),
            vt.as_slice_memory_order_mut().unwrap(),
        );

        println!("cuda singular values: {:?}", s);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_svd_batch() {
        let rank = 6;
        let a = gen_test_matrix_batch();
        let (m, n, batch_size): (usize, usize, usize) = a.dim().into();

        let mut s = Array2::<f32>::zeros((rank, batch_size).f());
        let mut u = Array3::from_elem((m, rank, batch_size).f(), Complex32::ZERO);
        let mut vt = Array3::from_elem((rank, m, batch_size).f(), Complex32::ZERO);

        let now = Instant::now();
        for i in 0..13 { // loop through multiple batches to simulate very large data sets (~100 GB)
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
    fn test_lapack_svd() {
        let rank = 10;
        let a = gen_test_matrix();
        let now = Instant::now();
        let (u, s, vt) = a.svddc(JobSvd::Some).unwrap();
        let dur = now.elapsed().as_millis();
        println!("lapack singular values: {:?}", &s.as_slice_memory_order().unwrap()[0..rank]);
        println!("lapack svd took {} ms", dur);
    }

    fn gen_test_matrix() -> Array2<Complex32> {
        let m = 1000;
        let n = 67;
        Array2::<Complex32>::from_shape_fn((m, n).f(), |(i, j)| Complex32::new(i as f32 + j as f32, 0.))
    }

    fn gen_test_matrix_batch() -> Array3<Complex32> {
        let batch_size = 550;
        let m = 27000;
        let n = 67;
        println!("generating {} MB of data...", batch_size * m * n * size_of::<Complex32>() / 2usize.pow(20));
        Array3::<Complex32>::from_shape_fn((m, n, batch_size).f(), |(i, j, k)| Complex32::new(i as f32 + j as f32 + k as f32, 0.))
    }
}