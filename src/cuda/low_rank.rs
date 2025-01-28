use crate::cuda::bindings::cuComplex;
use crate::cuda::cublas::{cublas_cgemm_strided_batched_device, rhs_cdgmm_batched_device};
use crate::cuda::cuda_api::{copy_to_device, copy_to_host, cuda_free, cuda_malloc_memset};
use crate::cuda::cusolver::cu_svd_device;
use cfl::ndarray::parallel::prelude::{IntoParallelRefIterator, ParallelIterator};
use cfl::ndarray::{Array2, ShapeBuilder};
use cfl::num_complex::Complex32;
use std::time::Instant;

pub fn cu_low_rank_approx_batch(m: usize, n: usize, rank: usize, batch_size: usize, matrix_data: &mut [Complex32]) {
    //assert_eq!(m * n * batch_size, matrix_data.len(), "incorrect number of elements for matrix data");
    assert!(matrix_data.len() >= m * n * batch_size, "matrix data buffer is too small");
    let matrix_data = &mut matrix_data[0..(m * n * batch_size)];

    let now = Instant::now();
    // copy matrix data to gpu
    let d_a = copy_to_device(matrix_data);
    let dur = now.elapsed();
    //println!("data copy took {} ms", dur.as_millis());

    let now = Instant::now();
    // allocate temp device arrays for svd calculation
    let d_u = cuda_malloc_memset::<Complex32>(m * rank * batch_size);
    let d_s = cuda_malloc_memset::<f32>(rank * batch_size);
    let d_v = cuda_malloc_memset::<Complex32>(rank * n * batch_size);
    let dur = now.elapsed();
    //println!("temp alloc took {} ms", dur.as_millis());

    let now = Instant::now();
    // perform svd
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
    let dur = now.elapsed();
    //println!("svd took {} ms", dur.as_millis());

    let now = Instant::now();
    // reconstruct U * S
    rhs_cdgmm_batched_device(m, rank, batch_size, d_u as *mut cuComplex, d_s as *mut f32)
        .expect("batched diag matrix multiply failed");
    let dur = now.elapsed();
    //println!("diag mul took {} ms", dur.as_millis());

    let alpha = Complex32::ONE;
    let beta = Complex32::ZERO;
    let now = Instant::now();
    // reconstruct (U * S) * V^H, and write the result back into the original matrix data array on device
    cublas_cgemm_strided_batched_device(m, rank, n, batch_size, alpha, beta, false, true, d_u as *mut cuComplex, d_v as *mut cuComplex, d_a as *mut cuComplex)
        .expect("cublas matrix multiply failed");
    let dur = now.elapsed();
    //println!("mat mul took {} ms", dur.as_millis());

    let now = Instant::now();
    copy_to_host(matrix_data, d_a);
    let dur = now.elapsed();
    //println!("data copy to host took {} ms", dur.as_millis());

    cuda_free(d_a);
    cuda_free(d_u);
    cuda_free(d_s);
    cuda_free(d_v);
}

#[cfg(all(test, feature = "cuda"))]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::cusolver::cu_svd;
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

        cublas_cgemm_strided_batched_device(
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
        ).expect("matrix multiply failed");

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
