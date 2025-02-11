use crate::cuda::bindings::cuComplex;
use crate::cuda::cublas::cublas_cgemm_strided_batched_device;
use crate::cuda::cuda_api::{copy_to_device, copy_to_host, cuda_free, cuda_malloc_memset};
use cfl::ndarray::{Array1, Array2, ShapeBuilder};
use cfl::num_complex::Complex32;
use cfl::num_traits::Zero;
use rayon::prelude::*;
use std::f32::consts::PI;
use std::f32::EPSILON;
use std::ffi::c_void;
use std::io::Read;

pub fn eigenvector_matrix(principal_dir: &[f32; 3], eig_matrix: &mut Array2<f32>) {
    assert_eq!(principal_dir.len(), 3, "Input must be a 3D vector");

    let principal_dir = Array1::from_iter(principal_dir.iter().cloned());


    // Ensure the vector is a unit vector
    let norm = principal_dir.dot(&principal_dir).sqrt();
    let principal_dir = principal_dir * (1. / norm);

    let e1 = principal_dir.clone();

    // Find an arbitrary vector not parallel to e1
    let mut temp = Array1::from_vec(vec![1.0, 0.0, 0.0]);
    if (e1[0] - 1.0).abs() < EPSILON {
        temp = Array1::from_vec(vec![0.0, 1.0, 0.0]);
    }

    // Compute e2 as a vector perpendicular to e1
    let e2 = e1.cross(&temp);
    let e2_norm = e2.dot(&e2).sqrt();
    let e2 = &e2 / e2_norm;

    // Compute e3 as the cross product of e1 and e2 (to ensure orthogonality)
    let e3 = e1.cross(&e2);

    assert_eq!(eig_matrix.shape(), &[3, 3], "expected a 3x3 matrix");
    // Construct the 3x3 matrix
    eig_matrix.column_mut(0).assign(&e1);
    eig_matrix.column_mut(1).assign(&e2);
    eig_matrix.column_mut(2).assign(&e3);
}

/// Computes the cross product of two 3D vectors.
trait CrossProduct {
    fn cross(&self, other: &Array1<f32>) -> Array1<f32>;
}

impl CrossProduct for Array1<f32> {
    fn cross(&self, other: &Array1<f32>) -> Array1<f32> {
        assert_eq!(self.len(), 3);
        assert_eq!(other.len(), 3);
        Array1::from_vec(vec![
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        ])
    }
}

/// sample the unit sphere with 2 random variables u1 and u2 assumed to be uniformly
/// distributed on the interval [0,1). The supplied cartesian vector is modified in-place
fn sample_unit_sphere(u1: f32, u2: f32, v: &mut [f32; 3]) {
    let theta = 2.0 * PI * u1;
    let phi = (2.0 * u2 - 1.0).acos();
    let sin_phi = phi.sin();
    let sin_theta = theta.sin();
    let cos_phi = phi.cos();
    let cos_theta = theta.cos();

    v[0] = sin_phi * cos_theta;
    v[1] = sin_phi * sin_theta;
    v[2] = cos_phi;
}

fn sample_range(u1: f32, lower_bound: f32, upper_bound: f32) -> f32 {
    assert!(upper_bound >= lower_bound);
    (upper_bound - lower_bound) * u1 + lower_bound
}


/// subspace data is assumed to be a single col-major (nq x srank) matrix where nq is n number of q-space samples and srank is the subspace rank.
/// image_matrices are assumed to be in col-major (nvox x nq) where nvox is image space, and nq is n number of q-space samples.
/// This means that both image_matrices and subspace matrix will be transposed prior to matrix multiplication and is handled
/// internally in this function. X_hat = S^T * X^H, where S is the subspace, X is the image data.
/// batch_size is the number of image_matrices to process.
pub fn subspace_constraint(subspace_data: &[Complex32], image_matrices: &mut [Complex32], batch_size: usize, nq: usize, nvox: usize, srank: usize) {
    assert!(image_matrices.len() >= nq * nvox * batch_size, "image matrices doesn't have enough entries");
    assert_eq!(subspace_data.len(), nq * srank, "unexpected subspace matrix size");


    let result_size = nvox * srank * batch_size;

    let d_a = copy_to_device(subspace_data) as *const cuComplex;
    let d_b = copy_to_device(image_matrices) as *const cuComplex;
    let d_c = cuda_malloc_memset::<Complex32>(result_size) as *mut cuComplex;

    let alpha = Complex32::ONE;
    let beta = Complex32::ZERO;
    let m = srank;
    let k = nq;
    let n = nvox;
    cublas_cgemm_strided_batched_device(m, k, n, batch_size, true, alpha, beta, true, true, d_a, d_b, d_c)
        .expect("matrix multiply failed!");

    let k = srank;
    let m = nq;

    cuda_free(d_b as *mut c_void);
    let d_b = cuda_malloc_memset::<Complex32>(image_matrices.len()) as *mut cuComplex;

    cublas_cgemm_strided_batched_device(m, k, n, batch_size, true, alpha, beta, false, false, d_a, d_c, d_b)
        .expect("matrix multiply failed!");

    // this is the un-transposed result image matrices
    let mut tmp_result = vec![Complex32::ZERO; image_matrices.len()];

    copy_to_host(&mut tmp_result, d_b as *mut c_void);

    cuda_free(d_a as *mut c_void);
    cuda_free(d_b as *mut c_void);
    cuda_free(d_c as *mut c_void);

    // // transpose tmp result to write back into image_matrices
    image_matrices.par_chunks_exact_mut(nq * nvox).zip(tmp_result.par_chunks_exact(nq * nvox)).for_each(|(a, b)| {
        for j in 0..nvox {
            for i in 0..nq {
                a[i * nvox + j] = b[j * nq + i];
            }
        }
    });
}


/// project a batch of image-space matrices into a lower-dimensional space defined by the subspace matrix.
/// /// subspace_matrix is assumed to be a single col-major (nq x srank) matrix where nq is n number of q-space samples and srank is the subspace rank.
/// image_matrices are assumed to be in col-major (nvox x nq) where nvox is image space, and nq is n number of q-space samples.
/// compressed_matrices are assumed to be col-major (srank x nvox) where nvox is image space, and nq is n number of q-space samples.
pub fn dti_project(subspace_matrix: &[Complex32], image_matrices: &[Complex32], compressed_matrices: &mut [Complex32], batch_size: usize, nq: usize, nvox: usize, srank: usize) {
    assert!(image_matrices.len() >= nq * nvox * batch_size, "image matrices doesn't have enough entries");
    assert_eq!(subspace_matrix.len(), nq * srank, "unexpected subspace matrix size");

    let result_size = nvox * srank * batch_size;

    // because we are using copy_to_host, we need this to be the exact size to avoid potential seg faults
    assert_eq!(compressed_matrices.len(), result_size, "compressed result doesn't have enough entries");

    let d_a = copy_to_device(subspace_matrix) as *const cuComplex;
    let d_b = copy_to_device(image_matrices) as *const cuComplex;
    let d_c = cuda_malloc_memset::<Complex32>(result_size) as *mut cuComplex;

    let alpha = Complex32::ONE;
    let beta = Complex32::ZERO;
    let m = srank;
    let k = nq;
    let n = nvox;
    cublas_cgemm_strided_batched_device(m, k, n, batch_size, true, alpha, beta, true, true, d_a, d_b, d_c)
        .expect("matrix multiply failed!");

    copy_to_host(compressed_matrices, d_c as *mut c_void);
    cuda_free(d_a as *mut c_void);
    cuda_free(d_b as *mut c_void);
    cuda_free(d_c as *mut c_void);
}

/// back-project a batch of lower-dimensional matrices into a higher-dimensional space defined by the subspace matrix.
/// subspace_matrix is assumed to be a single col-major (nq x srank) matrix where nq is n number of q-space samples and srank is the subspace rank.
/// image_matrices are assumed to be in col-major (nvox x nq) where nvox is image space, and nq is n number of q-space samples.
/// compressed_matrices are assumed to be col-major (srank x nvox) where nvox is image space, and nq is n number of q-space samples.
pub fn dti_back_project(subspace_matrix: &[Complex32], image_matrices: &mut [Complex32], compressed_matrices: &[Complex32], batch_size: usize, nq: usize, nvox: usize, srank: usize) {

    // we are using copy_to_host so we need this size to be exact
    assert_eq!(image_matrices.len(), nq * nvox * batch_size, "image matrices doesn't have enough entries");
    assert_eq!(subspace_matrix.len(), nq * srank, "unexpected subspace matrix size");

    let compressed_mat_size = nvox * srank * batch_size;

    // because we are using copy_to_host, we need this to be the exact size to avoid potential seg faults
    assert_eq!(compressed_matrices.len(), compressed_mat_size, "compressed result doesn't have enough entries");

    let d_a = copy_to_device(subspace_matrix) as *const cuComplex;
    let d_c = copy_to_device(compressed_matrices) as *const cuComplex;
    let d_b = cuda_malloc_memset::<Complex32>(image_matrices.len()) as *mut cuComplex;

    let alpha = Complex32::ONE;
    let beta = Complex32::ZERO;

    let k = srank;
    let m = nq;
    let n = nvox;

    cublas_cgemm_strided_batched_device(m, k, n, batch_size, true, alpha, beta, false, false, d_a, d_c, d_b)
        .expect("matrix multiply failed!");

    // this is the un-transposed result image matrices
    //let mut tmp_result = vec![Complex32::ZERO; image_matrices.len()];

    copy_to_host(image_matrices, d_b as *mut c_void);

    cuda_free(d_a as *mut c_void);
    cuda_free(d_b as *mut c_void);
    cuda_free(d_c as *mut c_void);

    conj_transpose_matrix(image_matrices, nq, nvox, batch_size);
}


/// transpose a batch of matrices from an (m x n) to an (n x m)
pub fn conj_transpose_matrix(matrix_data: &mut [Complex32], m: usize, n: usize, batch_size: usize) {
    assert_eq!(matrix_data.len(), m * n * batch_size, "matrix data doesn't have enough entries");
    let tmp = matrix_data.to_vec();
    matrix_data.par_chunks_exact_mut(m * n).zip(tmp.par_chunks_exact(m * n)).for_each(|(a, b)| {
        for j in 0..n {
            for i in 0..m {
                a[i * n + j] = b[j * m + i].conj();
            }
        }
    });
}


mod tests {
    use crate::dti_subspace::{conj_transpose_matrix, eigenvector_matrix, subspace_constraint};
    use cfl::ndarray::{Array2, Array3, ShapeBuilder};
    use cfl::num_complex::Complex32;
    use std::ops::IndexMut;

    #[test]
    fn calc_eigenvector_matrix() {
        let dir = [1.0, 0., 0.];
        let mut mat = Array2::<f32>::zeros((3, 3));
        eigenvector_matrix(&dir, &mut mat);
        println!("{:?}", mat);
    }

    #[test]
    fn test_subspace_constraint() {
        let nq = 3;
        let rank = 3;
        let nvox = 1000;

        let mut subspace = Array2::<Complex32>::zeros((nq, rank).f());
        // trivial subspace matrix (identity)
        *subspace.index_mut([0, 0]) = Complex32::ONE;
        *subspace.index_mut([1, 1]) = Complex32::ONE;
        *subspace.index_mut([2, 2]) = Complex32::ONE;

        // test image data
        let batch_size = 10;
        let mut x = Array3::<Complex32>::from_shape_fn((nvox, nq, batch_size).f(), |(i, j, _)| Complex32::ONE * (i as f32 + j as f32));
        let x_orig = x.clone();

        let subspace_data = subspace.as_slice_memory_order().unwrap();
        let x_data = x.as_slice_memory_order_mut().unwrap();

        subspace_constraint(subspace_data, x_data, batch_size, nq, nvox, rank);

        assert_eq!(x_orig, x, "x should be mapped back to itself with an identity subspace");
    }

    #[test]
    fn test_conj_transpose() {
        let m: usize = 3;
        let n: usize = 2;

        let matrix_entries = (0..(m * n)).map(|x| Complex32::new(x as f32, 0.)).collect::<Vec<Complex32>>();

        let mut mat = Array2::<Complex32>::from_shape_vec((m, n).f(), matrix_entries).unwrap();

        println!("{:?}", mat);

        let data = mat.as_slice_memory_order_mut().unwrap();

        conj_transpose_matrix(data, m, n, 1);

        let mat = Array2::<Complex32>::from_shape_vec((n, m).f(), data.to_vec()).unwrap();

        println!("{:?}", mat);
    }
}