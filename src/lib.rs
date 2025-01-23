#[cfg(feature = "cuda")]
pub mod cufft;
#[cfg(feature = "cuda")]
pub mod cuda_api;
#[cfg(feature = "cuda")]
mod bindings_cuda;

use cfl;
use cfl::ndarray::{Array2, Array3, ShapeBuilder};
use cfl::ndarray_linalg::JobSvd;
use cfl::ndarray_linalg::SVDDC;
use cfl::num_complex::Complex32;
use std::ops::Range;
pub mod fftshift;
mod svd;

pub enum LlrError {
    ExtractMatrix
}

// extract 3-D patches from a series of volumes into a matrix
fn extract_matrix(data_set: &[&mut [Complex32]], vol_dims: [usize; 3], matrix: &mut Array2<Complex32>, patch_range: &[Range<usize>; 3]) {
    // infer matrix dims from data set slice
    let n_columns = data_set.len();
    let n_rows =
        (patch_range[0].end - patch_range[0].start) *
            (patch_range[1].end - patch_range[1].start) *
            (patch_range[2].end - patch_range[2].start);

    assert_eq!(n_rows * n_columns, matrix.len(), "n_rows and m_columns inconsistent with matrix data size");

    let mut xr = vec![0usize; patch_range[0].end - patch_range[0].start];
    let mut yr = vec![0usize; patch_range[1].end - patch_range[1].start];
    let mut zr = vec![0usize; patch_range[2].end - patch_range[2].start];

    // account for out-of-range indices by performing modulo over volume size
    patch_range[0].clone().zip(&mut xr).for_each(|(x_idx, xr)| *xr = x_idx % vol_dims[0]);
    patch_range[1].clone().zip(&mut yr).for_each(|(y_idx, yr)| *yr = y_idx % vol_dims[1]);
    patch_range[2].clone().zip(&mut zr).for_each(|(z_idx, zr)| *zr = z_idx % vol_dims[2]);

    // fill matrix with patch data assuming col-major ordering for both matrix and volumes
    let matrix_data = matrix.as_slice_memory_order_mut().expect("matrix is not contiguous");
    matrix_data.fill(Complex32::ZERO);
    let mut mat_idx = 0;
    for vol in data_set {
        for &x in &xr {
            for &y in &yr {
                for &z in &zr {
                    let col_maj_idx = x + y * vol_dims[0] + z * vol_dims[0] * vol_dims[1];
                    matrix_data[mat_idx] = vol[col_maj_idx];
                    mat_idx += 1;
                }
            }
        }
    }
}

fn insert_matrix(data_set: &mut [&mut [Complex32]], vol_dims: [usize; 3], matrix: &Array2<Complex32>, patch_range: &[Range<usize>; 3]) {
    // infer matrix dims from data set slice
    let n_columns = data_set.len();
    let n_rows =
        (patch_range[0].end - patch_range[0].start) *
            (patch_range[1].end - patch_range[1].start) *
            (patch_range[2].end - patch_range[2].start);

    assert_eq!(n_rows * n_columns, matrix.len(), "n_rows and m_columns inconsistent with matrix data size");

    let mut xr = vec![0usize; patch_range[0].end - patch_range[0].start];
    let mut yr = vec![0usize; patch_range[1].end - patch_range[1].start];
    let mut zr = vec![0usize; patch_range[2].end - patch_range[2].start];

    // account for out-of-range indices by performing modulo over volume size
    patch_range[0].clone().zip(&mut xr).for_each(|(x_idx, xr)| *xr = x_idx % vol_dims[0]);
    patch_range[1].clone().zip(&mut yr).for_each(|(y_idx, yr)| *yr = y_idx % vol_dims[1]);
    patch_range[2].clone().zip(&mut zr).for_each(|(z_idx, zr)| *zr = z_idx % vol_dims[2]);

    let matrix_data = matrix.as_slice_memory_order().expect("matrix is not contiguous");
    let mut mat_idx = 0;
    for vol in data_set.iter_mut() {
        for &x in &xr {
            for &y in &yr {
                for &z in &zr {
                    let col_maj_idx = x + y * vol_dims[0] + z * vol_dims[0] * vol_dims[1];
                    vol[col_maj_idx] = matrix_data[mat_idx];
                    mat_idx += 1;
                }
            }
        }
    }
}

fn generate_patch_ranges(vol_size: impl Into<[usize; 3]>, patch_size: impl Into<[usize; 3]>, offset: impl Into<[usize; 3]>) -> Vec<[Range<usize>; 3]> {
    let vol_size = vol_size.into();
    let patch_size = patch_size.into();
    let offset = offset.into();
    let n_x = vol_size[0] / patch_size[0];
    let n_y = vol_size[1] / patch_size[1];
    let n_z = vol_size[2] / patch_size[2];
    let mut patches = Vec::<[Range<usize>; 3]>::with_capacity(n_x * n_y * n_z);
    for x in 0..n_x {
        let x_start = &x * patch_size[0];
        for y in 0..n_y {
            let y_start = &y * patch_size[1];
            for z in 0..n_z {
                let z_start = &z * patch_size[2];
                let xr = (x_start + offset[0])..(x_start + patch_size[0] + offset[0]);
                let yr = (y_start + offset[1])..(y_start + patch_size[1] + offset[1]);
                let zr = (z_start + offset[2])..(z_start + patch_size[2] + offset[2]);
                patches.push([xr, yr, zr])
            }
        }
    }
    patches
}

fn low_rank_approx(input_mat: &mut Array2<Complex32>, rank: usize, tmp_s: &mut Array2<Complex32>) {
    let (u, mut s, v) = input_mat.svddc(JobSvd::Some).unwrap();

    let n = input_mat.dim().0.min(input_mat.dim().1);

    assert_eq!(tmp_s.dim(), (n, n), "tmp_s must have dimension {:?}", (n, n));

    let u = u.unwrap();
    let v = v.unwrap();

    s.iter_mut().enumerate().for_each(|(i, val)| if i >= rank { *val = 0. });

    tmp_s.fill(Complex32::ZERO);
    let mut diag_view = tmp_s.diag_mut();
    diag_view.assign(
        &s.map(|x| Complex32::new(*x, 0.))
    );

    let lr_matrix = if u.len() != 1 {
        u.dot(tmp_s).dot(&v)
    } else { // if u is 1x1
        tmp_s.dot(&v)
    };

    input_mat.assign(&lr_matrix);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    #[test]
    fn it_works() {
        let patch_size = [10, 10, 10];

        let mut volumes: Vec<Array3<Complex32>> = (0..2).map(|_| gen_test_cfl()).collect();
        let matrix_size = [patch_size.iter().product(), volumes.len()];
        let d = volumes[0].dim();
        let mut vol_data = volumes.iter_mut().map(|vol| vol.as_slice_memory_order_mut().unwrap()).collect::<Vec<_>>();

        println!("generating patch ranges ...");
        let patch_ranges = generate_patch_ranges(d, patch_size, [9, 0, 2]);
        println!("generated {} patches ...", patch_ranges.len());
        let mut matrix = Array2::from_elem(matrix_size.f(), Complex32::ZERO);
        let n = matrix_size[0].min(matrix_size[1]); // min dim size
        let mut tmp_mat = Array2::from_elem((n, n).f(), Complex32::ZERO);

        let now = Instant::now();
        for patch in &patch_ranges {
            extract_matrix(&vol_data, d.into(), &mut matrix, patch);
            low_rank_approx(&mut matrix, 1, &mut tmp_mat);
            insert_matrix(&mut vol_data, d.into(), &matrix, patch);
        }
        let elapsed = now.elapsed().as_millis();

        println!("done!");
        println!("took {elapsed} ms");
    }

    fn gen_test_cfl() -> Array3<Complex32> {
        Array3::from_shape_fn((128, 128, 128).f(), |(x, y, z)| { x as f32 * Complex32::ONE })
    }
}