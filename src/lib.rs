#[cfg(feature = "cuda")]
pub mod cuda {
    mod bindings;
    pub mod cufft;
    mod cuda_api;
    mod cublas;
    mod cusolver;
    pub mod low_rank;
}

use cfl;
use cfl::ndarray::{Array2, ShapeBuilder};
use cfl::ndarray_linalg::JobSvd;
use cfl::ndarray_linalg::SVDDC;
use cfl::num_complex::Complex32;
pub mod fftshift;

pub mod block;
mod array_utils;
pub mod signal_model;

pub enum LlrError {
    ExtractMatrix
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

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use std::time::Instant;
//     #[test]
//     fn it_works() {
//         let patch_size = [10, 10, 10];
//
//         let mut volumes: Vec<Array3<Complex32>> = (0..2).map(|_| gen_test_cfl()).collect();
//         let matrix_size = [patch_size.iter().product(), volumes.len()];
//         let d = volumes[0].dim();
//         let mut vol_data = volumes.iter_mut().map(|vol| vol.as_slice_memory_order_mut().unwrap()).collect::<Vec<_>>();
//
//         println!("generating patch ranges ...");
//         let patch_ranges = generate_patch_ranges(d, patch_size, [9, 0, 2]);
//         println!("generated {} patches ...", patch_ranges.len());
//         let mut matrix = Array2::from_elem(matrix_size.f(), Complex32::ZERO);
//         let n = matrix_size[0].min(matrix_size[1]); // min dim size
//         let mut tmp_mat = Array2::from_elem((n, n).f(), Complex32::ZERO);
//
//         let now = Instant::now();
//         for patch in &patch_ranges {
//             extract_matrix(&vol_data, d.into(), &mut matrix, patch);
//             low_rank_approx(&mut matrix, 1, &mut tmp_mat);
//             insert_matrix(&mut vol_data, d.into(), &matrix, patch);
//         }
//         let elapsed = now.elapsed().as_millis();
//
//         println!("done!");
//         println!("took {elapsed} ms");
//     }
//
//     fn gen_test_cfl() -> Array3<Complex32> {
//         Array3::from_shape_fn((128, 128, 128).f(), |(x, y, z)| { x as f32 * Complex32::ONE })
//     }
// }