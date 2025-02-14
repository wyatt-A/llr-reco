use crate::data_import::DtiDataSetBuilder;
use cfl::dump;
use cfl::ndarray::{Array1, Array2, Array3, Array4, Axis, ShapeBuilder};
use cfl::ndarray_linalg::{Eigh, LeastSquaresSvdInPlace, UPLO};
use cfl::num_complex::ComplexFloat;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

#[test]
fn test_calc() {
    let output_eig = Path::new("B:\\ProjectSpace\\wa41\\llr-reco-test-data\\recon_results\\dti_scalars");

    let cfl_dir = "B:/ProjectSpace/wa41/llr-reco-test-data/image_data/xformed/xformed_affine/cfl";
    let cfl_pattern = |i: usize| format!("{cfl_dir}/vol{:03}", i);

    let b_table = "B:/ProjectSpace/wa41/llr-reco-test-data/image_data/b_vec_info/btable";
    let b0_null_vec_tol = 1e-6;

    let cfl_images = (0..=130).collect::<Vec<_>>();
    let dataset = DtiDataSetBuilder::from_cfl_images(&cfl_images, &cfl_pattern)
        .with_b_table(b_table, b0_null_vec_tol);

    println!("loading data ...");
    let dti = dataset.load_dti_set();
    let b0s = dataset.load_b0_set();

    let [nx, ny, nz, nq]: [usize; 4] = dti.dim().into();
    let [_, _, _, nb0]: [usize; 4] = b0s.dim().into();

    let vol_stride = nx * ny * nz;

    println!("computing b0 average ...");
    let mut b0_average = vec![0f32; nx * ny * nz];
    let b0_data = b0s.as_slice_memory_order().unwrap();
    for vol in b0_data.chunks_exact(nx * ny * nz) {
        b0_average.par_iter_mut().zip(vol.par_iter()).for_each(|(a, b)| {
            *a += b.abs();
        })
    }
    b0_average.par_iter_mut().for_each(|x| *x /= nb0 as f32);


    let b0_average = Array3::from_shape_vec((nx, ny, nz).f(), b0_average).unwrap();

    println!("writing b0 average ...");
    dump(output_eig.join("b0av"), &b0_average.clone().into_dyn());


    let b0_mask = dataset.b0_mask();
    let g = dataset.b_vecs().iter().zip(b0_mask).filter(|(_, b)| !**b).map(|(g, _)| *g).collect::<Vec<_>>();
    let b = dataset.b_vals().iter().zip(b0_mask).filter(|(_, b)| !**b).map(|(g, _)| *g).collect::<Vec<_>>();

    let mut a = Array2::zeros((nq, 6).f());
    build_design_matrix(&b, &g, a.as_slice_memory_order_mut().unwrap());
    let design_mat_entries = a.as_slice_memory_order().unwrap();

    let mut result = Array4::zeros((6, nx, ny, nz).f());
    let r = result.as_slice_memory_order_mut().unwrap();

    let dti_data = dti.as_slice_memory_order().unwrap();
    let b0_data = b0_average.as_slice_memory_order().unwrap();

    println!("computing tensors ...");
    r.par_chunks_exact_mut(6).enumerate().for_each(|(i, scalars)| {
        let mut rhs = vec![0f32; nq];
        let mut signal_buff = vec![0f32; nq];
        let mut tensor_buff = vec![0f32; 6];
        let mut calc_mat = Array2::zeros(a.dim().f());
        let mut calc_vec = Array1::zeros(rhs.len().f());
        let mut eig_calc_mat = Array2::zeros((3, 3).f());
        let mut eigvals = Array1::zeros(3.f());
        let mut eigvec = Array1::zeros(3.f());

        signal_buff.iter_mut().enumerate().for_each(|(v_idx, s)| {
            *s = dti_data[vol_stride * v_idx + i].abs();
        });

        build_rhs(&signal_buff, b0_data[i], &mut rhs);
        solve_tensor(&rhs, &design_mat_entries, &mut tensor_buff, &mut calc_mat, &mut calc_vec);
        solve_eigenvalues(&tensor_buff, &mut eig_calc_mat, &mut eigvals, &mut eigvec);

        scalars[0] = eigvals[2];
        scalars[1] = eigvals[1];
        scalars[2] = eigvals[0];

        scalars[3] = eigvec[2];
        scalars[4] = eigvec[1];
        scalars[5] = eigvec[0];
    });


    let lamb1 = result.index_axis(Axis(0), 0);
    let lamb2 = result.index_axis(Axis(0), 1);
    let lamb3 = result.index_axis(Axis(0), 2);

    let px = result.index_axis(Axis(0), 3);
    let py = result.index_axis(Axis(0), 4);
    let pz = result.index_axis(Axis(0), 5);

    println!("computing FA ...");
    let md = (&lamb1 + &lamb2 + &lamb3) / 3.;
    let num = ((&lamb1 - &md).map(|x| x.powi(2)) + (&lamb2 - &md).map(|x| x.powi(2)) + (&lamb3 - &md).map(|x| x.powi(2))).map(|x| x.sqrt());
    let denom = (lamb1.map(|x| x.powi(2)) + lamb2.map(|x| x.powi(2)) + lamb3.map(|x| x.powi(2))).map(|x| x.sqrt());
    let fa = (3. / 2.).sqrt() * num / denom;

    println!("writing nifti ...");
    cfl::dump(output_eig.join("fa"), &fa.into_dyn());
    cfl::dump(output_eig.join("e1"), &lamb1.to_owned().into_dyn());
    cfl::dump(output_eig.join("e2"), &lamb2.to_owned().into_dyn());
    cfl::dump(output_eig.join("e3"), &lamb3.to_owned().into_dyn());

    cfl::dump(output_eig.join("px"), &px.to_owned().into_dyn());
    cfl::dump(output_eig.join("py"), &py.to_owned().into_dyn());
    cfl::dump(output_eig.join("pz"), &pz.to_owned().into_dyn());
}


#[test]
fn test_solve_system() {
    let design_mat_entries = vec![7., 5., 4., 2., 2., 4., 2., 10., 8., 6., 9., 1., 4., 4., 10., 5., 1., 1., 1., 1.];
    let rhs_entries = vec![21., 19., 25., 18.];
    let mut design_mat = Array2::zeros((4, 5).f());
    let mut rhs_vec = Array1::zeros(4.f());
    let mut results = vec![0.; 5];

    let now = Instant::now();
    solve_tensor(&rhs_entries, &design_mat_entries, &mut results, &mut design_mat, &mut rhs_vec);
    let dur = now.elapsed();

    println!("{:?}", results);
    println!("dur: {} us", dur.as_micros());
}

#[test]
fn test_eig() {
    let tensor_entries = [2., 1.5, 1., 0., 0., 0.];
    let mut scratch_mat = Array2::zeros((3, 3).f());
    let mut eig = Array1::zeros(3.f());
    let mut eig_vec = eig.clone();

    solve_eigenvalues(&tensor_entries, &mut scratch_mat, &mut eig, &mut eig_vec);

    println!("{:?}", eig);
}

fn solve_eigenvalues(tensor_entries: &[f32], scratch_matrix: &mut Array2<f32>, eigenvalues: &mut Array1<f32>, principle_eigenvector: &mut Array1<f32>) {
    assert!(scratch_matrix.nrows() == scratch_matrix.nrows() && scratch_matrix.nrows() == 3, "scratch matrix must be a 3 x 3 matrix");

    let s_data = scratch_matrix.as_slice_memory_order_mut().unwrap();

    // insert the lower triangular elements

    s_data[0] = tensor_entries[0];
    s_data[4] = tensor_entries[1];
    s_data[8] = tensor_entries[2];

    s_data[1] = tensor_entries[3];
    //s_data[3] = tensor_entries[3];

    s_data[2] = tensor_entries[4];
    //s_data[6] = tensor_entries[4];

    s_data[5] = tensor_entries[5];
    //s_data[7] = tensor_entries[5];

    //*eigenvalues = scratch_matrix.eigvalsh(UPLO::Lower).unwrap();

    let (eigs, vecs) = scratch_matrix.eigh(UPLO::Lower).unwrap();

    *eigenvalues = eigs;
    principle_eigenvector.assign(&vecs.index_axis(Axis(1), 0));
}

/// solves the linear system given a design matrix and rhs vector (b). You must supply the scratch matrix
/// and scratch vector to solve the system in-place. Once this function is run, the scratch matrix and
/// vector are mutated.
fn solve_tensor(rhs: &[f32], design_matrix: &[f32], tensor_entries: &mut [f32], scratch_matrix: &mut Array2<f32>, scratch_vec: &mut Array1<f32>) {

    //assert_eq!(scratch_matrix.ncols(), 6, "expected 6 columns in scratch matrix");
    assert_eq!(scratch_vec.len(), scratch_matrix.nrows(), "expected {} rows in scratch matrix", scratch_vec.len());

    // insert design matrix elements into scratch matrix
    scratch_matrix.as_slice_memory_order_mut().unwrap().copy_from_slice(design_matrix);
    scratch_vec.as_slice_memory_order_mut().unwrap().copy_from_slice(rhs);
    //let x = scratch_matrix.solve_inplace(scratch_vec).expect("unable to solve matrix");
    let x = scratch_matrix.least_squares_in_place(scratch_vec).expect("unable to solve matrix");
    tensor_entries.copy_from_slice(x.solution.as_slice().unwrap());
}


/// writes to the left-hand-side vector for diffusion tensor calculation. The lhs vector must have
/// signal length - #b0 entries. The S0 is calculated based on true entries of b0_mask
fn build_rhs(signal: &[f32], s0: f32, rhs: &mut [f32]) {
    let s0 = s0.max(f32::MIN_POSITIVE);

    assert_eq!(rhs.len(), signal.len(), "unexpected lhs length");

    signal.iter()
        .map(|&a| (a / s0).clamp(f32::MIN_POSITIVE, 1.0).ln())
        .zip(rhs.iter_mut())
        .for_each(|(a, b)| *b = a);
}

/// writes to a design matrix A based on b-values and b-vectors. A is assumed to be in column-maj
/// memory layout where columns correspond to each of the 6 unique tensor entries {Dxx Dyy Dzz Dxy Dxz Dyz}
/// and the rows correspond to each non-zero b-value / b-vector pair
fn build_design_matrix(b: &[f32], g: &[[f32; 3]], a: &mut [f32]) {
    assert_eq!(b.len(), g.len(), "b and g must have the same number of entries");
    assert_eq!(6 * g.len(), a.len(), "a must have exactly 6 x the number of entries as g");

    // the number of non-zero b-vectors
    let m = b.len();

    // g-vector indices
    let gx = 0;
    let gy = 1;
    let gz = 2;

    a.iter_mut().enumerate().for_each(|(i, a)| {
        let col_index = i / m;
        let row_idx = i % m;
        // -b_m g_x_m ^ 2, -b_m g_y_m ^ 2, -b_m g_z_m ^ 2
        if col_index < 3 {
            *a = -b[row_idx] * g[row_idx][col_index].powi(2);
        }
        // - 2 b_m g_x_m g_y_m
        else if col_index == 3 {
            *a = -2.0 * b[row_idx] * g[row_idx][gx] * g[row_idx][gy]
        }
        // - 2 b_m g_x_m g_z_m
        else if col_index == 4 {
            *a = -2.0 * b[row_idx] * g[row_idx][gx] * g[row_idx][gz]
        }
        // - 2 b_m g_y_m g_z_m
        else {
            *a = -2.0 * b[row_idx] * g[row_idx][gy] * g[row_idx][gz]
        }
    });
}