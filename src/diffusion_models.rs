use cfl::ndarray::{Array1, Array2, Array4, Axis, ShapeBuilder};
use cfl::ndarray_linalg::{EigValsh, LeastSquaresSvdInPlace, UPLO};
use cfl::num_complex::{Complex32, ComplexFloat};
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

#[test]
fn test_calc() {
    let data_set_dir = Path::new("B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32/cfl");
    let output_eig = Path::new("B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32");
    let filename_fn = |i: usize| format!("vol{:03}", i);


    let gx: Vec<f32> = vec![
        0.0, 0.033794, -0.96668, 0.31221, -0.29079, 0.72879, -0.72294, 0.35385, -0.53911, 0.81587,
        0.68086, 0.20412, -0.16191, 0.0, -0.61243, -0.95557, -0.14926, 0.4058, -0.82939, 0.53431,
        -0.4765, -0.23953, 0.92376, -0.40902, 0.6654, -0.023457, 0.0, 0.33294, 0.1399, -0.59938,
        -0.78006, 0.25721, 0.73355, -0.7928, -0.7225, 0.95202, 0.29426, -0.077671, 0.53455, 0.0,
        0.82325, -0.038707, 0.41348, -0.94387, -0.10143, 0.69104, -0.23112, -0.38169, 0.21563,
        0.52636, 0.93641, -0.53963, 0.0, -0.14172, -0.89182, -0.16908, 0.49378, 0.84456, -0.71776,
        0.64397, 0.43481, 0.15647, 0.078755, -0.49345, -0.99273, 0.0, 0.32967, -0.3883, -0.60483,
        -0.1469, 0.86123, -0.38573, -0.47056, 0.58351, 0.93105, -0.8611, 0.47787, 0.065866, 0.0,
        -0.76348, 0.18667, 0.71877, -0.25615, -0.077421, -0.85673, 0.78902, -0.74638, -0.22027,
        0.44745, 0.10268, 0.13143, 0.0, 0.99056, 0.090782, -0.57635, 0.55738, 0.54854, -0.45271,
        -0.88313, -0.31844, 0.076568, -0.88705, -0.67614, 0.61312, 0.0, -0.67729, 0.56989, 0.022258,
        -0.017076, 0.21131, 0.98653, -0.56648, -0.39722, -0.002568, 0.84825, -0.65414, 0.26805, 0.0,
        0.72315, -0.93849, 0.3104, 0.37924, -0.53648, -0.32909, -0.36561, 0.85867, -0.21945, 0.72011,
        -0.84588, 0.3432, 0.0
    ];

    let gy: Vec<f32> = vec![
        0.0, 0.12502, -0.2414, -0.94934, -0.75857, -0.15591, 0.44343, 0.82477, -0.11618, -0.55066,
        0.45849, -0.53052, 0.66058, 0.0, -0.76049, 0.26785, -0.97671, 0.32053, -0.21047, -0.72201,
        0.78032, -0.42142, 0.15758, 0.30625, 0.73029, 0.94, 0.0, -0.097442, -0.84259, -0.47081,
        0.59399, 0.54176, -0.38565, -0.55482, 0.21292, -0.084844, 0.95515, -0.26462, -0.84509, 0.0,
        0.54326, 0.33865, -0.51466, 0.12869, 0.82459, 0.070098, -0.89627, 0.039049, -0.93923,
        0.6462, -0.31818, 0.60868, 0.0, -0.64936, -0.34472, 0.97317, 0.10101, 0.38376, -0.026826,
        -0.73177, 0.87788, -0.31945, 0.71696, -0.65617, -0.01164, 0.0, -0.69856, 0.54722, 0.76936,
        -0.035574, -0.19268, -0.90423, -0.34763, -0.34175, 0.33052, 0.40959, 0.50046, 0.99276, 0.0,
        -0.43018, 0.34648, -0.57069, 0.88683, -0.82142, 0.023666, 0.24788, -0.6638, 0.45523,
        -0.86532, -0.096146, 0.85324, 0.0, -0.12119, -0.69784, 0.39252, 0.7515, -0.1219, -0.8035,
        0.46792, -0.1966, -0.99338, -0.4585, 0.62262, 0.29143, 0.0, -0.25923, -0.56396, 0.54353,
        -0.49325, 0.93693, 0.10669, 0.13677, 0.89737, -0.93942, 0.027588, -0.63247, 0.12371, 0.0,
        -0.69066, -0.11898, 0.70313, -0.3136, -0.84385, 0.74153, -0.56105, -0.37863, 0.19724,
        0.58412, 0.25661, -0.83127, 0.0
    ];

    let gz: Vec<f32> = vec![
        0.0, 0.99158, 0.085124, 0.035624, 0.58311, 0.66675, 0.52984, 0.44107, 0.83418, 0.17641,
        0.57115, 0.82273, 0.73309, 0.0, 0.21584, 0.12309, 0.15414, 0.85591, 0.51751, 0.43955,
        0.40502, 0.87466, 0.34904, 0.8596, 0.15467, 0.34038, 0.0, 0.9379, 0.52006, 0.64736,
        0.19667, 0.80021, 0.55962, 0.25227, 0.65777, 0.29405, 0.033191, 0.96122, 0.009124, 0.0,
        0.16474, 0.94011, 0.7511, 0.30422, 0.55657, 0.71941, 0.37853, 0.92346, 0.2671, 0.55261,
        0.14796, 0.58164, 0.0, 0.74716, 0.29296, 0.15605, 0.8637, 0.37343, 0.69578, 0.22319,
        0.20066, 0.93459, 0.69265, 0.57092, 0.11982, 0.0, 0.63509, 0.74147, 0.20558, 0.98851,
        0.47026, 0.18323, 0.811, 0.7367, 0.15459, 0.30124, 0.72193, 0.10048, 0.0, 0.48171,
        0.9193, 0.3971, 0.38461, 0.56504, 0.51523, 0.56214, 0.047787, 0.8627, 0.22586, 0.99006,
        0.50469, 0.0, 0.064022, 0.71048, 0.71677, 0.35296, 0.82719, 0.38658, 0.033626, 0.92733,
        0.085698, 0.054003, 0.39394, 0.73428, 0.0, 0.68853, 0.59764, 0.83909, 0.86972, 0.27841,
        0.12398, 0.81265, 0.19219, 0.34276, 0.52888, 0.41484, 0.95543, 0.0, 0.005659, 0.32416,
        0.63973, 0.87053, 0.010319, 0.58466, 0.74266, 0.34542, 0.95548, 0.3745, 0.46759, 0.43728,
        0.0
    ];

    let b: Vec<f32> = vec![
        70.523783, 4001.0782, 4001.0792, 4001.0841, 4001.084, 4001.0815, 4001.0822, 4001.0799,
        4001.0807, 4001.0837, 4001.0806, 4001.0805, 4001.0798, 70.523783, 4001.0821, 4001.0806,
        4001.0794, 4001.0822, 4001.0778, 4001.0783, 4001.0825, 4001.0762, 4001.0798, 4001.0826,
        4001.0841, 4001.0836, 70.523783, 4001.0781, 4001.0829, 4001.0782, 4001.0764, 4001.0802,
        4001.0865, 4001.0799, 4001.0783, 4001.0852, 4001.0775, 4001.0829, 4001.0798, 70.523783,
        4001.0796, 4001.0778, 4001.0818, 4001.0785, 4001.0858, 4001.0836, 4001.083, 4001.0771,
        4001.0782, 4001.0816, 4001.0837, 4001.0799, 70.523783, 4001.0835, 4001.0798, 4001.0793,
        4001.0822, 4001.0764, 4001.083, 4001.0808, 4001.0836, 4001.0849, 4001.084, 4001.0806,
        4001.0792, 70.523783, 4001.0775, 4001.0803, 4001.0798, 4001.0824, 4001.0817, 4001.0775,
        4001.0829, 4001.0831, 4001.0841, 4001.0834, 4001.0826, 4001.0829, 70.523783, 4001.0814,
        4001.083, 4001.0795, 4001.0825, 4001.0767, 4001.0803, 4001.0815, 4001.0786, 4001.0771,
        4001.0791, 4001.0809, 4001.0811, 70.523783, 4001.0819, 4001.0853, 4001.0814, 4001.0815,
        4001.0829, 4001.0832, 4001.0784, 4001.0776, 4001.084, 4001.0768, 4001.0821, 4001.0818,
        70.523783, 4001.0804, 4001.0829, 4001.0837, 4001.0835, 4001.0843, 4001.0794, 4001.0835,
        4001.0828, 4001.083, 4001.0797, 4001.0843, 4001.0787, 70.523783, 4001.0781, 4001.085,
        4001.0823, 4001.08, 4001.0768, 4001.0821, 4001.0851, 4001.0829, 4001.0809, 4001.0793,
        4001.0782, 4001.0847, 70.523783
    ];

    let b0_mask = b.iter().map(|&x| x < 100.).collect::<Vec<bool>>();

    let n_bvecs = b0_mask.iter().filter(|&x| !x).count();
    println!("n bvecs: {}", n_bvecs);

    let n_total_vols = 131;
    let data_set_size = [512, 284, 228, n_total_vols];
    let vol_stride = data_set_size[0..3].iter().product();

    let vols_indices_to_omit = vec![];

    let mut data_iterate = Array4::<Complex32>::from_elem(data_set_size.f(), Complex32::new(0.0, 0.0));
    let data = data_iterate.as_slice_memory_order_mut().unwrap();

    let vols_to_read = (0..n_total_vols).filter(|i| !vols_indices_to_omit.contains(&i)).collect::<Vec<_>>();
    println!("reading data set volumes: {:?}", vols_to_read);
    let now = Instant::now();
    data.par_chunks_exact_mut(vol_stride).zip(vols_to_read).for_each(|(vol, i)| {
        let vol_name = filename_fn(i);
        let x = cfl::to_array(data_set_dir.join(vol_name), true);
        vol.copy_from_slice(x.unwrap().as_slice_memory_order().unwrap());
    });
    let dur = now.elapsed();
    println!("data set read in {} sec", dur.as_secs());

    let g = gx.iter().zip(gy.iter().zip(gz.iter())).zip(&b0_mask).filter(|((_, (_, _)), &m)| !m).map(|((&x, (&y, &z)), m)| {
        [x, y, z]
    }).collect::<Vec<_>>();

    let b = b.iter().zip(&b0_mask).filter(|(_, &m)| !m).map(|(&b, _)| b).collect::<Vec<f32>>();

    println!("b: {:?}", b);

    let mut a = Array2::zeros((n_bvecs, 6).f());
    build_design_matrix(&b, &g, a.as_slice_memory_order_mut().unwrap());

    let design_mat_entries = a.as_slice_memory_order().unwrap();


    let mut result = Array4::zeros((3, 512, 284, 228).f());
    let r = result.as_slice_memory_order_mut().unwrap();

    let data = data_iterate.as_slice_memory_order().unwrap();

    println!("computing tensors ...");
    r.par_chunks_exact_mut(3).enumerate().for_each(|(i, eigs)| {
        let mut rhs = vec![0f32; n_bvecs];
        let mut signal_buff = vec![0f32; n_total_vols];
        let mut tensor_buff = vec![0f32; 6];
        let mut calc_mat = Array2::zeros(a.dim().f());
        let mut calc_vec = Array1::zeros(rhs.len().f());
        let mut eig_calc_mat = Array2::zeros((3, 3).f());
        let mut eigvals = Array1::zeros(3.f());

        signal_buff.iter_mut().enumerate().for_each(|(v_idx, s)| {
            *s = data[vol_stride * v_idx + i].abs();
        });

        build_rhs(&signal_buff, &b0_mask, &mut rhs);
        solve_tensor(&rhs, &design_mat_entries, &mut tensor_buff, &mut calc_mat, &mut calc_vec);
        solve_eigenvalues(&tensor_buff, &mut eig_calc_mat, &mut eigvals);

        eigs[0] = eigvals[2];
        eigs[1] = eigvals[1];
        eigs[2] = eigvals[0];
    });


    let lamb1 = result.index_axis(Axis(0), 0);
    let lamb2 = result.index_axis(Axis(0), 1);
    let lamb3 = result.index_axis(Axis(0), 2);

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

    solve_eigenvalues(&tensor_entries, &mut scratch_mat, &mut eig);

    println!("{:?}", eig);
}

fn solve_eigenvalues(tensor_entries: &[f32], scratch_matrix: &mut Array2<f32>, eigenvalues: &mut Array1<f32>) {
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

    *eigenvalues = scratch_matrix.eigvalsh(UPLO::Lower).unwrap();
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
fn build_rhs(signal: &[f32], b0_mask: &[bool], rhs: &mut [f32]) {
    assert_eq!(signal.len(), b0_mask.len(), "signal vector and b0 mask must have the same length");
    // calculate the mean b0 signal (s0)
    let mut b0_count: usize = 0;
    let s0 = b0_mask.iter().zip(signal.iter())
        .filter(|(&m, s)| {
            if m { b0_count += 1; }
            !m
        })
        .map(|(_, s)| s)
        .sum::<f32>() / b0_count as f32;

    let s0 = s0.max(f32::MIN_POSITIVE);

    assert_eq!(rhs.len(), signal.len() - b0_count, "unexpected lhs length");

    // calculate the left-hand side
    signal.iter().zip(b0_mask)
        .filter(|(s, b)| !**b) // filter out b0s
        .map(|(&s, _)| (s / s0).clamp(f32::MIN_POSITIVE, 1.0).ln()) // calculate signal attenuation in log space
        .zip(rhs.iter_mut()) // zip with result vector
        .for_each(|(s, l)| {
            //println!("s = {}", s);
            *l = s
        }); // write result signal into lhs vector
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