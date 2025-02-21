#[cfg(feature = "cuda")]
pub mod cuda {
    pub mod bindings;
    pub mod cufft;
    pub mod cuda_api;
    pub mod cublas;
    mod cusolver;
    pub mod low_rank;
    pub mod cudwt3;
}

#[cfg(feature = "cuda")]
pub mod signal_model;

pub mod fftshift;

pub mod block;
mod array_utils;

#[cfg(feature = "cuda")]
pub mod data_import;
pub mod diffusion_models;
#[cfg(feature = "cuda")]
pub mod dti_subspace;

use crate::block::{grid_dim, lift_data_set, un_lift_data_set};
use crate::cuda::low_rank::cu_low_rank_approx_batch;
use crate::data_import::{under_sample_kspace, DtiDataSetBuilder};
use crate::dti_subspace::{conj_transpose_matrix, dti_back_project, dti_project};
use crate::signal_model::{estimate_linear_correction, signal_model_batched, ModelDirection};
use cfl;
use cfl::ndarray::{Array3, Array4, ArrayD, Axis, Ix2, ShapeBuilder};
use cfl::ndarray_linalg::SVDDC;
use cfl::num_complex::{Complex32, ComplexFloat};
use cfl::num_traits::Zero;
use clap::Parser;
use dwt;
use dwt::wavelet::WaveletFilter;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Serialize, Deserialize, Debug)]
pub struct LlrReconParams {
    /// size of the 3-D llr image block
    llr_block_size: [usize; 3],
    /// max random shift for llr block offset. Defaults to volume size if not specified
    max_rand_shift: Option<[usize; 3]>,
    /// max number of volumes per batch for fast fourier transforms - necessary for max GPU saturation
    fft_max_batch_size: usize,
    /// max number of matrices per batch for computations - necessary for max GPU saturation
    cas_mat_max_batch_size: usize,
    /// low-rank approximation of image data
    llr_block_rank: usize,
    /// wavelet coefficient shrinkage parameter
    wavelet_shrink_factor: f32,
    /// number of iterations
    n_iter: usize,
    /// subspace projection matrix stored as a 2-D col-major cfl
    subspace_projection_mat: Option<PathBuf>,
}

impl Default for LlrReconParams {
    fn default() -> Self {
        Self {
            llr_block_size: [32, 32, 32],
            max_rand_shift: None,
            fft_max_batch_size: 200,
            cas_mat_max_batch_size: 200,
            llr_block_rank: 6,
            wavelet_shrink_factor: 0.005,
            n_iter: 60,
            subspace_projection_mat: Some(PathBuf::from("/path/to/matrix.cfl")),
        }
    }
}


#[derive(Serialize, Deserialize, Parser, Debug)]
pub struct DataSetParams {
    /// path to recon params config .toml file
    pub recon_params: PathBuf,
    /// directory containing cfl files
    input_cfl_dir: PathBuf,
    /// glob-like filename pattern without the .cfl or .hdr extension
    input_cfl_pattern: String,
    /// output directory for resulting image cfl volumes
    output_cfl_dir: PathBuf,
    /// output file prefix. The prefix will be extended by the volume index padded by the appropriate number of 0s
    output_cfl_prefix: String,
    /// path to the b-table .txt file. This is whitespace-separated n-by-4 array of numerics specifying the b
    ///-values in first column followed by normalized gradient vectors in remaining 3 columns
    b_table: PathBuf,
    /// optional tolerance value used to determine b0 volumes. This is read as the magnitude of the
    /// gradient vector and is assumed to be exactly 0 if not specified
    b0_null_vec_tol: Option<f32>,
    /// optional path to a sampling mask stored as a 3-D cfl volume where each z-slice specifies the
    /// phase encoding scheme along the y and z axes. The x-axis is assumed to be fully-sampled
    sample_mask_cfl: Option<PathBuf>,
    /// optional expected number of cfl volumes to be found
    expected_volumes: Option<usize>,
    /// flag differentiating input volumes in image space or k-space
    is_image: Option<bool>,
    /// flag to specify b0 only reconstruction. If false, only dti volumes will be reconstructed.
    recon_b0: Option<bool>,
}

/// main reconstruction routine
pub fn llr_recon_exec(data_set_params: &DataSetParams, recon_params: &LlrReconParams) {
    let mut dataset = DtiDataSetBuilder::from_cfl_volume_dir(
        &data_set_params.input_cfl_dir,
        &data_set_params.input_cfl_pattern,
        data_set_params.expected_volumes,
    )
        .with_b_table(&data_set_params.b_table, data_set_params.b0_null_vec_tol.unwrap_or(0.));

    if let Some(sample_mask) = &data_set_params.sample_mask_cfl {
        dataset = dataset.with_sample_mask(sample_mask);
    }


    // load data
    println!("loading data ...");
    let recon_b0 = data_set_params.recon_b0.unwrap_or(false);
    let n_dataset_volumes = dataset.total_volumes();

    let (mut data, vol_indices) = if recon_b0 {
        (
            dataset.load_b0_set(),
            dataset.b0_indices()
        )
    } else {
        (
            dataset.load_dti_set(),
            dataset.dti_indices()
        )
    };

    let [nx, ny, nz, nq]: [usize; 4] = data.dim().into();

    assert_eq!(vol_indices.len(), nq, "expected number of volume indices to agree with nq");

    // assume no signal model corrections for now
    let lin_img_corrections = vec![[0., 0., 0.]; nq];
    let lin_ksp_corrections = lin_img_corrections.clone();

    let mut k_space = if data_set_params.is_image.unwrap_or(false) {
        println!("converting image data to k-space ...");
        let mut image_data = data;
        // run forward signal operator to transform to k-space
        signal_model_batched(
            &mut image_data,
            recon_params.fft_max_batch_size,
            &lin_ksp_corrections,
            &lin_img_corrections,
            ModelDirection::Forward,
        );
        // re-bind to a new name to avoid confusion because we did a fft
        let mut k_space = image_data;
        // return k-space data and data set dimensions
        k_space
    } else {
        data
    };

    // apply under sampling if specified
    if data_set_params.sample_mask_cfl.is_some() {
        //let sample_mask = dataset.full_sample_mask();
        let sample_mask = if recon_b0 {
            dataset.b0_sample_mask()
        } else {
            dataset.dti_sample_mask()
        };
        under_sample_kspace(&mut k_space, &sample_mask);
    }

    // re-bind to convenience variables
    let block_size = recon_params.llr_block_size;
    let vol_size = [nx, ny, nz];
    let vol_stride = nx * ny * nz;
    let rank = recon_params.llr_block_rank;
    let alpha = recon_params.wavelet_shrink_factor;
    let max_fft_batch_size = recon_params.fft_max_batch_size;
    let max_matrix_batch_size = recon_params.cas_mat_max_batch_size;
    let n_iter = recon_params.n_iter;

    // raw size of each llr block in number of samples
    let n_vox_per_block: usize = block_size.iter().product();
    // determine matrix dimensions for llr block processing
    let matrix_size: [usize; 2] = [block_size.iter().product(), nq];
    // determine the grid size in units of blocks over each volume
    let grid_size = grid_dim(&vol_size, &block_size);
    // raw number of llr blocks to process per iteration
    let total_blocks: usize = grid_size.iter().product();
    // convenience variables to describe matrix dimensions
    let [m, n] = matrix_size;

    // the SVD algorithm employed only works for matrices where m is larger than n
    assert!(m > n, "only tall matrices are supported");
    // the maximum rank is simply the number of columns in the matrix
    assert!(rank <= n, "rank must not exceed n");
    // the rank needs to be positive unless you want 0 for the output images
    assert!(rank > 0, "rank should not be less than 1");

    // load the subspace projection matrix and check for compatibility
    let subspace_matrix = recon_params.subspace_projection_mat.as_ref().map(|f| {
        println!("loading subspace matrix ...");
        let sub = cfl::to_array(f, true).expect("failed to read subspace cfl matrix")
            .into_dimensionality::<Ix2>().expect("matrix to have 2 non-singleton dimensions");
        let [nq_subspace, sub_rank]: [usize; 2] = sub.dim().into();
        assert_eq!(nq, nq_subspace, "unexpected subspace dimension");
        println!("loaded subspace matrix with rank {}", sub_rank);
        assert!(sub_rank >= rank, "llr rank must be less than or equal to subspace rank");
        sub
    });

    // assume that the supplied data serves as a measurement, which needs to be stored as immutable
    println!("cloning measurement data ...");
    let measurements = k_space.clone();
    let (under_sampling_factor, total_energy) = calc_undersampling_factor(&measurements);
    println!("under sampling factor: {:.03}", under_sampling_factor);

    println!("finding linear k-space corrections ...");
    // estimate linear k-space shift for rough image phase corrections
    let lin_ksp_corrections = vol_ref(&k_space).par_iter().map(|vol| {
        estimate_linear_correction(vol, &vol_size)
    }).collect::<Vec<_>>();

    // calculate the minimum energy solution (zero-filled)
    signal_model_batched(
        &mut k_space,
        max_fft_batch_size,
        &lin_ksp_corrections,
        &lin_img_corrections,
        ModelDirection::Inverse,
    );
    // re-bind to data iterate variable after transform
    let mut data_iterate = k_space;

    println!("allocating matrix data batch ...");

    let mut matrices = Array3::from_elem((m, n, max_matrix_batch_size).f(), Complex32::ZERO);
    let matrix_stride = m * n;

    let block_indices = (0..total_blocks).collect::<Vec<_>>();

    // ceiling division to calculate total number of matrix batches per iteration
    let n_matrix_batches = (total_blocks + max_matrix_batch_size - 1) / max_matrix_batch_size;

    // allocate an array to store the relative error at each iteration
    let mut relative_iterate_error_hist = Vec::<f64>::with_capacity(n_iter);
    let mut init_err: Option<f64> = None;
    let mut relative_mean_squared_error_hist = Vec::<f64>::with_capacity(n_iter);

    // determine the max random llr block shift
    let max_random_shift = recon_params.max_rand_shift.unwrap_or(vol_size);

    for it in 0..n_iter {
        println!("iteration {} of {}", it + 1, n_iter);

        // generate a random shift variable for shift-invariant denoising
        let shift = rand_shift(&max_random_shift);

        println!("beginning LLR block regularization ...");
        let now = Instant::now();
        for (batch_id, batch) in block_indices.chunks(max_matrix_batch_size).enumerate() {
            println!("working on matrix batch {} of {}", batch_id + 1, n_matrix_batches);

            let matrix_start_idx = *batch.first().unwrap();
            let n_matrices = *batch.last().unwrap() - matrix_start_idx + 1;
            let _matrix_data = matrices.as_slice_memory_order_mut().unwrap(); // the entire matrix data slice
            let matrix_data = &mut _matrix_data[0..(n_matrices * matrix_stride)]; // trimmed data slice

            let volumes = vol_ref(&data_iterate);
            lift_data_set(&volumes, matrix_data, &vol_size, &block_size, &shift, matrix_start_idx, n_matrices);

            if let Some(subspace_matrix) = subspace_matrix.as_ref() {
                let [nq, sub_rank]: [usize; 2] = subspace_matrix.dim().into();
                let subspace_matrix = subspace_matrix.as_slice_memory_order().unwrap();
                let mut compressed_data = vec![Complex32::zero(); sub_rank * n_vox_per_block * n_matrices];
                dti_project(subspace_matrix, matrix_data, &mut compressed_data, n_matrices, nq, n_vox_per_block, sub_rank);
                conj_transpose_matrix(&mut compressed_data, sub_rank, n_vox_per_block, n_matrices);
                cu_low_rank_approx_batch(n_vox_per_block, sub_rank, rank, n_matrices, &mut compressed_data);
                conj_transpose_matrix(&mut compressed_data, n_vox_per_block, sub_rank, n_matrices);
                dti_back_project(subspace_matrix, matrix_data, &compressed_data, n_matrices, nq, n_vox_per_block, sub_rank);
            } else {
                cu_low_rank_approx_batch(m, n, rank, n_matrices, matrix_data);
            }

            let mut volumes = vol_ref_mut(&mut data_iterate);
            un_lift_data_set(&mut volumes, matrix_data, &vol_size, &block_size, &shift, matrix_start_idx, n_matrices);
        }
        let dur = now.elapsed();
        println!("llr regularization took  {} seconds", dur.as_secs());

        println!("performing wavelet denoising ...");

        let now = Instant::now();
        let mut data = data_iterate.as_slice_memory_order_mut().unwrap();
        data.par_chunks_exact_mut(vol_stride).enumerate().for_each(|(i, vol)| {
            println!("denoising vol {} ...", i + 1);
            wavelet_denoise(vol, &vol_size, alpha);
        });
        let dur = now.elapsed();
        println!("wavelet regularization took  {} seconds", dur.as_secs());

        signal_model_batched(
            &mut data_iterate,
            max_fft_batch_size,
            &lin_ksp_corrections,
            &lin_img_corrections,
            ModelDirection::Forward,
        );

        let sum_sq_error = hard_project(&mut data_iterate, &measurements);
        let relative_iterate_error = sum_sq_error / *init_err.get_or_insert(sum_sq_error);
        let relative_mean_squared_error = sum_sq_error / total_energy;
        relative_iterate_error_hist.push(relative_iterate_error);
        relative_mean_squared_error_hist.push(relative_mean_squared_error);
        println!("current relative error: {:.03}", relative_iterate_error);
        println!("mse / total energy: {:.03e}", sum_sq_error / total_energy);

        signal_model_batched(
            &mut data_iterate,
            max_fft_batch_size,
            &lin_ksp_corrections,
            &lin_img_corrections,
            ModelDirection::Inverse,
        );

        if it % 10 == 0 {
            println!("writing preview volume...");
            let width = n_iter.to_string().len();
            let vol = data_iterate.index_axis(Axis(3), 0).to_owned();
            cfl::dump_magnitude(data_set_params.output_cfl_dir.join(format!("preview_i{it:0width$}", width = width)), &vol.into_dyn());
        }
    }

    // iterations complete

    // write error history to json file
    //let mut err_hist_string = serde_json::to_string_pretty(&relative_iterate_error_hist).expect("JSON serialization failed");
    let mut err_hist_string = serde_json::to_string_pretty(&relative_mean_squared_error_hist).expect("JSON serialization failed");
    err_hist_string.push_str("\n");
    let mut f = File::create(data_set_params.output_cfl_dir.join("error_history.json"))
        .expect("failed to create json file");
    f.write_all(err_hist_string.as_bytes()).expect("failed to write to json file");

    // write cfl volumes
    println!("writing results ...");
    let filename_enum_width = n_dataset_volumes.to_string().len();
    let raw_data = data_iterate.as_slice_memory_order().unwrap();
    raw_data.par_chunks_exact(nx * ny * nz).zip(vol_indices).for_each(|(vol, idx)| {
        let filepath = data_set_params.output_cfl_dir.join(
            format!("{}{idx:0width$}", &data_set_params.output_cfl_prefix, width = filename_enum_width)
        );
        cfl::write_buffer(filepath, &[nx, ny, nz], vol).unwrap();
    })
}

fn llr_approx(matrices: &mut Array3<Complex32>, rank: usize) {
    let [m, n, batch_size]: [usize; 3] = matrices.dim().into();
    assert!(rank <= n, "rank must not exceed n");
    assert!(rank > 0, "rank should not be less than 1");
    cu_low_rank_approx_batch(m, n, rank, batch_size, matrices.as_slice_memory_order_mut().unwrap());
}

/// decomposes a 4-D data set into a series of data slices pointing
/// to each volume individually
fn vol_ref<T>(dataset: &Array4<T>) -> Vec<&[T]> {
    let [x, y, z, _]: [usize; 4] = dataset.dim().into();
    let vol_stride = x * y * z;
    let vol_data = dataset.as_slice_memory_order().unwrap();
    vol_data.chunks_exact(vol_stride).collect()
}

/// decomposes a 4-D data set into a series of data slices pointing
/// to each volume individually
fn vol_ref_mut<T>(dataset: &mut Array4<T>) -> Vec<&mut [T]> {
    let [x, y, z, _]: [usize; 4] = dataset.dim().into();
    let vol_stride = x * y * z;
    let vol_data = dataset.as_slice_memory_order_mut().unwrap();
    vol_data.chunks_exact_mut(vol_stride).collect()
}

/// insert measurements into dataset where the measurement is non-zero, returning the sum of the
/// squared error between the dataset and the measurements
fn hard_project(dataset: &mut Array4<Complex32>, measurements: &Array4<Complex32>) -> f64 {
    assert_eq!(dataset.dim(), measurements.dim(), "dataset and measurements must have the same dimensions");
    let x = dataset.as_slice_memory_order_mut().unwrap();
    let y = measurements.as_slice_memory_order().unwrap();
    let sum_squared_error = x.par_iter_mut().zip(y.par_iter()).filter_map(|(x, y)| {
        if !y.is_zero() {
            let err = (*x - *y).norm_sqr() as f64;
            *x = *y;
            Some(err)
        } else {
            None
        }
    }).sum::<f64>();
    sum_squared_error
}

/// returns the under sampling factor and total energy of samples
fn calc_undersampling_factor(measurements: &Array4<Complex32>) -> (f64, f64) {
    let x = measurements.as_slice_memory_order().unwrap();
    let non_zero = x.par_iter().map(|x| {
        (!x.is_zero()) as usize
    }).sum::<usize>() as f64;
    let undersampling = x.len() as f64 / non_zero;
    let total_energy = x.par_iter().map(|x| x.norm_sqr() as f64).sum::<f64>();
    (undersampling, total_energy)
}

/// return a random shift up to max
fn rand_shift(max: &[usize; 3]) -> [usize; 3] {
    let mut rng = rand::rng();
    let mut result = [0usize; 3];
    result.iter_mut().zip(max.iter()).for_each(|(r, &m)| {
        *r = rng.random_range(0..m);
    });
    result
}

/// denoise a volume with wavelet coefficient shrinkage
fn wavelet_denoise(volume_data: &mut [Complex32], volume_size: &[usize; 3], alpha: f32) {

    // create temp array to store coefficients
    let mut x = ArrayD::<Complex32>::zeros(volume_size.as_slice().f());

    // insert data from buffer
    x.as_slice_memory_order_mut().unwrap().copy_from_slice(volume_data);

    // set up wavelet and max levels
    let w = dwt::wavelet::Wavelet::new(dwt::wavelet::WaveletType::Daubechies2);
    let s_len = *x.shape().iter().min().unwrap();
    let n_levels = dwt::w_max_level(s_len, w.filt_len());

    // do wavelet decomposition on volume
    let mut dec = dwt::wavedec3(x, w, n_levels);

    // shrink coefficients
    for sub in dec.subbands.iter_mut() {
        sub.par_mapv_inplace(|x| {
            shrink_complex(x, alpha)
        });
    }

    // reconstruct volume from coefficients
    let y = dwt::waverec3(dec);

    // copy data back to buffer
    volume_data.copy_from_slice(y.as_slice_memory_order().unwrap());
}

fn shrink_complex(x: Complex32, alpha: f32) -> Complex32 {
    let m = x.abs();
    let d = (m - alpha).max(0.);
    let shrink_factor = d / m;
    x.scale(shrink_factor)
}

// fn low_rank_approx(input_mat: &mut Array2<Complex32>, rank: usize, tmp_s: &mut Array2<Complex32>) {
//     let (u, mut s, v) = input_mat.svddc(JobSvd::Some).unwrap();
//
//     let n = input_mat.dim().0.min(input_mat.dim().1);
//
//     assert_eq!(tmp_s.dim(), (n, n), "tmp_s must have dimension {:?}", (n, n));
//
//     let u = u.unwrap();
//     let v = v.unwrap();
//
//     s.iter_mut().enumerate().for_each(|(i, val)| if i >= rank { *val = 0. });
//
//     tmp_s.fill(Complex32::ZERO);
//     let mut diag_view = tmp_s.diag_mut();
//     diag_view.assign(
//         &s.map(|x| Complex32::new(*x, 0.))
//     );
//
//     let lr_matrix = if u.len() != 1 {
//         u.dot(tmp_s).dot(&v)
//     } else { // if u is 1x1
//         tmp_s.dot(&v)
//     };
//
//     input_mat.assign(&lr_matrix);
// }
