use cfl;
use cfl::ndarray::{Array3, Array4, Axis, Ix2, ShapeBuilder};
use cfl::num_complex::Complex32;
use cfl::num_traits::identities::Zero;
use llr_reco::block::{grid_dim, lift_data_set, un_lift_data_set};
use llr_reco::cuda::low_rank::cu_low_rank_approx_batch;
use llr_reco::data_import::load_and_undersample;
use llr_reco::dti_subspace::{conj_transpose_matrix, dti_back_project, dti_project};
use llr_reco::signal_model::{estimate_linear_correction, signal_model_batched, ModelDirection};
use rand::Rng;
//use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator};
use rayon::prelude::*;
use std::fs;
use std::path::Path;
use std::time::Instant;
//use llr_reco::cuda_api;

fn main() {

    //let data_set_dir = Path::new("B:/ProjectSpace/wa41/llr-reco-test-data/dti");
    let test_outputs = Path::new("B:/ProjectSpace/wa41/llr-reco-test-data/dti/out");
    //let filename_fn = |i: usize| format!("vol{:03}", i);
    // let affine_filename_fn = Some(
    //     |i: usize| format!("B:/ProjectSpace/wa41/llr-reco-test-data/dti/resources/translation_affines/{:03}_affine.txt", i)
    // );
    let affine_filename_fn = None::<fn(usize) -> String>;
    let subspace_file = Some("B:/ProjectSpace/wa41/llr-reco-test-data/dti/resources/dti_basis/ortho_q_basis");
    //let subspace_file = None::<&str>;

    let rank = 24;
    let subspace_llr = 6;
    let n_iter = 100;
    let max_subspace_iter = n_iter;
    let block_size = [32, 31, 38];
    let max_fft_vols_per_batch = 20;
    let max_matrices_per_batch = 200;

    if !test_outputs.exists() {
        fs::create_dir_all(test_outputs).unwrap();
    }

    let (_, mut data_iterate, _) = load_and_undersample(
        "B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32/nifti/xformed_affine/stack4d.nii",
        "B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32/bvec_120.txt",
        "B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32/msk_vol_01",
    );

    println!("writing test k-space");
    let vol = data_iterate.index_axis(Axis(3), 0).to_owned();
    cfl::from_array(test_outputs.join("test_ksp"), &vol.into_dyn()).unwrap();

    let [nx, ny, nz, n_vols_to_process]: [usize; 4] = data_iterate.dim().into();
    let vol_size = [nx, ny, nz];

    let max_random_shift = vol_size;

    assert!(max_subspace_iter <= n_iter, "number of subspace iteration cannot exceed total iterations");

    let n_vox_per_block: usize = block_size.iter().product();
    let matrix_size: [usize; 2] = [block_size.iter().product(), n_vols_to_process];
    let grid_size = grid_dim(&vol_size, &block_size);
    let total_blocks: usize = grid_size.iter().product();
    let [m, n] = matrix_size;
    assert!(m > n, "only tall matrices are supported");
    assert!(rank <= n, "rank must not exceed n");
    assert!(rank > 0, "rank should not be less than 1");

    let subspace_matrix = subspace_file.map(|f| {
        println!("loading subspace matrix ...");
        let sub = cfl::to_array(f, true).expect("failed to read subspace cfl matrix")
            .into_dimensionality::<Ix2>().expect("matrix to have 2 non-singleton dimensions");
        let [nq, sub_rank]: [usize; 2] = sub.dim().into();
        assert_eq!(nq, n_vols_to_process, "unexpected subspace dimension");
        println!("loaded subspace matrix with rank {}", sub_rank);
        sub
    });

    // let total_bytes = data_set_size.iter().product::<usize>() * size_of::<Complex32>();
    // let vol_stride: usize = vol_size.iter().product();
    // println!("allocating data iterate ({:.03} GB) ...", total_bytes as f64 / 2f64.powi(30));
    // let mut data_iterate = Array4::<Complex32>::from_elem(data_set_size.f(), Complex32::new(0.0, 0.0));
    // let data = data_iterate.as_slice_memory_order_mut().unwrap();
    //
    // let vols_to_read = (0..n_total_vols).filter(|i| !vols_indices_to_omit.contains(&i)).collect::<Vec<_>>();
    // println!("reading data set volumes: {:?}", vols_to_read);
    // let now = Instant::now();
    // data.par_chunks_exact_mut(vol_stride).zip(&vols_to_read).for_each(|(vol, &i)| {
    //     let vol_name = filename_fn(i);
    //     println!("reading vol {} ...", i);
    //     let x = cfl::to_array(data_set_dir.join(vol_name), true);
    //     vol.copy_from_slice(x.unwrap().as_slice_memory_order().unwrap());
    // });
    // let dur = now.elapsed();
    // println!("data set read in {} sec", dur.as_secs());


    println!("cloning measurement data ...");
    let measurements = data_iterate.clone();
    let under_sampling_factor = calc_undersampling_factor(&measurements);
    println!("under sampling factor: {:.03}", under_sampling_factor);

    println!("finding linear k-space corrections ...");
    // find linear phase shifts
    let linear_corrections = vol_ref(&data_iterate).par_iter().map(|vol| {
        estimate_linear_correction(vol, &vol_size)
    }).collect::<Vec<_>>();

    // // load translation data from ANTS if supplied ...
    // let lin_img_corr = if let Some(affine_files) = &affine_filename_fn {
    //     println!("loading affine translation coefficients ...");
    //     read_affine_translation(&vols_to_read, affine_files, -1.)
    // } else {
    //     vec![[0f32; 3]; vols_to_read.len()]
    // };
    let lin_img_corr = vec![[0f32; 3]; n_vols_to_process];
    //let vol = data_iterate.index_axis(Axis(3), 0).to_owned();
    //cfl::dump_magnitude(test_outputs.join("test_vol_pre_k"), &vol.into_dyn());

    //println!("linear corrections: {:?}", linear_corrections);

    // go to image space in batches of 10 volumes to not overload GPU
    println!("performing ifft on data set ...");
    //ifft3c_chunked(&mut data_iterate, max_fft_vols_per_batch);
    signal_model_batched(&mut data_iterate, max_fft_vols_per_batch, &linear_corrections, &lin_img_corr, ModelDirection::Inverse);

    println!("writing test volume to nii ...");
    // write test volume to look at
    let vol = data_iterate.index_axis(Axis(3), 0).to_owned();
    cfl::dump_magnitude(test_outputs.join("test_vol_pre"), &vol.clone().into_dyn());
    //cfl::dump_phase(test_outputs.join("test_volp_pre"), &vol.into_dyn());
    // build a vector of contiguous volume slices from the 4-D array
    //let volumes = dataset.axis_iter(Axis(3)).map(|vol| vol.as_slice_memory_order().unwrap()).collect::<Vec<_>>();

    println!("allocating matrix data batch ...");
    // define the matrix batch size to be no larger than 500
    assert!(max_matrices_per_batch <= total_blocks);
    let mut matrices = Array3::from_elem((matrix_size[0], matrix_size[1], max_matrices_per_batch).f(), Complex32::ZERO);
    let matrix_stride = matrix_size[0] * matrix_size[1];

    let block_indices = (0..total_blocks).collect::<Vec<_>>();
    let n_batches = (total_blocks + max_matrices_per_batch - 1) / max_matrices_per_batch;

    let mut relative_iterate_error_hist = Vec::<f64>::with_capacity(n_iter);
    let mut init_err: Option<f64> = None;
    println!("beginning iteration loop ...");

    for it in 0..n_iter {
        println!("iteration {} of {}", it + 1, n_iter);

        // generate a random shift variable for shift-invariant denoising
        let shift = rand_shift(&max_random_shift);
        //println!("using random shift: {:?}", shift);

        println!("beginning LLR block regularization ...");
        let now = Instant::now();
        for (batch_id, batch) in block_indices.chunks(max_matrices_per_batch).enumerate() {
            println!("working on matrix batch {} of {}", batch_id + 1, n_batches);

            let matrix_start_idx = *batch.first().unwrap();
            let n_matrices = *batch.last().unwrap() - matrix_start_idx + 1;
            let _matrix_data = matrices.as_slice_memory_order_mut().unwrap(); // the entire slice
            let matrix_data = &mut _matrix_data[0..(n_matrices * matrix_stride)]; // trimmed slice
            //println!("processing {} matrices ...", n_matrices);

            //println!("lifting image data into matrices ...");
            // load the matrix data from image volume data in batches
            //let volumes = dataset.axis_iter(Axis(3)).map(|vol| vol.as_slice_memory_order().unwrap()).collect::<Vec<_>>();
            //println!("lifting voxel data into matrix ...");
            let volumes = vol_ref(&data_iterate);
            lift_data_set(&volumes, matrix_data, &vol_size, &block_size, &shift, matrix_start_idx, n_matrices);

            if subspace_matrix.is_some() && it < max_subspace_iter {
                let subspace = subspace_matrix.as_ref().unwrap();
                let [nq, sub_rank]: [usize; 2] = subspace.dim().into();
                assert!(subspace_llr <= sub_rank, "when a subspace is used, the llr block rank must be less than or equal to the subspace rank");
                let subspace_matrix = subspace.as_slice_memory_order().unwrap();
                let mut compressed_data = vec![Complex32::zero(); sub_rank * n_vox_per_block * n_matrices];
                //println!("running projection ...");
                dti_project(subspace_matrix, matrix_data, &mut compressed_data, n_matrices, nq, n_vox_per_block, sub_rank);
                //println!("xposing ...");
                conj_transpose_matrix(&mut compressed_data, sub_rank, n_vox_per_block, n_matrices);
                //println!("running low-rank ...");
                cu_low_rank_approx_batch(n_vox_per_block, sub_rank, subspace_llr, n_matrices, &mut compressed_data);
                //println!("xposing ...");
                conj_transpose_matrix(&mut compressed_data, n_vox_per_block, sub_rank, n_matrices);
                //println!("running back-projection ...");
                dti_back_project(subspace_matrix, matrix_data, &compressed_data, n_matrices, nq, n_vox_per_block, sub_rank);
            } else {
                //println!("performing low-rank approximation ...");
                cu_low_rank_approx_batch(m, n, rank, n_matrices, matrix_data);
            }

            // project block data into subspace to perform spatio-angular compression

            //println!("writing image data back to volumes ...");
            // write matrix data back into image volumes
            //let mut volumes = dataset.axis_iter_mut(Axis(3)).map(|mut vol| vol.as_slice_memory_order_mut().unwrap()).collect::<Vec<_>>();
            //println!("inserting voxel data from matrix ...");
            let mut volumes = vol_ref_mut(&mut data_iterate);
            un_lift_data_set(&mut volumes, matrix_data, &vol_size, &block_size, &shift, matrix_start_idx, n_matrices);
        }
        let dur = now.elapsed();
        println!("regularization took  {} seconds", dur.as_secs());

        let vol = data_iterate.index_axis(Axis(3), 0).to_owned();
        cfl::dump_magnitude(test_outputs.join(format!("test_vol_llr_iter{:02}", it)), &vol.into_dyn());

        //println!("entering k-space ...");
        // return to k-space
        //fft3c_chunked(&mut data_iterate, max_fft_vols_per_batch);
        signal_model_batched(&mut data_iterate, max_fft_vols_per_batch, &linear_corrections, &lin_img_corr, ModelDirection::Forward);

        //println!("performing hard-projection ...");
        let sum_sq_error = hard_project(&mut data_iterate, &measurements);
        let relative_iterate_error = sum_sq_error / *init_err.get_or_insert(sum_sq_error);
        relative_iterate_error_hist.push(relative_iterate_error);
        println!("current error: {:.03}", relative_iterate_error);

        //println!("returning to image space ...");
        // return to image space
        //ifft3c_chunked(&mut data_iterate, max_fft_vols_per_batch);
        signal_model_batched(&mut data_iterate, max_fft_vols_per_batch, &linear_corrections, &lin_img_corr, ModelDirection::Inverse);

        //println!("writing test volume to nii ...");
        // write test volume to look at
        let vol = data_iterate.index_axis(Axis(3), 0).to_owned();
        cfl::dump_magnitude(test_outputs.join(format!("test_vol_iter{:02}", it)), &vol.into_dyn());
    }


    println!("error history: {:?}", relative_iterate_error_hist);
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

fn calc_undersampling_factor(measurements: &Array4<Complex32>) -> f64 {
    let x = measurements.as_slice_memory_order().unwrap();
    let non_zero = x.par_iter().map(|x| {
        (!x.is_zero()) as usize
    }).sum::<usize>() as f64;
    x.len() as f64 / non_zero
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
