use cfl;
use cfl::ndarray::{Array3, Array4, Axis, ShapeBuilder};
use cfl::num_complex::Complex32;
use cfl::num_traits::identities::Zero;
use llr_reco::block::{grid_dim, lift_data_set, un_lift_data_set};
use llr_reco::cuda::low_rank::cu_low_rank_approx_batch;
use llr_reco::signal_model::{estimate_linear_correction, signal_model_batched, ModelDirection};
use rand::Rng;
//use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator};
use rayon::prelude::*;
use std::fs;
use std::path::Path;
use std::time::Instant;
//use llr_reco::cuda_api;

fn main() {
    let data_set_dir = Path::new("B:/ProjectSpace/wa41/llr-reco-test-data/dti");
    let test_outputs = Path::new("B:/ProjectSpace/wa41/llr-reco-test-data/dti/out");
    let filename_fn = |i: usize| format!("vol{:03}", i);

    let rank = 24;
    let n_vols_to_process = 120;
    let n_total_vols = 131;
    let n_iter = 50;
    let vol_size = [512, 284, 228];
    let data_set_size = [512, 284, 228, n_vols_to_process];
    let block_size = [32, 31, 38];
    let max_fft_vols_per_batch = 20;
    let max_matrices_per_batch = 200;
    let max_random_shift = vol_size;
    let vols_indices_to_omit = [0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130];

    if !test_outputs.exists() {
        fs::create_dir_all(test_outputs).unwrap();
    }

    let matrix_size: [usize; 2] = [block_size.iter().product(), n_vols_to_process];
    let grid_size = grid_dim(&vol_size, &block_size);
    let total_blocks: usize = grid_size.iter().product();
    let [m, n] = matrix_size;
    assert!(m > n, "only tall matrices are supported");
    assert!(rank <= n, "rank must not exceed n");
    assert!(rank > 0, "rank should not be less than 1");

    let total_bytes = data_set_size.iter().product::<usize>() * size_of::<Complex32>();
    let vol_stride: usize = vol_size.iter().product();
    println!("allocating data iterate ({:.03} GB) ...", total_bytes as f64 / 2f64.powi(30));
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

    println!("cloning measurement data ...");
    let measurements = data_iterate.clone();
    let under_sampling_factor = calc_undersampling_factor(&measurements);
    println!("under sampling factor: {:.03}", under_sampling_factor);


    println!("finding linear corrections ...");
    // find linear phase shifts
    let linear_corrections = vol_ref(&data_iterate).par_iter().map(|vol| {
        estimate_linear_correction(vol, &vol_size)
    }).collect::<Vec<_>>();
    let lin_corr_ref = linear_corrections.iter().map(|x| x).collect::<Vec<_>>();

    let vol = data_iterate.index_axis(Axis(3), 0).to_owned();
    cfl::dump_magnitude(test_outputs.join("test_vol_pre_k"), &vol.into_dyn());

    println!("linear corrections: {:?}", linear_corrections);

    // go to image space in batches of 10 volumes to not overload GPU
    println!("performing ifft on data set ...");
    //ifft3c_chunked(&mut data_iterate, max_fft_vols_per_batch);
    signal_model_batched(&mut data_iterate, max_fft_vols_per_batch, &lin_corr_ref, ModelDirection::Inverse);

    println!("writing test volume to nii ...");
    // write test volume to look at
    let vol = data_iterate.index_axis(Axis(3), 0).to_owned();
    cfl::dump_magnitude(test_outputs.join("test_vol_pre"), &vol.clone().into_dyn());
    cfl::dump_phase(test_outputs.join("test_volp_pre"), &vol.into_dyn());
    // build a vector of contiguous volume slices from the 4-D array
    //let volumes = dataset.axis_iter(Axis(3)).map(|vol| vol.as_slice_memory_order().unwrap()).collect::<Vec<_>>();

    println!("allocating matrix data batch ...");
    // define the matrix batch size to be no larger than 500
    assert!(max_matrices_per_batch <= total_blocks);
    let mut matrices = Array3::from_elem((matrix_size[0], matrix_size[1], max_matrices_per_batch).f(), Complex32::ZERO);
    let matix_data = matrices.as_slice_memory_order_mut().unwrap();
    let block_indices = (0..total_blocks).collect::<Vec<_>>();
    let n_batches = (total_blocks + max_matrices_per_batch - 1) / max_matrices_per_batch;


    let mut relative_iterate_error_hist = Vec::<f64>::with_capacity(n_iter);
    let mut init_err: Option<f64> = None;
    println!("beginning iteration loop ...");

    for it in 0..n_iter {
        println!("iteration {} of {}", it + 1, n_iter);
        // generate a random shift variable for shift-invariant denoising
        let shift = rand_shift(&max_random_shift);
        println!("using random shift: {:?}", shift);

        for (batch_id, batch) in block_indices.chunks(max_matrices_per_batch).enumerate() {
            println!("working on matrix batch {} of {}", batch_id + 1, n_batches);

            let matrix_start_idx = *batch.first().unwrap();
            let n_matrices = *batch.last().unwrap() - matrix_start_idx + 1;
            //println!("processing {} matrices ...", n_matrices);

            //println!("lifting image data into matrices ...");
            // load the matrix data from image volume data in batches
            //let volumes = dataset.axis_iter(Axis(3)).map(|vol| vol.as_slice_memory_order().unwrap()).collect::<Vec<_>>();
            println!("preparing matrix data for GPU xfer ...");
            let volumes = vol_ref(&data_iterate);
            lift_data_set(&volumes, matix_data, &vol_size, &block_size, &shift, matrix_start_idx, n_matrices);

            println!("performing low-rank approximation ...");
            cu_low_rank_approx_batch(m, n, rank, max_matrices_per_batch, matix_data);

            //println!("writing image data back to volumes ...");
            // write matrix data back into image volumes
            //let mut volumes = dataset.axis_iter_mut(Axis(3)).map(|mut vol| vol.as_slice_memory_order_mut().unwrap()).collect::<Vec<_>>();
            println!("inserting matrix data ...");
            let mut volumes = vol_ref_mut(&mut data_iterate);
            un_lift_data_set(&mut volumes, matix_data, &vol_size, &block_size, &shift, matrix_start_idx, n_matrices);
        }

        let vol = data_iterate.index_axis(Axis(3), 0).to_owned();
        cfl::dump_magnitude(test_outputs.join(format!("test_vold_iter{:02}", it)), &vol.into_dyn());

        println!("entering k-space ...");
        // return to k-space
        //fft3c_chunked(&mut data_iterate, max_fft_vols_per_batch);
        signal_model_batched(&mut data_iterate, max_fft_vols_per_batch, &lin_corr_ref, ModelDirection::Forward);

        println!("performing hard-projection ...");
        let sum_sq_error = hard_project(&mut data_iterate, &measurements);
        let relative_iterate_error = sum_sq_error / *init_err.get_or_insert(sum_sq_error);
        relative_iterate_error_hist.push(relative_iterate_error);
        println!("current error: {:.03}", relative_iterate_error);

        println!("returning to image space ...");
        // return to image space
        //ifft3c_chunked(&mut data_iterate, max_fft_vols_per_batch);
        signal_model_batched(&mut data_iterate, max_fft_vols_per_batch, &lin_corr_ref, ModelDirection::Inverse);

        //println!("writing test volume to nii ...");
        // write test volume to look at
        let vol = data_iterate.index_axis(Axis(3), 0).to_owned();
        cfl::dump_magnitude(test_outputs.join(format!("test_vol_iter{:02}", it)), &vol.into_dyn());
    }
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