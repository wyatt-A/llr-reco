use cfl;
//use llr_reco::cuda_api;


fn main() {

    //llr_recon_exec()

}


// fn main() {
//     //let subspace_file = Some("B:/ProjectSpace/wa41/llr-reco-test-data/dti_basis/ortho_q_basis");
//     let subspace_file = None::<&str>;
//
//     let rank = 6;
//     let subspace_llr = 6;
//     let n_iter = 1;
//     let block_size = [32, 31, 38];
//     let max_fft_vols_per_batch = 20;
//     let max_matrices_per_batch = 200;
//
//     let output_dir = "B:/ProjectSpace/wa41/llr-reco-test-data/dti/out2";
//     let cfl_dir = "B:/ProjectSpace/wa41/llr-reco-test-data/image_data/xformed/xformed_affine/cfl";
//     let out_cfl_dir = "B:/ProjectSpace/wa41/llr-reco-test-data/recon_results/result";
//     let cfl_pattern = |i: usize| format!("{cfl_dir}/vol{:03}", i);
//     let out_cfl_pattern = |i: usize| format!("{out_cfl_dir}/vol{:03}", i);
//     let sample_mask = "B:/ProjectSpace/wa41/llr-reco-test-data/image_data/sampling_masks/poisson";
//
//     let b_table = "B:/ProjectSpace/wa41/llr-reco-test-data/image_data/b_vec_info/btable";
//     let b0_null_vec_tol = 1e-6;
//
//     let output_dir = Path::new(output_dir);
//     if !output_dir.exists() {
//         fs::create_dir_all(output_dir).unwrap();
//     }
//
//     let vol_indices = (0..=130).collect::<Vec<usize>>();
//     let dataset = DtiDataSetBuilder::from_cfl_images(&vol_indices, &cfl_pattern)
//         .with_b_table(b_table, b0_null_vec_tol)
//         .with_sample_mask(sample_mask);
//
//     println!("loading volumes ...");
//     let now = Instant::now();
//     let mut fully_sampled_images = dataset.load_dti_set();
//     let dur = now.elapsed();
//     println!("data set loaded in {:.03} secs", dur.as_secs_f32());
//
//     println!("loading sample mask ...");
//     let dti_sample_mask = dataset.dti_sample_mask();
//
//     let [nx, ny, nz, nq]: [usize; 4] = fully_sampled_images.dim().into();
//
//     let subspace_matrix = subspace_file.map(|f| {
//         println!("loading subspace matrix ...");
//         let sub = cfl::to_array(f, true).expect("failed to read subspace cfl matrix")
//             .into_dimensionality::<Ix2>().expect("matrix to have 2 non-singleton dimensions");
//         let [nq_subspace, sub_rank]: [usize; 2] = sub.dim().into();
//         assert_eq!(nq, nq_subspace, "unexpected subspace dimension");
//         println!("loaded subspace matrix with rank {}", sub_rank);
//         sub
//     });
//
//     let lin_img_corrections = vec![[0., 0., 0.]; nq];
//     let lin_ksp_corrections = lin_img_corrections.clone();
//
//     println!("running forward model ...");
//     signal_model_batched(&mut fully_sampled_images, max_fft_vols_per_batch, &lin_ksp_corrections, &lin_img_corrections, ModelDirection::Forward);
//
//     println!("under-sampling data set ...");
//     under_sample_kspace(&mut fully_sampled_images, &dti_sample_mask);
//
//     let mut data_iterate = fully_sampled_images;
//
//     println!("writing test k-space");
//     let vol = data_iterate.index_axis(Axis(3), 0).to_owned();
//     cfl::from_array(output_dir.join("test_ksp"), &vol.into_dyn()).unwrap();
//
//     let vol_size = [nx, ny, nz];
//
//     let max_random_shift = vol_size;
//
//     let n_vox_per_block: usize = block_size.iter().product();
//     let matrix_size: [usize; 2] = [block_size.iter().product(), nq];
//     let grid_size = grid_dim(&vol_size, &block_size);
//     let total_blocks: usize = grid_size.iter().product();
//     let [m, n] = matrix_size;
//     assert!(m > n, "only tall matrices are supported");
//     assert!(rank <= n, "rank must not exceed n");
//     assert!(rank > 0, "rank should not be less than 1");
//
//     println!("cloning measurement data ...");
//     let measurements = data_iterate.clone();
//     let under_sampling_factor = calc_undersampling_factor(&measurements);
//     println!("under sampling factor: {:.03}", under_sampling_factor);
//
//     println!("finding linear k-space corrections ...");
//     // find linear phase shifts
//     let linear_corrections = vol_ref(&data_iterate).par_iter().map(|vol| {
//         estimate_linear_correction(vol, &vol_size)
//     }).collect::<Vec<_>>();
//
//     // go to image space in batches of 10 volumes to not overload GPU
//     println!("performing ifft on data set ...");
//     //ifft3c_chunked(&mut data_iterate, max_fft_vols_per_batch);
//     signal_model_batched(&mut data_iterate, max_fft_vols_per_batch, &linear_corrections, &lin_img_corrections, ModelDirection::Inverse);
//
//     println!("writing test volume to nii ...");
//     // write test volume to look at
//     let vol = data_iterate.index_axis(Axis(3), 0).to_owned();
//     cfl::dump_magnitude(output_dir.join("test_vol_pre"), &vol.clone().into_dyn());
//
//     println!("allocating matrix data batch ...");
//     // define the matrix batch size to be no larger than 500
//     assert!(max_matrices_per_batch <= total_blocks);
//     let mut matrices = Array3::from_elem((matrix_size[0], matrix_size[1], max_matrices_per_batch).f(), Complex32::ZERO);
//     let matrix_stride = matrix_size[0] * matrix_size[1];
//
//     let block_indices = (0..total_blocks).collect::<Vec<_>>();
//     let n_batches = (total_blocks + max_matrices_per_batch - 1) / max_matrices_per_batch;
//
//     let mut relative_iterate_error_hist = Vec::<f64>::with_capacity(n_iter);
//     let mut init_err: Option<f64> = None;
//     println!("beginning iteration loop ...");
//
//     for it in 0..n_iter {
//         println!("iteration {} of {}", it + 1, n_iter);
//
//         // generate a random shift variable for shift-invariant denoising
//         let shift = rand_shift(&max_random_shift);
//
//         println!("beginning LLR block regularization ...");
//         let now = Instant::now();
//         for (batch_id, batch) in block_indices.chunks(max_matrices_per_batch).enumerate() {
//             println!("working on matrix batch {} of {}", batch_id + 1, n_batches);
//
//             let matrix_start_idx = *batch.first().unwrap();
//             let n_matrices = *batch.last().unwrap() - matrix_start_idx + 1;
//             let _matrix_data = matrices.as_slice_memory_order_mut().unwrap(); // the entire slice
//             let matrix_data = &mut _matrix_data[0..(n_matrices * matrix_stride)]; // trimmed slice
//
//             let volumes = vol_ref(&data_iterate);
//             lift_data_set(&volumes, matrix_data, &vol_size, &block_size, &shift, matrix_start_idx, n_matrices);
//
//             if let Some(subspace_matrix) = subspace_matrix.as_ref() {
//                 let subspace = subspace_matrix.as_ref().unwrap();
//                 let [nq, sub_rank]: [usize; 2] = subspace.dim().into();
//                 assert!(subspace_llr <= sub_rank, "when a subspace is used, the llr block rank must be less than or equal to the subspace rank");
//                 let subspace_matrix = subspace.as_slice_memory_order().unwrap();
//                 let mut compressed_data = vec![Complex32::zero(); sub_rank * n_vox_per_block * n_matrices];
//                 dti_project(subspace_matrix, matrix_data, &mut compressed_data, n_matrices, nq, n_vox_per_block, sub_rank);
//                 conj_transpose_matrix(&mut compressed_data, sub_rank, n_vox_per_block, n_matrices);
//                 cu_low_rank_approx_batch(n_vox_per_block, sub_rank, subspace_llr, n_matrices, &mut compressed_data);
//                 conj_transpose_matrix(&mut compressed_data, n_vox_per_block, sub_rank, n_matrices);
//                 dti_back_project(subspace_matrix, matrix_data, &compressed_data, n_matrices, nq, n_vox_per_block, sub_rank);
//             } else {
//                 cu_low_rank_approx_batch(m, n, rank, n_matrices, matrix_data);
//             }
//
//             let mut volumes = vol_ref_mut(&mut data_iterate);
//             un_lift_data_set(&mut volumes, matrix_data, &vol_size, &block_size, &shift, matrix_start_idx, n_matrices);
//         }
//         let dur = now.elapsed();
//         println!("regularization took  {} seconds", dur.as_secs());
//
//         let vol = data_iterate.index_axis(Axis(3), 0).to_owned();
//         cfl::dump_magnitude(output_dir.join(format!("test_vol_llr_iter{:02}", it + 1)), &vol.into_dyn());
//
//         signal_model_batched(&mut data_iterate, max_fft_vols_per_batch, &linear_corrections, &lin_img_corrections, ModelDirection::Forward);
//
//         //println!("performing hard-projection ...");
//         let sum_sq_error = hard_project(&mut data_iterate, &measurements);
//         let relative_iterate_error = sum_sq_error / *init_err.get_or_insert(sum_sq_error);
//         relative_iterate_error_hist.push(relative_iterate_error);
//         println!("current error: {:.03}", relative_iterate_error);
//
//         //println!("returning to image space ...");
//         // return to image space
//         //ifft3c_chunked(&mut data_iterate, max_fft_vols_per_batch);
//         signal_model_batched(&mut data_iterate, max_fft_vols_per_batch, &linear_corrections, &lin_img_corrections, ModelDirection::Inverse);
//     }
//
//
//     println!("error history: {:?}", relative_iterate_error_hist);
//
//     // write cfl volumes out
//     println!("writing results ...");
//     let data = data_iterate.as_slice_memory_order().unwrap();
//     data.par_chunks_exact(nx * ny * nz).enumerate().for_each(|(i, vol)| {
//         cfl::write_buffer(out_cfl_pattern(i), &[nx, ny, nz], vol).unwrap();
//     })
// }


