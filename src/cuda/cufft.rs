use crate::cuda::bindings::{cuComplex, cublasCreate_v2, cublasCscal_v2, cublasDestroy_v2, cublasHandle_t, cublasStatus_t_CUBLAS_STATUS_SUCCESS, cufftComplex, cufftDestroy, cufftExecC2C, cufftHandle, cufftPlan3d, cufftPlanMany, cufftResult_t_CUFFT_SUCCESS, cufftType_t_CUFFT_C2C, CUFFT_FORWARD, CUFFT_INVERSE};
use crate::cuda::cuda_api::{copy_to_device, copy_to_host, cuda_free};
use cfl::ndarray::parallel::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use cfl::ndarray::{s, Array3, Array4, ArrayViewMut, Dim};
use cfl::num_complex::Complex32;
use rayon::slice::ParallelSliceMut;
use std::f32::consts::PI;
use std::ffi::c_int;
use std::ptr::null_mut;

#[derive(Debug, Clone, Copy)]
pub enum FftDirection {
    Forward,
    Inverse,
}

pub(crate) enum NormalizationType {
    None,
    Inverse,
    Unitary,
}

impl Default for NormalizationType {
    fn default() -> Self {
        NormalizationType::Inverse
    }
}


pub fn fft3(x: &mut Array3<Complex32>) {
    cu_fft3(x, FftDirection::Forward);
}

pub fn ifft3(x: &mut Array3<Complex32>) {
    cu_fft3(x, FftDirection::Inverse);
}

/// creates a plan, returning the plan handle
fn create_c2c_plan_3d_batch(array_size_col_maj: [usize; 4]) -> cufftHandle {
    let mut plan: cufftHandle = 0;

    let rank = 3 as c_int;

    let mut dims = [
        array_size_col_maj[2] as c_int,
        array_size_col_maj[1] as c_int,
        array_size_col_maj[0] as c_int,
    ];

    let inembed = dims.as_mut_ptr();
    let onembed = dims.as_mut_ptr();

    let i_stride = 1;
    let i_dist = array_size_col_maj[0..3].iter().product::<usize>() as c_int;
    let batch = array_size_col_maj[3] as c_int;

    let cufft_err = unsafe {
        cufftPlanMany(
            &mut plan as *mut cufftHandle,
            rank,
            dims.as_mut_ptr(),   // the dimensions array
            inembed,             // inembed
            i_stride,            // istride
            i_dist,              // idist
            onembed,             // onembed
            i_stride,            // ostride
            i_dist,              // odist
            cufftType_t_CUFFT_C2C,
            batch,
        )
    };
    if cufft_err != cufftResult_t_CUFFT_SUCCESS {
        panic!("cufftPlanMany failed: {}", cufft_err);
    }
    plan
}

/// create a batched c2c fft plan for contiguous data arranged in col-major layout, where the batch
/// dimension varies the slowest
fn plan_c2c_batch(dims: &[usize], batches: usize) -> cufftHandle {
    let dist = dims.iter().product::<usize>();
    let rank = dims.len() as c_int;
    let batches = batches as c_int;
    let mut dims = dims.iter().rev().map(|&x| x as c_int).collect::<Vec<_>>();
    let dims_ptr = dims.as_mut_ptr();

    let inembed = null_mut() as *mut c_int;
    let onembed = null_mut() as *mut c_int;
    let istride = 1 as c_int;
    let ostride = 1 as c_int;
    let idist = dist as c_int;
    let odist = dist as c_int;

    let mut plan: cufftHandle = 0;

    let cufft_err = unsafe {
        cufftPlanMany(
            &mut plan as *mut cufftHandle,
            rank,
            dims_ptr,   // the dimensions array
            inembed,             // inembed
            istride,            // istride
            idist,              // idist
            onembed,             // onembed
            ostride,            // ostride
            odist,              // odist
            cufftType_t_CUFFT_C2C,
            batches,
        )
    };

    if cufft_err != cufftResult_t_CUFFT_SUCCESS {
        panic!("cufftPlanMany failed: {}", cufft_err);
    }

    plan
}

/// perform batched n-d fft on single-precision complex inputs (c2c). Data is assumed to be contiguous
/// and in column-major order, with the batch dimension varying the slowest, and the first fft
/// dimension varying the fastest.
pub(crate) fn cu_fftn_batch(dims: &[usize], batches: usize, fft_direction: FftDirection, norm: NormalizationType, data: &mut [Complex32]) {
    assert_eq!(dims.iter().product::<usize>() * batches, data.len(), "data layout and data length inconsistency");

    let plan = plan_c2c_batch(dims, batches);


    let device_data = copy_to_device(data);
    let cufft_flag = match fft_direction {
        FftDirection::Forward => CUFFT_FORWARD as c_int,
        FftDirection::Inverse => CUFFT_INVERSE as c_int,
    };

    unsafe {
        let cufft_result = cufftExecC2C(
            plan,
            device_data as *mut cufftComplex,
            device_data as *mut cufftComplex,
            cufft_flag,
        );

        if cufft_result != cufftResult_t_CUFFT_SUCCESS {
            cufftDestroy(plan);
            cuda_free(device_data);
            panic!("cufftExecC2C failed: {}", cufft_result);
        }
    }


    let normalization = match norm {
        NormalizationType::Inverse => {
            if let FftDirection::Inverse = fft_direction {
                Some((1. / dims.iter().product::<usize>() as f64) as f32)
            } else {
                None
            }
        }
        NormalizationType::Unitary => {
            Some((1. / (dims.iter().product::<usize>() as f64).sqrt()) as f32)
        }
        NormalizationType::None => None,
    };


    if let Some(scale_factor) = normalization {
        scale_complex_device_data(device_data as *mut cufftComplex, data.len(), scale_factor)
    }

    copy_to_host(data, device_data);
    cuda_free(device_data);
}

/// scale a large array of complex data already on the device by some scale factor. You must provide
/// the number of array elements. This uses cublas routines under the hood.
fn scale_complex_device_data(device_data: *mut cufftComplex, n: usize, scalar: f32) {
    assert!(!device_data.is_null(), "device data cannot be null!");

    let mut cublas_handle: cublasHandle_t = null_mut();
    unsafe {
        let status = cublasCreate_v2(&mut cublas_handle);
        if status != cublasStatus_t_CUBLAS_STATUS_SUCCESS {
            panic!("cublasCreate_v2 failed");
        }
    }

    let alpha = cuComplex { x: scalar, y: 0. };

    unsafe {
        let status = cublasCscal_v2(
            cublas_handle,
            n as c_int,
            &alpha,
            device_data,
            1,
        );
        if status != cublasStatus_t_CUBLAS_STATUS_SUCCESS {
            cublasDestroy_v2(cublas_handle);
            panic!("cublasCscal_v2 failed");
        }
        cublasDestroy_v2(cublas_handle);
    }
}

fn cu_fft3(x: &mut Array3<Complex32>, fft_dir: FftDirection) {
    let dims: [usize; 3] = x.dim().into();

    // we are assuming that the array is in col-maj order
    let device_data = copy_to_device(x.as_slice_memory_order().unwrap());

    let mut plan: cufftHandle = 0;
    unsafe {
        // cufftPlan3d(&mut plan, nx, ny, nz, CUFFT_C2C)

        // we have to reverse the dims to be compliant with row-maj ordering
        let nx = dims[2] as c_int;
        let ny = dims[1] as c_int;
        let nz = dims[0] as c_int;

        let cufft_err = cufftPlan3d(&mut plan, nx, ny, nz, cufftType_t_CUFFT_C2C);
        if cufft_err != cufftResult_t_CUFFT_SUCCESS {
            cuda_free(device_data);
            panic!("cufftPlan3d failed: {}", cufft_err);
        }
    }

    let cufft_flag = match fft_dir {
        FftDirection::Forward => CUFFT_FORWARD as c_int,
        FftDirection::Inverse => CUFFT_INVERSE as c_int,
    };

    unsafe {
        let cufft_result = cufftExecC2C(
            plan,
            device_data as *mut cufftComplex,
            device_data as *mut cufftComplex,
            cufft_flag,
        );

        if cufft_result != cufftResult_t_CUFFT_SUCCESS {
            cufftDestroy(plan);
            cuda_free(device_data);
            panic!("cufftExecC2C failed: {}", cufft_result);
        }
    }

    copy_to_host(x.as_slice_memory_order_mut().unwrap(), device_data);

    // account for fft scaling similar to matlab impl
    if let FftDirection::Inverse = fft_dir {
        let n = x.len() as f32;
        x.par_mapv_inplace(|x| x / n);
    }

    unsafe {
        cufftDestroy(plan);
        cuda_free(device_data);
    }
}

/// performs a phase shift on a 3-D volume of complex-values for use in centered fft
pub fn phase_shift3(dims: &[usize], x: &mut [Complex32], direction: FftDirection) {
    assert!(dims.len() <= 3, "only dims up to 3 are supported");
    assert_eq!(dims.iter().product::<usize>(), x.len(), "dims must agree with the size of x");

    let mut dims3 = [1; 3];
    dims3.iter_mut().zip(dims.iter()).for_each(|(d, i)| *d = *i);

    let sign = match direction {
        FftDirection::Forward => -1.,
        FftDirection::Inverse => 1.
    };

    x.par_iter_mut().enumerate().for_each(|(idx, value)| {
        let iz = idx / (dims3[0] * dims3[1]);
        let rem = idx % (dims3[0] * dims3[1]);
        let iy = rem / dims3[0];
        let ix = rem % dims3[0];

        let z_shift = phase_shift(iz, dims3[2]);
        let y_shift = phase_shift(iy, dims3[1]);
        let x_shift = phase_shift(ix, dims3[0]);
        let total_shift = sign * (x_shift + y_shift + z_shift);

        *value = *value * Complex32::from_polar(1., total_shift);
    });
}

#[inline]
/// returns the phase shift associated with the centered fft
fn phase_shift(index: usize, n: usize) -> f32 {
    assert!(index < n, "index out of range");
    PI * (index as f32 - (n as f32 / 2.))
}

/// unitary forward centered fast fourier transform for CUDA devices
pub fn fft3c(x: &mut Array3<Complex32>) {
    let dims: [usize; 3] = x.dim().into();
    phase_shift3(&dims, x.as_slice_memory_order_mut().unwrap(), FftDirection::Inverse);
    cu_fftn_batch(&dims, 1, FftDirection::Forward, NormalizationType::Unitary, x.as_slice_memory_order_mut().unwrap());
    phase_shift3(&dims, x.as_slice_memory_order_mut().unwrap(), FftDirection::Forward);
}

/// unitary inverse centered fast fourier transform for CUDA devices
pub fn ifft3c(x: &mut Array3<Complex32>) {
    let dims: [usize; 3] = x.dim().into();
    phase_shift3(&dims, x.as_slice_memory_order_mut().unwrap(), FftDirection::Inverse);
    cu_fftn_batch(&dims, 1, FftDirection::Inverse, NormalizationType::Unitary, x.as_slice_memory_order_mut().unwrap());
    phase_shift3(&dims, x.as_slice_memory_order_mut().unwrap(), FftDirection::Forward);
}

fn _fft3c_batched(x: &mut Array4<Complex32>, direction: FftDirection) {
    let dims: [usize; 4] = x.dim().into();
    let batch_size = *dims.last().unwrap();
    let vol_dims = &dims[0..3];
    let vol_stride: usize = vol_dims.iter().product();
    let data = x.as_slice_memory_order_mut().unwrap();
    data.par_chunks_exact_mut(vol_stride).for_each(|vol| {
        phase_shift3(vol_dims, vol, FftDirection::Inverse);
    });
    cu_fftn_batch(&vol_dims, batch_size, direction, NormalizationType::Unitary, data);
    data.par_chunks_exact_mut(vol_stride).for_each(|vol| {
        phase_shift3(vol_dims, vol, FftDirection::Forward);
    });
}

fn fft3c_batched_view(x: &mut ArrayViewMut<Complex32, Dim<[usize; 4]>>, direction: FftDirection) {
    let dims: [usize; 4] = x.dim().into();
    let batch_size = *dims.last().unwrap();
    let vol_dims = &dims[0..3];
    let vol_stride: usize = vol_dims.iter().product();
    let data = x.as_slice_memory_order_mut().unwrap();
    data.par_chunks_exact_mut(vol_stride).for_each(|vol| {
        phase_shift3(vol_dims, vol, FftDirection::Inverse);
    });
    cu_fftn_batch(&vol_dims, batch_size, direction, NormalizationType::Unitary, data);
    data.par_chunks_exact_mut(vol_stride).for_each(|vol| {
        phase_shift3(vol_dims, vol, FftDirection::Forward);
    });
}

pub fn fft3c_chunked(x: &mut Array4<Complex32>, vols_per_batch: usize) {
    _fft3c_chunked(x, FftDirection::Forward, vols_per_batch);
}

pub fn ifft3c_chunked(x: &mut Array4<Complex32>, vols_per_batch: usize) {
    _fft3c_chunked(x, FftDirection::Inverse, vols_per_batch);
}

fn _fft3c_chunked(x: &mut Array4<Complex32>, dir: FftDirection, vols_per_batch: usize) {
    let dims: [usize; 4] = x.dim().into();
    let n_vols = dims[3];
    (0..n_vols).collect::<Vec<usize>>().chunks(vols_per_batch).for_each(|chunk| {
        let index_start = *chunk.first().unwrap();
        let index_end = *chunk.last().unwrap();
        let mut v = x.slice_mut(s![..,..,..,index_start..=index_end]);
        fft3c_batched_view(&mut v, dir);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfl::ndarray::{stack, Axis, Ix3, ShapeBuilder};
    use cfl::ndarray_linalg::Scalar;
    use cfl::ndarray_stats::QuantileExt;
    use cfl::to_array;
    use std::time::Instant;

    //cargo test --features cuda --release --package llr-reco --lib cufft::tests::test_fft3 -- --nocapture
    #[test]
    fn test_fft3() {
        let mut x = Array3::from_elem((128, 128, 128).f(), Complex32::new(1.0, 0.0));
        let y = x.clone();
        cu_fft3(&mut x, FftDirection::Forward);
        cu_fft3(&mut x, FftDirection::Inverse);
        assert_eq!(y, x, "fft consistency failed");
        println!("fft consistency succeeded");
    }

    // cargo test --features cuda --release --package llr-reco --lib cufft::tests::test_fft3_2 -- --nocapture
    #[test]
    fn test_fft3_2() {
        //let mut x = to_array("single_vol_x", true).unwrap().into_dimensionality::<Ix3>().unwrap();
        let mut x = to_array("single_vol_x", true).unwrap();
        let dims = x.shape().to_vec();
        //let y = x.clone();
        //cu_fft3(&mut x, FftDirection::Forward);
        //cu_fft3(&mut x, FftDirection::Inverse);

        cu_fftn_batch(&dims, 1, FftDirection::Forward, NormalizationType::default(), x.as_slice_memory_order_mut().unwrap());
        //cu_fft_batch(&dims, 1, FftDirection::Inverse, x.as_slice_memory_order_mut().unwrap());

        let y = to_array("single_vol_y", true).unwrap();

        println!("{:?}", (x - y).map(|x| x.abs()).max());

        //assert_abs_diff_eq!(x.map(|x|x.re),y.map(|y|y.re),epsilon=1e-1);
    }


    #[test]
    fn test_fft3_multi() {
        println!("loading test data ...");
        let mut x = to_array("multi_vol_x", true).unwrap();
        let y = x.clone();
        //let y = to_array("multi_vol_y", true).unwrap();
        let dims = x.shape()[0..3].to_vec();
        let batches = *x.shape().last().unwrap();
        println!("dims: {:?}", dims);
        println!("batches: {}", batches);
        println!("running xforms ...");
        let now = Instant::now();
        cu_fftn_batch(&dims, batches, FftDirection::Forward, NormalizationType::default(), x.as_slice_memory_order_mut().unwrap());
        cu_fftn_batch(&dims, batches, FftDirection::Inverse, NormalizationType::default(), x.as_slice_memory_order_mut().unwrap());
        let dur = now.elapsed().as_millis();
        println!("{:?}", (x - y).map(|x| x.abs()).max());
        println!("both ffts took {dur} ms");
    }

    #[test]
    fn test_fft3_shepp() {
        println!("loading test data ...");
        let mut ksp = to_array("shepp_512", true).unwrap().into_dimensionality::<Ix3>().unwrap();
        let orig = ksp.clone();
        let now = Instant::now();
        ifft3c(&mut ksp);
        fft3c(&mut ksp);
        let dur = now.elapsed().as_millis();
        println!("two ffts took {dur} ms");

        let orig = orig.map(|x| x.abs());
        let ksp = ksp.map(|x| x.abs());

        let mean_error = (&orig - &ksp).map(|x| x.abs()).mean().unwrap();
        let mean_value = orig.mean().unwrap();

        println!("mean error: {mean_error}");
        println!("mean value: {mean_value}");
        println!("relative error: {:.2e}", mean_error / mean_value);
    }

    #[test]
    fn test_fft3_batched_shepp() {
        println!("loading test data ...");
        let mut ksp = to_array("shepp_512", true).unwrap().into_dimensionality::<Ix3>().unwrap();

        let x1 = ksp.clone();

        let batch_size = 10;
        let views = (0..batch_size).map(|_| x1.view()).collect::<Vec<_>>();

        let mut x: Array4<Complex32> = stack(Axis(3), &views).unwrap();
        println!("{:?}", x.dim());
        let now = Instant::now();
        _fft3c_batched(&mut x, FftDirection::Inverse);
        let dur = now.elapsed().as_millis();
        println!("batched fft to {dur} ms");
        //println!("converting to dyn ...");
        //let x = x.into_dyn();
        //println!("writing out ...");
        //cfl::dump_magnitude("img", &x);
        //cfl::dump_phase("imgp", &x);
    }
}