use crate::cuda::bindings::{cuComplex, cublasCreate_v2, cublasCscal_v2, cublasDestroy_v2, cublasHandle_t, cublasStatus_t_CUBLAS_STATUS_SUCCESS, cufftComplex, cufftDestroy, cufftExecC2C, cufftHandle, cufftPlan3d, cufftPlanMany, cufftResult_t_CUFFT_SUCCESS, cufftType_t_CUFFT_C2C, CUFFT_FORWARD, CUFFT_INVERSE};
use crate::cuda::cuda_api::{copy_to_device, copy_to_host, cuda_free};
use cfl::ndarray::parallel::prelude::{IntoParallelRefMutIterator, ParallelIterator};
use cfl::ndarray::Array3;
use cfl::num_complex::Complex32;
use std::ffi::c_int;
use std::ptr::null_mut;

// I need to investigate the use of cuTensor to perform centered fft optimally
// the initial plan is to store the pre-computed phase values in an array on the
// device and use cuTensor elementwise multiplication to perform the shifting
// instead of reordering the data for improved cache friendliness


pub(crate) enum FftDirection {
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


fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfl::ndarray::ShapeBuilder;
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
}