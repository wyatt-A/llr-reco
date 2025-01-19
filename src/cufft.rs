use crate::bindings_cuda::{cufftComplex, cufftDestroy, cufftExecC2C, cufftHandle, cufftPlan3d, cufftPlanMany, cufftResult_t_CUFFT_SUCCESS, cufftType_t_CUFFT_C2C, CUFFT_FORWARD, CUFFT_INVERSE};
use crate::cuda_api::{copy_to_device, copy_to_host, cuda_free, get_device_memory_info};
use cfl::ndarray::{Array3, Array4};
use cfl::num_complex::Complex32;
use std::ffi::c_int;

enum FftDirection {
    Forward,
    Inverse,
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

    unsafe {
        let cufft_err = match fft_dir {
            FftDirection::Forward => {
                cufftExecC2C(
                    plan,
                    device_data as *mut cufftComplex,
                    device_data as *mut cufftComplex,
                    CUFFT_FORWARD as c_int,
                )
            }
            FftDirection::Inverse => {
                cufftExecC2C(
                    plan,
                    device_data as *mut cufftComplex,
                    device_data as *mut cufftComplex,
                    CUFFT_INVERSE as c_int,
                )
            }
        };

        if cufft_err != cufftResult_t_CUFFT_SUCCESS {
            cufftDestroy(plan);
            cuda_free(device_data);
            panic!("cufftExecC2C failed: {}", cufft_err);
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


pub fn fft3_batched(x: &mut Array4<Complex32>) {
    cu_fft3_batched(x, FftDirection::Forward);
}

pub fn ifft3_batched(x: &mut Array4<Complex32>) {
    cu_fft3_batched(x, FftDirection::Inverse);
}

fn cu_fft3_batched(x: &mut Array4<Complex32>, fft_dir: FftDirection) {
    // we need to determine how many batches need to be run from huge array
    let dims: [usize; 4] = x.dim().into();

    let bytes_per_vol = dims[0..3].iter().product::<usize>() * size_of::<Complex32>();
    let n_vols = dims[3];

    let total_array_bytes: usize = n_vols * bytes_per_vol;

    // get available mem on device
    let (free_mem_bytes, _) = get_device_memory_info();

    let free_mem_bytes = free_mem_bytes / 10;

    if total_array_bytes < free_mem_bytes {
        // in this case, lets try to send the entire 4-D array to device ...

        let plan = create_c2c_plan_3d_batch(dims);
        let device_data = copy_to_device(x.as_slice_memory_order_mut().unwrap());

        let direction = match fft_dir {
            FftDirection::Forward => CUFFT_FORWARD,
            FftDirection::Inverse => CUFFT_INVERSE as i32,
        };

        let cufft_err = unsafe {
            cufftExecC2C(
                plan,
                device_data as *mut cufftComplex,
                device_data as *mut cufftComplex,
                direction,
            )
        };
        if cufft_err != cufftResult_t_CUFFT_SUCCESS {
            cuda_free(device_data);
            panic!("cufftExecC2C failed: {}", cufft_err);
        }
        unsafe { cufftDestroy(plan); }
        copy_to_host(x.as_slice_memory_order_mut().unwrap(), device_data);
        cuda_free(device_data);
    } else {
        let n_vols_per_full_batch = free_mem_bytes / bytes_per_vol;
        let n_full_batches = n_vols / n_vols_per_full_batch;
        let n_remainder_vols = n_vols % n_vols_per_full_batch;

        println!("n_remainder_vols = {n_remainder_vols}");

        let mut vols_per_batch = vec![n_vols_per_full_batch; n_full_batches];
        println!("n total vols = {}", n_vols);

        if n_remainder_vols > 0 {
            vols_per_batch.push(n_remainder_vols)
        }

        println!("vols per batch = {:?}", vols_per_batch);

        let host_data = x.as_slice_memory_order_mut().unwrap();

        let elements_per_vol = dims[0..3].iter().product::<usize>();
        for (i, &n_vols_batch) in vols_per_batch.iter().enumerate().take(n_full_batches) {
            let elem_range = (i * elements_per_vol)..(i * elements_per_vol + n_vols_batch * elements_per_vol);
            let batch_data = &mut host_data[elem_range];
            let mut _dims = dims.clone();
            _dims[3] = n_vols_batch;
            let plan = create_c2c_plan_3d_batch(_dims);

            let device_data = copy_to_device(batch_data);

            let cufft_err = match fft_dir {
                FftDirection::Forward => {
                    unsafe {
                        cufftExecC2C(
                            plan,
                            device_data as *mut cufftComplex,
                            device_data as *mut cufftComplex,
                            CUFFT_FORWARD as c_int,
                        )
                    }
                }
                FftDirection::Inverse => {
                    unsafe {
                        cufftExecC2C(
                            plan,
                            device_data as *mut cufftComplex,
                            device_data as *mut cufftComplex,
                            CUFFT_INVERSE as c_int,
                        )
                    }
                }
            };

            if cufft_err != cufftResult_t_CUFFT_SUCCESS {
                cuda_free(device_data);
                panic!("cufftExecC2C failed: {}", cufft_err);
            }
            unsafe { cufftDestroy(plan); }
            copy_to_host(batch_data, device_data);
            cuda_free(device_data);
        }

        if let FftDirection::Inverse = fft_dir {
            let n = x.shape()[0..3].iter().product::<usize>() as f32;
            println!("divisor = {n}");
            x.par_mapv_inplace(|x| x / n);
        }
    }
}

fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfl::ndarray::ShapeBuilder;
    use std::time::Instant;

    #[test]
    fn test_fft3() {
        let mut x = Array3::from_elem((128, 128, 128).f(), Complex32::new(1.0, 0.0));
        let y = x.clone();
        cu_fft3(&mut x, FftDirection::Forward);
        cu_fft3(&mut x, FftDirection::Inverse);
        assert_eq!(y, x, "fft consistency failed");
        println!("fft consistency succeeded");
    }

    //cargo test --release --package llr-reco --lib cufft::tests::test_fft3_batched -- --nocapture
    #[test]
    fn test_fft3_batched() {
        let mut x = Array4::from_elem((256, 256, 256, 20).f(), Complex32::new(1.0, 0.0));
        let y = x.clone();
        let now = Instant::now();
        println!("starting fft ...");
        fft3_batched(&mut x);
        println!("starting ifft ...");
        ifft3_batched(&mut x);
        let dur = now.elapsed().as_millis();
        assert_eq!(y, x, "fft consistency failed");
        println!("fft consistency succeeded");
        println!("forward + inverse cufft took {dur} ms");
    }
}