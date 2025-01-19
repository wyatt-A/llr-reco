use crate::bindings_cuda::{cudaDeviceProp, cudaError_cudaSuccess, cudaFree, cudaGetDeviceCount, cudaGetDeviceProperties_v2, cudaMalloc, cudaMemGetInfo, cudaMemcpy, cudaMemcpyKind_cudaMemcpyDeviceToHost, cudaMemcpyKind_cudaMemcpyHostToDevice, cudaSetDevice};
use std::ffi::{c_void, CStr};
use std::os::raw::c_int;
use std::ptr;

// allocates memory on device and copies data, returning pointer to device memory
pub fn copy_to_device<T: Sized>(data: &[T]) -> *mut c_void {
    let data_bytes = data.len() * size_of::<T>();
    let mut device_pointer: *mut c_void = ptr::null_mut();
    let cuda_status = unsafe {
        cudaMalloc(&mut device_pointer as *mut *mut c_void, data_bytes)
    };
    if cuda_status != cudaError_cudaSuccess {
        unsafe { cudaFree(device_pointer); }
        panic!("cudaMalloc failed");
    }

    let host_pointer = data.as_ptr() as *const c_void;

    // copy data to device
    let cuda_status = unsafe {
        cudaMemcpy(
            device_pointer,
            host_pointer,
            data_bytes,
            cudaMemcpyKind_cudaMemcpyHostToDevice,
        )
    };

    if cuda_status != cudaError_cudaSuccess {
        unsafe { cudaFree(device_pointer); }
        panic!("cudaMemcpy failed");
    }

    device_pointer
}

// copies data from device to host based on the size of data. This will panic if the device pointer
// is null or if there is a mem overrun
pub fn copy_to_host<T: Sized>(data: &mut [T], device_pointer: *mut c_void) {
    if device_pointer.is_null() {
        panic!("device pointer must not be null");
    }
    let data_bytes = data.len() * size_of::<T>();
    let host_pointer = data.as_mut_ptr() as *mut c_void;

    let cuda_status = unsafe {
        cudaMemcpy(
            host_pointer,
            device_pointer,
            data_bytes,
            cudaMemcpyKind_cudaMemcpyDeviceToHost,
        )
    };

    if cuda_status != cudaError_cudaSuccess {
        panic!("cudaMemcpy failed");
    }
}

pub fn cuda_free(device_pointer: *mut c_void) {
    unsafe {
        cudaFree(device_pointer);
    }
}

pub fn get_device_count() -> usize {
    unsafe {
        // 1. Get the number of GPU devices
        let mut device_count: c_int = 0;
        let error_code = cudaGetDeviceCount(&mut device_count as *mut c_int);
        if error_code != cudaError_cudaSuccess {
            panic!("cudaGetDeviceCount failed with error code {}", error_code);
        } else {
            device_count as usize
        }
    }
}

pub fn get_device_name(device_index: usize) -> String {
    let idx = device_index as c_int;
    let mut props = std::mem::MaybeUninit::<cudaDeviceProp>::uninit();
    // Call cudaGetDeviceProperties(...).
    let err = unsafe { cudaGetDeviceProperties_v2(props.as_mut_ptr(), idx) };
    if err != cudaError_cudaSuccess {
        panic!("cudaGetDeviceProperties_v2 failed with error code {}", err);
    }
    let props = unsafe { props.assume_init() };
    let cstr_ptr = props.name.as_ptr();
    let device_name = unsafe { CStr::from_ptr(cstr_ptr) }
        .to_string_lossy()
        .into_owned();
    device_name
}

pub fn select_cuda_device(device_index: usize) {
    let device_index = device_index as c_int;
    unsafe {
        let err = cudaSetDevice(device_index);
        if err != cudaError_cudaSuccess {
            panic!("cudaSetDevice failed with error code {}", err);
        }
    }
}

/// returns current device memory info in bytes (free_mem,total_mem)
pub fn get_device_memory_info() -> (usize, usize) {
    let mut free_mem: usize = 0;
    let mut total_mem: usize = 0;

    // cudaMemGetInfo writes the free and total memory (in bytes)
    let err = unsafe { cudaMemGetInfo(&mut free_mem as *mut _, &mut total_mem as *mut _) };
    if err != cudaError_cudaSuccess {
        panic!("failed to get device memory info");
    }
    // Returns free and total in bytes
    (free_mem, total_mem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfl::ndarray::{Array2, ShapeBuilder};
    use cfl::num_complex::Complex32;
    #[test]
    fn test() {
        // get device count
        let device_count = get_device_count();
        if device_count == 0 {
            println!("no cuda devices found");
            return;
        }
        let device_index = device_count - 1;
        let device_name = get_device_name(device_index);
        println!("device {device_index} has name: {device_name}");
        select_cuda_device(device_index);
        println!("device {device_index} selected");
        println!("checking available memory ...");
        let (free_mem, total_mem) = get_device_memory_info();
        println!("{free_mem} bytes free of {total_mem} ({}%)", free_mem as f32 * 100. / total_mem as f32);
        println!("copying data to device ...");
        let matrix = Array2::from_shape_fn((10, 10).f(), |(i, j)| Complex32::new(i as f32, j as f32));
        let device_data = copy_to_device(matrix.as_slice_memory_order().unwrap());
        let mut target_matrix = Array2::from_elem(matrix.dim().f(), Complex32::ZERO);
        println!("copying data back to host ...");
        copy_to_host(target_matrix.as_slice_memory_order_mut().unwrap(), device_data);
        cuda_free(device_data);
        assert_eq!(matrix, target_matrix, "data consistency check failed");
        println!("data consistency check passed");
    }
}