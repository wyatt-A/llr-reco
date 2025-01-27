use crate::cuda::bindings::{cudaDeviceProp, cudaError_cudaSuccess, cudaFree, cudaGetDeviceCount, cudaGetDeviceProperties_v2, cudaMalloc, cudaMemGetInfo, cudaMemcpy, cudaMemcpyKind_cudaMemcpyDeviceToHost, cudaMemcpyKind_cudaMemcpyHostToDevice, cudaMemset, cudaSetDevice};
use std::ffi::{c_void, CStr};
use std::os::raw::c_int;
use std::ptr;

/// Allocates memory for some number of elements of a specified type, setting all entries to 0.
/// Panics if calls to cudaMalloc or cudaMemset fail
pub fn cuda_malloc_memset<T: Sized>(n_elements: usize) -> *mut c_void {
    // total number of bytes to request
    let data_bytes = n_elements * size_of::<T>();
    // null pointer to data
    let mut device_pointer: *mut c_void = ptr::null_mut();

    // check if we are exceeding available memory ...
    // some mem allocations may still work even if there is not enough free mem, especially with
    // modern device drivers. We will warn the user anyway...
    let (free_mem, _) = get_device_memory_info();
    if free_mem < data_bytes {
        println!(
            "WARNING!: requested allocation ({:.3e} bytes) exceeds free device memory ({:.3e} bytes)",
            data_bytes as f64, free_mem as f64);
    }

    // call to memory request
    let cuda_status = unsafe {
        cudaMalloc(&mut device_pointer as *mut *mut c_void, data_bytes)
    };

    // crash the program if the allocation is unsuccessful
    if cuda_status != cudaError_cudaSuccess {
        unsafe { cudaFree(device_pointer); }
        panic!("cudaMalloc failed with error code: {cuda_status}");
    }

    let cuda_status = unsafe {
        cudaMemset(device_pointer, 0, data_bytes)
    };

    // crash the program if memset was unsuccessful
    if cuda_status != cudaError_cudaSuccess {
        unsafe { cudaFree(device_pointer); }
        panic!("cudaMemset failed with error code: {cuda_status}");
    }

    device_pointer
}


/// Allocates memory on device and copies data, returning raw pointer to device memory. It is up to the user to properly cast the pointer when using it.
pub fn copy_to_device<T: Sized>(data: &[T]) -> *mut c_void {

    // total number of bytes to request
    let data_bytes = data.len() * size_of::<T>();
    // null pointer to data
    let mut device_pointer: *mut c_void = ptr::null_mut();

    // check if we are exceeding available memory ...
    // some mem allocations may still work even if there is not enough free mem, especially with
    // modern device drivers. We will warn the user anyway...
    let (free_mem, _) = get_device_memory_info();
    if free_mem < data_bytes {
        println!(
            "WARNING!: requested allocation ({:.3e} bytes) exceeds free device memory ({:.3e} bytes)",
            data_bytes as f64, free_mem as f64);
    }

    // call to memory request
    let cuda_status = unsafe {
        cudaMalloc(&mut device_pointer as *mut *mut c_void, data_bytes)
    };

    // crash the program if the allocation is unsuccessful
    if cuda_status != cudaError_cudaSuccess {
        unsafe { cudaFree(device_pointer); }
        panic!("cudaMalloc failed with error code: {cuda_status}");
    }

    // copy data from host to device
    let host_pointer = data.as_ptr() as *const c_void;
    let cuda_status = unsafe {
        cudaMemcpy(
            device_pointer,
            host_pointer,
            data_bytes,
            cudaMemcpyKind_cudaMemcpyHostToDevice,
        )
    };

    // crash if memory copy is unsuccessful
    if cuda_status != cudaError_cudaSuccess {
        unsafe { cudaFree(device_pointer); }
        panic!("cudaMemcpy failed with error code: {cuda_status}");
    }

    // return the raw pointer to device memory. It is up to the user to properly
    // cast this pointer to the actual type when used
    device_pointer
}

/// Copy data from the device back to the host. It is up to the user to ensure that the device
/// pointer has the correct type and memory allocated to it based on the data slice. This function
/// will panic if the device pointer is invalid (NULL) or if the cudaMemcpy call fails
pub fn copy_to_host<T: Sized>(data: &mut [T], device_pointer: *mut c_void) {
    if device_pointer.is_null() {
        panic!("device pointer must not be null");
    }
    // get data size from slice len and type
    let data_bytes = data.len() * size_of::<T>();
    let host_pointer = data.as_mut_ptr() as *mut c_void;

    // cudaMemcpy call
    let cuda_status = unsafe {
        cudaMemcpy(
            host_pointer,
            device_pointer,
            data_bytes,
            cudaMemcpyKind_cudaMemcpyDeviceToHost,
        )
    };
    // crash if unsuccessful
    if cuda_status != cudaError_cudaSuccess {
        panic!("cudaMemcpy failed with error code: {cuda_status}");
    }
}

/// Free device memory. Panics if device pointer is invalid or cudaFree returns an error.
pub fn cuda_free(device_pointer: *mut c_void) {
    if device_pointer.is_null() {
        panic!("device pointer must not be null");
    }
    unsafe {
        let stat = cudaFree(device_pointer);
        if stat != cudaError_cudaSuccess {
            panic!("cudaFree failed: {}", stat);
        }
    }
}

/// Get the number of devices available from the host. Panics if cudaGetDeviceCount fails.
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
#[cfg(feature = "cuda")]
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