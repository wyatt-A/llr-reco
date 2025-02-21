use crate::cuda::bindings::{cuComplex, cudaError_t};
use crate::cuda::cuda_api::{copy_to_device, copy_to_host, cuda_free, cuda_malloc_memset};
use cfl::num_complex::Complex32;
use dwt::dwt3::sub_band_size;
use std::ffi::{c_float, c_int, c_uint};
use std::time::Instant;

#[cfg(test)]
mod tests {
    use super::*;
    use cfl::ndarray::{Array3, ShapeBuilder};
    use cfl::ndarray_linalg::Scalar;
    use cfl::num_complex::Complex32;
    use dwt::dwt3::sub_band_size;
    use dwt::wavelet::{Wavelet, WaveletFilter, WaveletType};
    use std::time::Instant;

    #[test]
    fn test_dwt3_all() {
        let vol_size = [512, 512, 512];
        let w: Wavelet<f32> = Wavelet::new(WaveletType::Daubechies2);

        let mut x = Array3::<Complex32>::from_shape_fn(vol_size.f(), |(i, j, k)| Complex32::new((i + j + k) as f32, 0.0));
        let x_original = x.clone();
        let rand_shift = [0, 0, 0];

        cu_dwt_denoise(
            x.as_slice_memory_order_mut().unwrap(),
            &vol_size,
            w.lo_d(), w.hi_d(),
            w.lo_r(), w.hi_r(),
            &rand_shift,
        );

        x.par_mapv_inplace(|x| Complex32::new(x.re.round(), x.im.round()));
        assert_eq!(x, x_original);
    }

    #[test]
    fn test_dwt3_axis() {
        for axis in 0..3 {
            let vol_size = [512, 512, 512];
            let w: Wavelet<f32> = Wavelet::new(WaveletType::Daubechies2);

            let mut x = Array3::<Complex32>::from_shape_fn(vol_size.f(), |(i, j, k)| Complex32::new((i + j + k) as f32, 0.0));
            let x_original = x.clone();

            let n_coeffs_x = sub_band_size(vol_size[axis], w.filt_len());
            let result_size_axis = 2 * n_coeffs_x;

            let mut decomp_size = vol_size.clone();
            decomp_size[axis] = result_size_axis;

            let mut r = Array3::<Complex32>::zeros(decomp_size.f());

            let shift = [0, 0, 0];

            let now = Instant::now();
            //println!("running forward xform ...");
            cu_dwt3_axis(
                x.as_slice_memory_order().unwrap(),
                r.as_slice_memory_order_mut().unwrap(),
                &vol_size,
                &decomp_size,
                axis,
                w.lo_d(),
                w.hi_d(),
                &shift,
            );

            cu_idwt3_axis(
                x.as_slice_memory_order_mut().unwrap(),
                r.as_slice_memory_order().unwrap(),
                &vol_size,
                &decomp_size,
                axis,
                w.lo_r(),
                w.hi_r(),
                &shift,
            );

            let elapsed = now.elapsed();
            println!("forward - inverse xform took {} ms", elapsed.as_millis());

            let diff = &x - &x_original;
            let max_error = *diff.map(|x| x.abs()).iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

            x.mapv_inplace(|x| Complex32::new(x.re.round(), x.im.round()));

            println!("max error {:?}", max_error);
            assert_eq!(x, x_original);
        }
    }
}


extern "C" {
    fn dwt3_axis(vol_data: *const cuComplex, decomp: *mut cuComplex, vol_size: *const c_uint, decomp_size: *const c_uint, axis: c_uint, filter_len: c_uint, lo_d: *const c_float, hi_d: *const c_float, rand_shift: *const c_int) -> cudaError_t;
    fn idwt3_axis(vol_data: *mut cuComplex, decomp: *const cuComplex, vol_size: *const c_uint, decomp_size: *const c_uint, axis: c_uint, filter_len: c_uint, lo_r: *const c_float, hi_r: *const c_float, rand_shift: *const c_int) -> cudaError_t;
}

pub fn cu_dwt_denoise(vol_data: &mut [Complex32], vol_size: &[usize; 3], lo_d: &[f32], hi_d: &[f32], lo_r: &[f32], hi_r: &[f32], rand_shift: &[i32; 3]) {

    // multi-axis decomposition

    let n_samples = vol_size.iter().product::<usize>();
    assert_eq!(vol_data.len(), n_samples);

    assert_eq!(lo_d.len(), hi_d.len());
    assert_eq!(lo_r.len(), hi_r.len());
    assert_eq!(lo_d.len(), lo_r.len());
    let f_len = lo_d.len();

    let now = Instant::now();

    let d_vol_data = copy_to_device(vol_data);

    let n_coeffs_x = sub_band_size(vol_size[0], f_len);
    let n_coeffs_y = sub_band_size(vol_size[1], f_len);
    let n_coeffs_z = sub_band_size(vol_size[2], f_len);

    let tmp1_size = [2 * n_coeffs_x, vol_size[1], vol_size[2]];
    let tmp2_size = [2 * n_coeffs_x, 2 * n_coeffs_y, vol_size[2]];
    let decomp_size = [2 * n_coeffs_x, 2 * n_coeffs_y, 2 * n_coeffs_z];

    let d_tmp1 = cuda_malloc_memset::<Complex32>(tmp1_size.iter().product());
    let d_tmp2 = cuda_malloc_memset::<Complex32>(tmp2_size.iter().product());
    let d_decomp = cuda_malloc_memset::<Complex32>(decomp_size.iter().product());

    let vol_size: Vec<c_uint> = vol_size.iter().map(|&x| x as c_uint).collect();
    let tmp1_size: Vec<c_uint> = tmp1_size.iter().map(|&x| x as c_uint).collect();
    let tmp2_size: Vec<c_uint> = tmp2_size.iter().map(|&x| x as c_uint).collect();
    let decomp_size: Vec<c_uint> = decomp_size.iter().map(|&x| x as c_uint).collect();

    let rand_shift: Vec<c_int> = rand_shift.iter().map(|&x| x as c_int).collect();

    let d_lo_d = copy_to_device(lo_d);
    let d_hi_d = copy_to_device(hi_d);
    let d_lo_r = copy_to_device(lo_r);
    let d_hi_r = copy_to_device(hi_r);
    let d_rand_shift = copy_to_device(&rand_shift);

    let to_device = now.elapsed();

    let now = Instant::now();
    let err = unsafe {
        let mut stat = 0;

        stat = dwt3_axis(
            d_vol_data as *const cuComplex,
            d_tmp1 as *mut cuComplex,
            vol_size.as_ptr() as *const c_uint,
            tmp1_size.as_ptr() as *const c_uint,
            0 as c_uint,
            f_len as c_uint,
            d_lo_d as *const c_float,
            d_hi_d as *const c_float,
            d_rand_shift as *const c_int,
        );

        stat = dwt3_axis(
            d_tmp1 as *const cuComplex,
            d_tmp2 as *mut cuComplex,
            tmp1_size.as_ptr() as *const c_uint,
            tmp2_size.as_ptr() as *const c_uint,
            1 as c_uint,
            f_len as c_uint,
            d_lo_d as *const c_float,
            d_hi_d as *const c_float,
            d_rand_shift as *const c_int,
        );

        stat = dwt3_axis(
            d_tmp2 as *const cuComplex,
            d_decomp as *mut cuComplex,
            tmp2_size.as_ptr() as *const c_uint,
            decomp_size.as_ptr() as *const c_uint,
            2 as c_uint,
            f_len as c_uint,
            d_lo_d as *const c_float,
            d_hi_d as *const c_float,
            d_rand_shift as *const c_int,
        );

        stat = idwt3_axis(
            d_tmp2 as *mut cuComplex,
            d_decomp as *const cuComplex,
            tmp2_size.as_ptr() as *const c_uint,
            decomp_size.as_ptr() as *const c_uint,
            2 as c_uint,
            f_len as c_uint,
            d_lo_r as *const c_float,
            d_hi_r as *const c_float,
            d_rand_shift as *const c_int,
        );

        stat = idwt3_axis(
            d_tmp1 as *mut cuComplex,
            d_tmp2 as *const cuComplex,
            tmp1_size.as_ptr() as *const c_uint,
            tmp2_size.as_ptr() as *const c_uint,
            1 as c_uint,
            f_len as c_uint,
            d_lo_r as *const c_float,
            d_hi_r as *const c_float,
            d_rand_shift as *const c_int,
        );

        stat = idwt3_axis(
            d_vol_data as *mut cuComplex,
            d_tmp1 as *const cuComplex,
            vol_size.as_ptr() as *const c_uint,
            tmp1_size.as_ptr() as *const c_uint,
            0 as c_uint,
            f_len as c_uint,
            d_lo_r as *const c_float,
            d_hi_r as *const c_float,
            d_rand_shift as *const c_int,
        );
        stat
    };

    cuda_free(d_tmp1);
    cuda_free(d_tmp2);
    cuda_free(d_decomp);
    cuda_free(d_lo_d);
    cuda_free(d_hi_d);
    cuda_free(d_lo_r);
    cuda_free(d_hi_r);
    cuda_free(d_rand_shift);

    if err != 0 {
        cuda_free(d_vol_data);
        panic!("cu dwt failed: {}", err);
    }

    let kernel_elapsed = now.elapsed();

    let now = Instant::now();
    copy_to_host(vol_data, d_vol_data);
    cuda_free(d_vol_data);
    let to_host = now.elapsed();

    println!("to device: {} ms", to_device.as_millis());
    println!("kernels: {} ms", kernel_elapsed.as_millis());
    println!("to host: {} ms", to_host.as_millis());
}


pub fn cu_dwt3_axis(vol_data: &[Complex32], decomp: &mut [Complex32], vol_size: &[usize; 3], decomp_size: &[usize; 3], axis: usize, lo_d: &[f32], hi_d: &[f32], rand_shift: &[i32; 3]) {
    assert_eq!(lo_d.len(), hi_d.len());
    let f_len = lo_d.len() as c_uint;

    let n_decomp_elements = decomp_size.iter().product::<usize>();
    assert_eq!(n_decomp_elements, decomp.len());

    let n_vol_elements = vol_size.iter().product::<usize>();
    assert_eq!(n_vol_elements, vol_data.len());

    let vol_data_d = copy_to_device(vol_data);
    let decomp_d = cuda_malloc_memset::<Complex32>(n_decomp_elements);

    let vol_size: Vec<c_uint> = vol_size.iter().map(|&x| x as c_uint).collect();
    let decomp_size: Vec<c_uint> = decomp_size.iter().map(|&x| x as c_uint).collect();
    let axis = axis as c_uint;
    let rand_shift: Vec<c_int> = rand_shift.iter().map(|&x| x as c_int).collect();

    let rand_shift_d = copy_to_device(&rand_shift);
    let lo_d_d = copy_to_device(&lo_d);
    let hi_d_d = copy_to_device(&hi_d);

    let err = unsafe {
        dwt3_axis(
            vol_data_d as *const cuComplex,
            decomp_d as *mut cuComplex,
            vol_size.as_ptr() as *const c_uint,
            decomp_size.as_ptr() as *const c_uint,
            axis,
            f_len,
            lo_d_d as *const c_float,
            hi_d_d as *const c_float,
            rand_shift_d as *const c_int,
        )
    };

    cuda_free(vol_data_d);
    cuda_free(rand_shift_d);
    cuda_free(lo_d_d);
    cuda_free(hi_d_d);

    if err != 0 {
        cuda_free(decomp_d);
        panic!("cu_dwt3_axis failed: {}", err);
    }

    copy_to_host(decomp, decomp_d);
    cuda_free(decomp_d);
}

pub fn cu_idwt3_axis(vol_data: &mut [Complex32], decomp: &[Complex32], vol_size: &[usize; 3], decomp_size: &[usize; 3], axis: usize, lo_r: &[f32], hi_r: &[f32], rand_shift: &[i32; 3]) {
    assert_eq!(lo_r.len(), hi_r.len());
    let f_len = lo_r.len() as c_uint;

    let n_decomp_elements = decomp_size.iter().product::<usize>();
    assert_eq!(n_decomp_elements, decomp.len());

    let n_vol_elements = vol_size.iter().product::<usize>();
    assert_eq!(n_vol_elements, vol_data.len());

    let decomp_d = copy_to_device(decomp);
    let vol_data_d = cuda_malloc_memset::<Complex32>(n_decomp_elements);

    let vol_size: Vec<c_uint> = vol_size.iter().map(|&x| x as c_uint).collect();
    let decomp_size: Vec<c_uint> = decomp_size.iter().map(|&x| x as c_uint).collect();
    let axis = axis as c_uint;
    let rand_shift: Vec<c_int> = rand_shift.iter().map(|&x| x as c_int).collect();

    let rand_shift_d = copy_to_device(&rand_shift);
    let lo_r_d = copy_to_device(&lo_r);
    let hi_r_d = copy_to_device(&hi_r);

    let now = Instant::now();
    let err = unsafe {
        idwt3_axis(
            vol_data_d as *mut cuComplex,
            decomp_d as *const cuComplex,
            vol_size.as_ptr() as *const c_uint,
            decomp_size.as_ptr() as *const c_uint,
            axis,
            f_len,
            lo_r_d as *const c_float,
            hi_r_d as *const c_float,
            rand_shift_d as *const c_int,
        )
    };
    let elapsed = now.elapsed();
    println!("inverse kernel took {} ms", elapsed.as_millis());

    cuda_free(decomp_d);
    cuda_free(rand_shift_d);
    cuda_free(lo_r_d);
    cuda_free(hi_r_d);

    if err != 0 {
        cuda_free(vol_data_d);
        panic!("cu_idwt3_axis failed: {}", err);
    }

    copy_to_host(vol_data, vol_data_d);
    cuda_free(vol_data_d);
}