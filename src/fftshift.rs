use cfl::ndarray::parallel::prelude::ParallelIterator;
use cfl::ndarray::parallel::prelude::{IndexedParallelIterator, IntoParallelRefIterator};
use cfl::ndarray::parallel::prelude::{IntoParallelIterator, IntoParallelRefMutIterator};
use cfl::num_complex::Complex32;
use std::f32::consts::PI;

/// specifies the direction of the pi phase shift. This has the same convention as the DFT
pub enum PhaseShiftDir {
    Forward,
    Inverse,
}

fn coord_to_col_maj_index(coords: &[i32], dims: &[usize]) -> usize {
    assert_eq!(coords.len(), dims.len(), "coordinate length and number of dimensions disagree");
    let mut idx = 0;
    let mut stride = 1;
    for (coord, dim) in coords.iter().zip(dims.iter()) {
        let c = coord.rem_euclid(*dim as i32) as usize;
        idx += c * stride;
        stride *= dim;
    }
    idx
}

fn col_maj_index_to_coord(idx: usize, dims: &[usize], coord: &mut [i32]) {
    assert_eq!(dims.len(), coord.len(), "coordinate length and number of dimensions disagree");
    let mut tmp = idx;
    for (i, &dim_size) in dims.iter().enumerate() {
        coord[i] = (tmp % dim_size) as i32;
        tmp /= dim_size;
    }
}

/// returns the frequency bin in standard DFT ordering (DC is index 0)
fn linear_index_to_frequency_bin(index: usize, n: usize) -> i32 {
    let half_n = (n as i32 + 1) / 2;
    let index_i32 = index as i32;
    index_i32 - ((index_i32 >= half_n) as i32 * n as i32)
}

#[inline]
/// returns the phase shift associated with the centered fft
fn phase_shift(index: usize, n: usize) -> f32 {
    assert!(index < n, "index out of range");
    PI * (index as f32 - (n as f32 / 2.))
}

/// performs a phase shift on a 3-D volume of complex-values for use in centered fft
pub fn phase_shift3(dims: &[usize], x: &mut [Complex32], direction: PhaseShiftDir) {
    assert!(dims.len() <= 3, "only dims up to 3 are supported");
    assert_eq!(dims.iter().product::<usize>(), x.len(), "dims must agree with the size of x");

    let mut dims3 = [1; 3];
    dims3.iter_mut().zip(dims.iter()).for_each(|(d, i)| *d = *i);

    let sign = match direction {
        PhaseShiftDir::Forward => -1.,
        PhaseShiftDir::Inverse => 1.
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

fn circshift3_col_maj<T: Copy + Send + Sync>(
    dims: &[usize],
    shift: &[i32],
    data: &mut [T],
) {
    assert_eq!(dims.len(), 3, "dims must have 3 elements");
    assert_eq!(dims.iter().product::<usize>(), data.len(),
               "dims and source size must agree");
    assert_eq!(dims.len(), shift.len(),
               "the shift must have the same number of dimensions as dims");
    let result = (0..data.len())
        .into_par_iter()
        .map(|final_idx| {
            let mut coord = [0i32; 3];
            col_maj_index_to_coord(final_idx, dims, &mut coord);
            //Reverse the shift to find the original source coordinate
            for (c, &s) in coord.iter_mut().zip(shift) {
                *c -= s;
            }
            let source_idx = coord_to_col_maj_index(&coord, dims);
            data[source_idx]
        })
        .collect::<Vec<_>>();
    data.copy_from_slice(&result);
}

/// forward fft shift of col-maj ordered array
pub fn fftshift<T: Copy + Send + Sync>(dims: &[usize], data: &mut [T]) {
    assert!(dims.len() <= 3, "greater than 3 dimensions is not supported here");
    let mut shift = [0i32; 3];
    shift.iter_mut().zip(dims.iter()).for_each(|(s, d)| *s = (*d / 2) as i32);
    circshift3_col_maj(dims, &shift, data);
}

/// inverse fft shift of col-maj ordered array
pub fn ifftshift<T: Copy + Send + Sync>(dims: &[usize], data: &mut [T]) {
    assert!(dims.len() <= 3, "greater than 3 dimensions is not supported here");
    let mut shift = [0i32; 3];
    shift.iter_mut().zip(dims.iter()).for_each(|(s, d)| *s = ((*d + 1) / 2) as i32);
    circshift3_col_maj(dims, &shift, data);
}


#[cfg(test)]
mod tests {
    use crate::fftshift::{col_maj_index_to_coord, coord_to_col_maj_index, fftshift, ifftshift, linear_index_to_frequency_bin, phase_shift3, PhaseShiftDir};
    use cfl::ndarray::{Array3, Array4, ShapeBuilder};
    use cfl::num_complex::Complex32;
    use std::time::Instant;

    #[test]
    fn test() {
        let test_coord = [0, 0, -1];
        let mut result_coord = [0, 0, 0];
        let dims = [256, 128, 256];
        let idx = coord_to_col_maj_index(&test_coord, &dims);
        col_maj_index_to_coord(idx, &dims, &mut result_coord);
        assert_eq!(result_coord, [0, 0, 255]);
    }

    #[test]
    fn test_circshift() {
        let mut a = Array3::from_shape_fn((512, 512, 512).f(), |(i, j, k)| Complex32::new((i + j + k) as f32, 0.));
        let dims = a.shape().to_vec();
        println!("starting circshift ...");
        let now = Instant::now();
        ifftshift(&dims, a.as_slice_memory_order_mut().unwrap());
        //crate::cufft::cu_fftn_batch(&dims, 1, crate::cufft::FftDirection::Forward, crate::cufft::NormalizationType::default(), a.as_slice_memory_order_mut().unwrap());
        fftshift(&dims, a.as_slice_memory_order_mut().unwrap());
        //circshift3_col_maj(&dims, &[-1, 0, 0], a.as_slice_memory_order_mut().unwrap());
        let dur = now.elapsed().as_millis();
        println!("took {} ms", dur);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_phase_shift() {
        println!("building test array ...");
        let n_vols = 1;
        let now = Instant::now();
        let mut a = Array4::from_shape_fn((788, 480, 480, n_vols).f(), |(i, j, k, l)| Complex32::new((i + j + k + l) as f32, 0.));
        let dur = now.elapsed().as_millis();
        println!("attempting to xform {} MB", (a.len() * size_of::<Complex32>()) as f64 / 2f64.powi(20));
        println!("test array built in {} ms", dur);
        let dims = a.shape().to_vec();
        println!(" starting cufft...");
        let now = Instant::now();
        phase_shift3(&dims[0..3], &mut a.as_slice_memory_order_mut().unwrap()[0..(788 * 480 * 480)], PhaseShiftDir::Forward);
        //crate::cufft::cu_fftn_batch(&dims[0..3], n_vols, crate::cufft::FftDirection::Forward, crate::cufft::NormalizationType::default(), a.as_slice_memory_order_mut().unwrap());
        //crate::cufft::cu_fftn_batch(&dims[0..3], n_vols, crate::cufft::FftDirection::Inverse, crate::cufft::NormalizationType::default(), a.as_slice_memory_order_mut().unwrap());
        phase_shift3(&dims[0..3], &mut a.as_slice_memory_order_mut().unwrap()[0..(788 * 480 * 480)], PhaseShiftDir::Inverse);
        let dur = now.elapsed().as_millis();
        println!("fft took {} ms", dur);
        //println!("{:?}", a);
    }

    #[test]
    fn test_linear_index_to_frequency_bin() {
        let n = 1024 * 1024;
        let mut result = vec![0; n];
        let now = Instant::now();
        result.iter_mut().enumerate().for_each(|(i, x)| {
            *x = linear_index_to_frequency_bin(i, n)
        });
        let dur = now.elapsed().as_millis();
        println!("{} - {}", result.first().unwrap(), result.last().unwrap());
        println!("took {dur} ms");
    }
}