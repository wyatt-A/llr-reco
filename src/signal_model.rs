// Forward and inverse MRI signal model
// the purpose of this module is to transform MRI signals to images and back while accounting
// for phase errors and coil sensitivities

use crate::cuda::cufft::{cu_fftn_batch, max_coord, phase_shift3, phase_shift3_sub_vox, FftDirection, NormalizationType};
use cfl::ndarray::Array4;
use cfl::num_complex::Complex32;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
pub enum ModelDirection {
    Forward,
    Inverse,
}

impl Into<FftDirection> for ModelDirection {
    fn into(self) -> FftDirection {
        match self {
            Self::Forward => FftDirection::Forward,
            Self::Inverse => FftDirection::Inverse,
        }
    }
}

/// Performs the forward and inverse signal operator for a batch of volumes. Linear phase corrections
/// can be specified to correct for sub-voxel linear phase errors for both forward and inverse operators.
/// The corrections should remain constant regardless of operator sign. This is set up to be as efficient
/// as possible, preventing the need for storing entire phase/coil sensitivity maps in memory. Phase corrections
/// are instead applied on-the-fly in one shot to keep memory overhead to a minimum. If you need non-linear
/// phase corrections or magnitude maps for coil sensitivities, this function is not for you.
pub fn signal_model(x: &mut [Complex32], image_size: &[usize; 3], batch_size: usize, linear_corrections: &[&[f32; 3]], direction: ModelDirection) {
    assert_eq!(batch_size, linear_corrections.len(), "the number of linear corrections must be equal to batch size");
    let [nx, ny, nz] = *image_size;
    let vol_stride = nx * ny * nz;
    assert!(x.len() >= nx * ny * nz * batch_size, "data buffer not large enough");

    // if run in forward mode, the image phase correction is first reversed back to native state
    // prior to inverse fft. If run in inverse mode, a standard FOV/2 shift is applied to perform
    // the centered inverse fft.
    match direction {
        ModelDirection::Forward => {
            x.par_chunks_exact_mut(vol_stride).enumerate().for_each(|(idx, vol)| {
                let shift = calc_shift(&linear_corrections[idx], image_size);
                phase_shift3_sub_vox(image_size, vol, &shift, FftDirection::Inverse);
            });
        }
        ModelDirection::Inverse => {
            x.par_chunks_exact_mut(vol_stride).for_each(|vol| {
                phase_shift3(image_size, vol, FftDirection::Inverse);
            });
        }
    }

    // standard fft call
    cu_fftn_batch(image_size, batch_size, direction.into(), NormalizationType::Unitary, x);

    // if run in forward mode, the k-space phase is corrected only accounting for the FOV/2 shift
    // if run in inverse mode, the phase corrections are applied directly to image space
    match direction {
        ModelDirection::Forward => {
            x.par_chunks_exact_mut(vol_stride).for_each(|vol| {
                phase_shift3(image_size, vol, FftDirection::Forward);
            });
        }
        ModelDirection::Inverse => {
            x.par_chunks_exact_mut(vol_stride).enumerate().for_each(|(idx, vol)| {
                let shift = calc_shift(&linear_corrections[idx], image_size);
                phase_shift3_sub_vox(image_size, vol, &shift, FftDirection::Forward);
            });
        }
    }
}

/// Performs the forward and inverse signal operator on a large 4-D data set in batches to avoid
/// memory-related challenges on the GPU. See [signal_model] for more usage insights
pub fn signal_model_batched(x: &mut Array4<Complex32>, batch_size: usize, linear_corrections: &[&[f32; 3]], model_direction: ModelDirection) {
    let [nx, ny, nz, nq]: [usize; 4] = x.dim().into();
    assert_eq!(linear_corrections.len(), nq, "the number of linear corrections must be equal to number of volumes");
    let vol_stride = nx * ny * nz;
    let batch_stride = vol_stride * batch_size;
    let mut data = x.as_slice_memory_order_mut().unwrap();
    for (chunk, lin_corr) in data.chunks_mut(batch_stride).zip(linear_corrections.chunks(batch_size)) {
        let n_vols = chunk.len() / vol_stride;
        assert_eq!(n_vols, lin_corr.len(), "expected number of vols and correction to be equal");
        signal_model(chunk, &[nx, ny, nz], n_vols, lin_corr, model_direction);
    }
}

/// Estimates the linear phase correction for a 3D k-space volume, returning shift coefficients
/// in units of samples. Shift coefficients of 0 indicate that the DC k-space sample if properly
/// centered in the volume to produce no linear phase variations in the image after the centered
/// inverse FFT. Note that fractional values are supported.
pub fn estimate_linear_correction(ksp: &[Complex32], vol_size: &[usize; 3]) -> [f32; 3] {
    let [ix, iy, iz] = max_coord(vol_size, ksp);
    let ix = ix as i32 - (vol_size[0] as i32 + 2 - 1) / 2;
    let iy = iy as i32 - (vol_size[1] as i32 + 2 - 1) / 2;
    let iz = iz as i32 - (vol_size[2] as i32 + 2 - 1) / 2;
    [ix as f32, iy as f32, iz as f32]
}

/// Calculates the shift relative to the first sample in the k-space volume k(0,0,0). This is used
/// to adjust the standard FFT where the DC sample is assumed to be the first sample.
fn calc_shift(linear_correction: &[f32; 3], image_size: &[usize; 3]) -> [f32; 3] {
    let [mut sx, mut sy, mut sz] = *linear_correction;
    let [mut nx, mut ny, mut nz] = *image_size;
    sx += 0.5 * nx as f32;
    sy += 0.5 * ny as f32;
    sz += 0.5 * nz as f32;
    [sx, sy, sz]
}


mod tests {
    use crate::signal_model::{signal_model, ModelDirection};
    use cfl::ndarray::Ix3;

    #[test]
    fn test_inverse() {
        println!("loading test volume ...");
        let x = cfl::to_array("C:/Users/wa41/llr-reco-test-data/vol00", true).unwrap();

        cfl::dump_magnitude("C:/Users/wa41/llr-reco-test-data/out/ks", &x);

        let mut x = x.into_dimensionality::<Ix3>().unwrap();

        println!("running xforms ...");

        let image_size: [usize; 3] = x.dim().into();
        let x_dat = x.as_slice_memory_order_mut().unwrap();
        let shift = [1., 0., 0.];

        signal_model(x_dat, &image_size, 1, &[&shift], ModelDirection::Inverse);
        signal_model(x_dat, &image_size, 1, &[&shift], ModelDirection::Forward);

        //inverse(x_dat, &image_size, 1, &[&shift]);

        //forward(x_dat, &image_size, 1, &[&[0., 0., 0.]]);


        println!("writing nii ...");
        cfl::dump_magnitude("C:/Users/wa41/llr-reco-test-data/out/kso", &x.into_dyn());
    }
}