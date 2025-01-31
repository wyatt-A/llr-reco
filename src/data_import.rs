use byteorder::ByteOrder;
use cfl::num_complex::Complex32;
use cfl::num_traits;
use cfl::num_traits::{ToPrimitive, Zero};
use num_traits::FromBytes;
use rayon::prelude::*;
use std::fs::File;
use std::io::Read;
use std::path::Path;


pub fn read_civm_i16_volume(file_pattern: impl AsRef<Path>, vol_dims: &[usize; 3], complex_volume_data: &mut [Complex32]) {
    let [nx, ny, nz] = *vol_dims;
    let slice_stride = nx * ny;
    assert_eq!(nx * ny * nz, complex_volume_data.len(), "unexpected complex buffer length");
    let p = glob::glob(&file_pattern.as_ref().display().to_string().as_str()).expect("Failed to read glob pattern");
    let mut paths = p.map(|path| path.unwrap().to_path_buf()).collect::<Vec<_>>();
    assert_eq!(paths.len(), nz, "z dim and number of files found must match");
    paths.sort();
    complex_volume_data.par_chunks_exact_mut(slice_stride).zip(paths.par_iter()).for_each(|(s, p)| {
        read_i16_be_to_cfl(p, s);
    });
}


fn read_i16_be_to_cfl(file: impl AsRef<Path>, complex_values: &mut [Complex32]) {
    let mut byte_buffer = Vec::<u8>::with_capacity(complex_values.len() * size_of::<Complex32>());
    let mut f = File::open(file);
    f.unwrap().read_to_end(&mut byte_buffer).expect("failed to read file");
    let n_elems = byte_buffer.len() / size_of::<i16>();
    let mut dst = vec![0i16; n_elems];

    byteorder::BigEndian::read_i16_into(&byte_buffer, &mut dst);

    assert_eq!(complex_values.len(), dst.len(), "Mismatched buffer sizes");
    complex_values.par_iter_mut().zip(dst.par_iter_mut()).for_each(|(x, y)| {
        *x = Complex32::new(y.to_f32().expect("f32 conversion failure"), 0.);
    });
}

#[cfg(test)]
mod tests {
    use crate::data_import::{read_civm_i16_volume, read_i16_be_to_cfl};
    use crate::signal_model::{signal_model_batched, ModelDirection};
    use cfl::dump_magnitude;
    use cfl::ndarray::{Array2, Array3, Array4, ArrayD, ShapeBuilder};
    use cfl::num_complex::Complex32;
    use cfl::num_traits::Zero;
    use rayon::prelude::*;
    use std::path::Path;

    #[test]
    fn read_civm_raw() {
        let file = "B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32/N51200_m000/N51200_m000t9imx.0142.raw";

        let mut slice = Array2::<Complex32>::zeros((512, 284).f());
        let data = slice.as_slice_memory_order_mut().unwrap();

        read_i16_be_to_cfl(file, data);

        dump_magnitude("B:/ProjectSpace/wa41/llr-reco-test-data/test_dump", &slice.into_dyn());
    }

    #[test]
    fn read_volume() {
        let pattern = "B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32/N51200_m000/N51200_m000t9imx.*.raw";
        let vol_dim: [usize; 3] = [512, 284, 228];
        let mut vol = Array3::zeros(vol_dim.f());
        read_civm_i16_volume(pattern, &vol_dim, vol.as_slice_memory_order_mut().unwrap());
        dump_magnitude("B:/ProjectSpace/wa41/llr-reco-test-data/test_dump", &vol.into_dyn());
    }


    #[test]
    fn compile_cfls() {
        let n_vols = 131;
        let vol_dim: [usize; 3] = [512, 284, 228];
        let mut vol_buffer = ArrayD::zeros(vol_dim.as_slice().f());

        let output_data = "B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32/cfl";

        for i in 0..n_vols {
            println!("working on vol {i} ...");
            let vol_pattern = format!("B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32/N51200_m{:03}/N51200_m{:03}t9imx.*.raw", i, i);
            read_civm_i16_volume(vol_pattern, &vol_dim, vol_buffer.as_slice_memory_order_mut().unwrap());
            let out_vol = Path::new(output_data).join(format!("vol{:03}", i));
            cfl::from_array(out_vol, &vol_buffer).unwrap()
        }
    }


    #[test]
    fn scale_cfls() {
        let n_vols = 131;
        let vol_dim: [usize; 3] = [512, 284, 228];

        let data_dir = "B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32/cfl";

        for i in 0..n_vols {
            println!("working on vol {i} ...");
            let vol_name = Path::new(data_dir).join(format!("vol{:03}", i));
            let mut vol = cfl::to_array(&vol_name, true).unwrap();
            vol.par_mapv_inplace(|x| x / i16::MAX as f32);
            cfl::from_array(vol_name, &vol).unwrap()
        }
    }

    #[test]
    fn apply_fft() {
        let n_vols = 131;
        let vol_dim: [usize; 3] = [512, 284, 228];
        let vol_stride = vol_dim.iter().product::<usize>();

        let data_dir = "B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32/cfl";

        let out_data = "B:/ProjectSpace/wa41/llr-reco-test-data/dti";


        let data_set_size = [vol_dim[0], vol_dim[1], vol_dim[2], n_vols];

        println!("allocating mem for {:.03} GB ...", (vol_stride * n_vols * size_of::<Complex32>()) as f64 / 2f64.powi(30));
        let mut data_set = Array4::<Complex32>::zeros(data_set_size.f());


        let lin_corr = vec![[0f32, 0f32, 0f32]; n_vols];
        let lin_corr_ref = lin_corr.iter().map(|x| x).collect::<Vec<_>>();

        println!("reading data ...");

        let data_buffer = data_set.as_slice_memory_order_mut().unwrap();
        data_buffer.par_chunks_exact_mut(vol_stride).enumerate().for_each(|(i, x)| {
            println!("working on vol {i} ...");
            let vol_name = Path::new(data_dir).join(format!("vol{:03}", i));
            let vol = cfl::to_array(&vol_name, true).unwrap();
            x.copy_from_slice(vol.as_slice_memory_order().unwrap());
        });


        println!("performing fft ...");
        signal_model_batched(&mut data_set, 20, &lin_corr_ref, ModelDirection::Forward);

        let data_buffer = data_set.as_slice_memory_order_mut().unwrap();
        println!("writing data ...");
        data_buffer.par_chunks_exact(vol_stride).enumerate().for_each(|(i, x)| {
            println!("working on vol {i} ...");
            let vol_name = Path::new(out_data).join(format!("vol{:03}", i));
            let mut vol_buff = ArrayD::zeros(vol_dim.as_slice().f());
            vol_buff.as_slice_memory_order_mut().unwrap().copy_from_slice(x);
            let vol = cfl::from_array(&vol_name, &vol_buff).unwrap();
        });
    }

    #[test]
    fn apply_undersampling() {
        let n_vols = 131;
        let vol_dim: [usize; 3] = [512, 284, 228];
        let pe_stride = vol_dim[1..3].iter().product::<usize>();
        let vol_stride = vol_dim.iter().product::<usize>();

        let data_dir = "B:/ProjectSpace/wa41/llr-reco-test-data/dti";

        let msk = cfl::to_array("B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32/msk_vol", true).unwrap();
        let msk_buffer = msk.as_slice_memory_order().unwrap();

        let data_set_size = [vol_dim[0], vol_dim[1], vol_dim[2], n_vols];

        println!("allocating mem for {:.03} GB ...", (vol_stride * n_vols * size_of::<Complex32>()) as f64 / 2f64.powi(30));
        let mut data_set = Array4::<Complex32>::zeros(data_set_size.f());

        println!("reading data ...");
        let data_buffer = data_set.as_slice_memory_order_mut().unwrap();
        data_buffer.par_chunks_exact_mut(vol_stride).enumerate().for_each(|(i, x)| {
            println!("working on vol {i} ...");
            let vol_name = Path::new(data_dir).join(format!("vol{:03}", i));
            let vol = cfl::to_array(&vol_name, true).unwrap();
            x.copy_from_slice(vol.as_slice_memory_order().unwrap());
        });

        println!("applying mask ...");
        data_buffer.par_chunks_exact_mut(vol_stride).zip(msk_buffer.par_chunks_exact(pe_stride)).for_each(|(x, y)| {
            x.chunks_exact_mut(vol_dim[0]).zip(y).for_each(|(line, msk)| {
                if msk.is_zero() {
                    line.fill(Complex32::zero());
                }
            })
        });

        println!("writing data ...");
        let data_buffer = data_set.as_slice_memory_order_mut().unwrap();
        println!("writing data ...");
        data_buffer.par_chunks_exact(vol_stride).enumerate().for_each(|(i, x)| {
            println!("working on vol {i} ...");
            let vol_name = Path::new(data_dir).join(format!("vol{:03}", i));
            let mut vol_buff = ArrayD::zeros(vol_dim.as_slice().f());
            vol_buff.as_slice_memory_order_mut().unwrap().copy_from_slice(x);
            let vol = cfl::from_array(&vol_name, &vol_buff).unwrap();
        });
    }

    #[test]
    fn to_nifti() {
        let vol = "B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32/cfl/vol001";
        let vol = cfl::to_array(vol, true).unwrap();
        dump_magnitude("B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32/cfl/vol001", &vol);
    }
}