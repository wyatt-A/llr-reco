use crate::signal_model::{signal_model_batched, ModelDirection};
use byteorder::ByteOrder;
use cfl::ndarray::{Array2, Array4, Axis, Ix4, ShapeBuilder};
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

/// read vectors from a text file with whitespace delimiter
pub fn read_bvecs(txt_file: impl AsRef<Path>) -> Vec<[f32; 3]> {
    let mut s = String::new();
    let mut f = File::open(txt_file).expect("failed to open file");
    f.read_to_string(&mut s).expect("failed to read file");
    let values = s.split_ascii_whitespace().into_iter().filter_map(|s| s.parse::<f32>().ok()).collect::<Vec<f32>>();
    // assert that the number of total values is divisible by 3
    assert_eq!(values.len() % 3, 0, "invalid number of values. Expecting");
    let n_bvecs = values.len() / 3;
    let a = Array2::<f32>::from_shape_vec((n_bvecs, 3).f(), values).unwrap();
    a.axis_iter(Axis(0)).map(|bvec| {
        let mut v = [0., 0., 0.];
        v.iter_mut().zip(bvec.iter()).for_each(|(a, &b)| *a = b);
        v
    }).collect::<Vec<[f32; 3]>>()
}

/// return an index mask of b0 volumes (null vectors) with some magnitude tolerance
pub fn b0_mask(b_vecs: &[[f32; 3]], tolerance: f32) -> Vec<bool> {
    assert!(tolerance >= 0., "vector magnitude tolerance must be non-negative");
    b_vecs.iter().map(|b| {
        let norm_sq = b[0].powi(2) + b[1].powi(2) + b[2].powi(2);
        norm_sq.sqrt() <= tolerance
    }).collect()
}

/// load a 4-D nifti file to a 4-D array of B0 volumes, DTI volumes, and their b-vectors
pub fn load_dataset(nii_4d: impl AsRef<Path>, b_vecs: impl AsRef<Path>) -> (Array4<Complex32>, Array4<Complex32>, Vec<[f32; 3]>) {
    let b_vecs = read_bvecs(b_vecs);
    let b0_mask = b0_mask(&b_vecs, 0.);

    let b_vecs = b_vecs.into_iter().zip(&b0_mask).filter(|(_, b)| !**b).map(|(v, _)| v).collect::<Vec<_>>();

    let n_vols = b0_mask.len();

    let dti_indices = b0_mask.iter().enumerate().filter(|(_, b)| !**b).map(|(idx, _)| idx).collect::<Vec<_>>();
    let b0_indices = b0_mask.iter().enumerate().filter(|(_, b)| **b).map(|(idx, _)| idx).collect::<Vec<_>>();

    let n_dti = dti_indices.len();
    let n_b0s = b0_indices.len();


    println!("found {} angles, {} b0s", n_dti, n_b0s);

    assert!(n_dti > 0, "expected at least one dti volume");
    assert!(n_b0s > 0, "expected at least one b0 volume");

    println!("loading data ...");
    let big_array = cfl::read_nifti_to_cfl(nii_4d, None::<&str>);
    let stack = big_array.into_dimensionality::<Ix4>().expect("failed to cast array into 4D");
    let [nx, ny, nz, nq]: [usize; 4] = stack.dim().into();

    assert_eq!(n_vols, nq, "unexpected dimensionality. Expected {} volumes from b_vec file", n_vols);

    println!("reshaping ...");
    let mut dti = Array4::<Complex32>::zeros((nx, ny, nz, n_dti).f());
    dti.axis_iter_mut(Axis(3)).zip(dti_indices).for_each(|(mut d, i)| {
        d.assign(&stack.index_axis(Axis(3), i));
    });
    let mut b0s = Array4::<Complex32>::zeros((nx, ny, nz, n_b0s).f());
    b0s.axis_iter_mut(Axis(3)).zip(b0_indices).for_each(|(mut b, i)| {
        b.assign(&stack.index_axis(Axis(3), i));
    });

    println!("data successfully loaded.");
    (b0s, dti, b_vecs)
}

/// loads image 4-D dti data and performs under sampling, returning under sampled k-space for both dti
/// and b0 volumes, as well as the b-vector table without the null vectors associated with b0s
pub fn load_and_undersample(nii_4d: impl AsRef<Path>, b_vecs: impl AsRef<Path>, sample_mask_cfl: impl AsRef<Path>) -> (Array4<Complex32>, Array4<Complex32>, Vec<[f32; 3]>) {
    let sample_mask = cfl::to_array(sample_mask_cfl, true).unwrap();
    let (mut b0s, mut dti, b_vecs) = load_dataset(nii_4d, b_vecs);

    let [nx, ny, nz, nq]: [usize; 4] = dti.dim().into();
    let [_, _, _, nb0]: [usize; 4] = b0s.dim().into();
    let vol_stride = nx * ny * nz;
    let pe_stride = ny * nz;

    let ksp_corr = vec![[0., 0., 0.]; nq];
    let img_corr = vec![[0., 0., 0.]; nq];

    println!("performing fft ...");
    signal_model_batched(&mut dti, 20, &ksp_corr, &img_corr, ModelDirection::Forward);
    signal_model_batched(&mut b0s, 20, &ksp_corr[0..nb0], &img_corr[0..nb0], ModelDirection::Forward);

    let dti_data_buff = dti.as_slice_memory_order_mut().unwrap();
    let b0_data_buff = b0s.as_slice_memory_order_mut().unwrap();
    let msk_buffer = sample_mask.as_slice_memory_order().unwrap();

    println!("applying sample mask to dti ...");
    dti_data_buff.par_chunks_exact_mut(vol_stride).zip(msk_buffer.par_chunks_exact(pe_stride)).for_each(|(x, y)| {
        x.chunks_exact_mut(nx).zip(y).for_each(|(line, msk)| {
            if msk.is_zero() {
                line.fill(Complex32::zero());
            }
        })
    });

    println!("applying sample mask to b0s ...");
    b0_data_buff.par_chunks_exact_mut(vol_stride).zip(msk_buffer[(nq * pe_stride)..].par_chunks_exact(pe_stride)).for_each(|(x, y)| {
        x.chunks_exact_mut(nx).zip(y).for_each(|(line, msk)| {
            if msk.is_zero() {
                line.fill(Complex32::zero());
            }
        })
    });

    (b0s, dti, b_vecs)
}

#[cfg(test)]
mod tests {
    use crate::data_import::{b0_mask, load_and_undersample, read_bvecs, read_civm_i16_volume, read_i16_be_to_cfl};
    use crate::signal_model::{signal_model_batched, ModelDirection};
    use cfl::dump_magnitude;
    use cfl::ndarray::{Array2, Array3, Array4, ArrayD, ShapeBuilder};
    use cfl::num_complex::Complex32;
    use cfl::num_traits::Zero;
    use rayon::prelude::*;
    use std::path::Path;
    use std::time::Instant;

    #[test]
    fn load_bvec() {
        let bvecs = read_bvecs("C:/Users/wa41/dti_subspace_model/bvec_120.txt");
        println!("{:?}", bvecs);
        println!("{:?}", b0_mask(&bvecs, 0.));
    }

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
        let lin_corr_img = vec![[0f32, 0f32, 0f32]; n_vols];

        println!("reading data ...");

        let data_buffer = data_set.as_slice_memory_order_mut().unwrap();
        data_buffer.par_chunks_exact_mut(vol_stride).enumerate().for_each(|(i, x)| {
            println!("working on vol {i} ...");
            let vol_name = Path::new(data_dir).join(format!("vol{:03}", i));
            let vol = cfl::to_array(&vol_name, true).unwrap();
            x.copy_from_slice(vol.as_slice_memory_order().unwrap());
        });


        println!("performing fft ...");
        signal_model_batched(&mut data_set, 20, &lin_corr, &lin_corr_img, ModelDirection::Forward);

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

    #[test]
    fn read_from_nifti() {
        let stack = "B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32/nifti/translated/stack4d.nii";
        println!("reading nifti data set ...");
        let now = Instant::now();
        let big_array = cfl::read_nifti_to_cfl(stack, None::<&str>);
        let dur = now.elapsed();
        println!("data set read in {} secs", dur.as_secs());
    }

    #[test]
    fn test_load_nii() {
        // let (b0s, dti, _) = load_dataset(
        //     "B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32/nifti/xformed_affine/stack4d.nii",
        //     "B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32/bvec_120.txt",
        // );

        let (b0s, dti, _) = load_and_undersample(
            "B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32/nifti/xformed_affine/stack4d.nii",
            "B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32/bvec_120.txt",
            "B:/ProjectSpace/wa41/llr-reco-test-data/13.gaj.32/msk_vol",
        );

        println!("b0s: {:?}", b0s.shape());
        println!("dti: {:?}", dti.shape());
    }
}