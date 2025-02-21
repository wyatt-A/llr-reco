use crate::signal_model::{signal_model_batched, ModelDirection};
use byteorder::ByteOrder;
use cfl::ndarray::{Array2, Array3, Array4, Axis, Ix3, Ix4, ShapeBuilder};
use cfl::num_complex::Complex32;
use cfl::num_traits;
use cfl::num_traits::{ToPrimitive, Zero};
use num_traits::FromBytes;
use rayon::prelude::*;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

pub struct DtiDataSetBuilder {
    cfl_files: Vec<PathBuf>,
    vol_dims: [usize; 3],
    b_vecs: Option<Vec<[f32; 3]>>,
    b_vals: Option<Vec<f32>>,
    /// mask of cfl headers pointing to B0 volumes
    b0_mask: Option<Vec<bool>>,
    sample_mask: Option<Array3<Complex32>>,
}

impl DtiDataSetBuilder {
    pub fn b_vecs(&self) -> &[[f32; 3]] {
        self.b_vecs.as_ref().unwrap()
    }

    pub fn b_vals(&self) -> &[f32] {
        self.b_vals.as_ref().unwrap()
    }

    pub fn b0_mask(&self) -> &[bool] {
        self.b0_mask.as_ref().unwrap()
    }

    pub fn from_cfl_images<P: AsRef<Path> + Send + Sync, F: Fn(usize) -> P + Sync>(vol_indices: &[usize], filename_fn: &F) -> Self {
        let n_vols = vol_indices.len();
        assert!(n_vols > 0, "vol indices cannot be empty");

        // load dimensions of first volume
        let vol_dims = cfl::get_dims(filename_fn(vol_indices[0]))
            .expect("failed to get volume dimensions")
            // remove singleton dims
            .into_iter().filter(|&dim| dim != 1).map(|x| x).collect::<Vec<_>>();

        // load the rest of the headers to check consistency
        vol_indices[1..].par_iter().for_each(|&idx| {
            let dims = cfl::get_dims(filename_fn(idx))
                .expect("failed to get volume dimensions")
                // remove singleton dims
                .into_iter().filter(|&dim| dim != 1).map(|x| x).collect::<Vec<_>>();
            assert_eq!(vol_dims, dims, "mismatch between first volume dims and volume dims {idx}");
        });

        assert_eq!(vol_dims.len(), 3, "expecting n volume dims to be 3");

        DtiDataSetBuilder {
            cfl_files: vol_indices.iter().map(|&idx| filename_fn(idx).as_ref().to_path_buf()).collect(),
            vol_dims: [vol_dims[0], vol_dims[1], vol_dims[2]],
            b_vecs: None,
            b_vals: None,
            b0_mask: None,
            sample_mask: None,
        }
    }

    pub fn from_cfl_volume_dir(cfl_dir: impl AsRef<Path>, filename_pattern: impl AsRef<str>, num_expected_images: Option<usize>) -> Self {
        let full_pattern = cfl_dir.as_ref().join(filename_pattern.as_ref()).with_extension("cfl").to_string_lossy().to_string();
        let mut matching_files = glob::glob(&full_pattern)
            .expect("failed to read glob pattern")
            .flat_map(|f| f.ok())
            .collect::<Vec<PathBuf>>();

        if let Some(expected_images) = num_expected_images {
            assert_eq!(matching_files.len(), expected_images, "unexpected number of filed found");
        }

        assert!(!matching_files.is_empty(), "no files found");

        /// sort in ascending order, assuming numerical naming
        matching_files.sort();

        // load dimensions of first volume
        let vol_dims = cfl::get_dims(&matching_files[0])
            .expect("failed to get volume dimensions")
            // remove singleton dims
            .into_iter().filter(|&dim| dim != 1).map(|x| x).collect::<Vec<_>>();

        // load the rest of the headers to check consistency
        matching_files.par_iter().for_each(|file| {
            let dims = cfl::get_dims(file)
                .expect("failed to get volume dimensions")
                // remove singleton dims
                .into_iter().filter(|&dim| dim != 1).map(|x| x).collect::<Vec<_>>();
            assert_eq!(vol_dims, dims, "mismatch between first volume dims and volume {}", file.file_name().unwrap().to_string_lossy());
        });

        assert_eq!(vol_dims.len(), 3, "expecting n volume dims to be 3");

        DtiDataSetBuilder {
            cfl_files: matching_files,
            vol_dims: [vol_dims[0], vol_dims[1], vol_dims[2]],
            b_vecs: None,
            b_vals: None,
            b0_mask: None,
            sample_mask: None,
        }
    }

    pub fn with_b_table(mut self, b_table: impl AsRef<Path>, b0_tol: f32) -> Self {
        let mut f = File::open(b_table.as_ref().with_extension("txt")).expect("failed to open b table");
        let mut s = String::new();

        f.read_to_string(&mut s).expect("failed to read string");

        let mut values: Vec<f32> = vec![];

        s.split_ascii_whitespace().for_each(|character| {
            if let Ok(val) = character.parse::<f32>() {
                values.push(val);
            } else {
                panic!("failed to parse b-table at character: {}", character);
            }
        });

        assert_eq!(values.len() % 4, 0, "number b-table entries must be divisible by 4: [b-value, gx, gy, gz]");
        let n_vecs = values.len() / 4;

        assert_eq!(n_vecs, self.cfl_files.len(), "expected {} bvecs, received {}", self.cfl_files.len(), n_vecs);

        let mut b_vals = vec![0f32; n_vecs];
        let mut b_vecs = vec![[0., 0., 0.]; n_vecs];

        values.chunks_exact(4).zip(b_vals.iter_mut().zip(b_vecs.iter_mut())).for_each(|(a, (b, c))| {
            *b = a[0];
            c.copy_from_slice(&a[1..])
        });

        let b0_mask = b0_mask(&b_vecs, b0_tol);

        self.b0_mask = Some(b0_mask);
        self.b_vals = Some(b_vals);
        self.b_vecs = Some(b_vecs);
        self
    }

    pub fn load_dti_set(&self) -> Array4<Complex32> {
        assert!(self.b0_mask.is_some(), "you must load a b-table first");
        let b0_mask = self.b0_mask.as_ref().unwrap();
        let dti_files = self.cfl_files.iter()
            .zip(b0_mask.iter())
            .filter(|(_, b)| !**b)
            .map(|(cfl, _)| cfl)
            .collect::<Vec<_>>();
        self.load_volumes(&dti_files)
    }

    pub fn load_b0_set(&self) -> Array4<Complex32> {
        assert!(self.b0_mask.is_some(), "you must load a b-table first");
        let b0_mask = self.b0_mask.as_ref().unwrap();
        let b0_files = self.cfl_files.iter()
            .zip(b0_mask.iter())
            .filter(|(_, b)| **b)
            .map(|(cfl, _)| cfl)
            .collect::<Vec<_>>();
        self.load_volumes(&b0_files)
    }

    pub fn load_all(&self) -> Array4<Complex32> {
        self.load_volumes(&self.cfl_files)
    }

    fn load_volumes<T: AsRef<Path> + Send + Sync>(&self, vols: &[T]) -> Array4<Complex32> {
        let vol_stride = self.vol_dims.iter().product();
        let total_samples = vol_stride * vols.len();
        let mut array_data = vec![Complex32::zero(); total_samples];
        array_data.par_chunks_exact_mut(vol_stride).zip(vols.par_iter()).for_each(|(vol_data, cfl)| {
            cfl::read_to_buffer(cfl, vol_data).expect("failed to read cfl");
        });
        let [nx, ny, nz]: [usize; 3] = self.vol_dims;
        Array4::from_shape_vec(
            (nx, ny, nz, vols.len()).f(),
            array_data,
        ).expect("failed to create array")
    }

    pub fn with_sample_mask(mut self, sample_mask_cfl: impl AsRef<Path>) -> Self {
        let sample_mask = cfl::to_array(sample_mask_cfl, true)
            .unwrap()
            .into_dimensionality::<Ix3>()
            .unwrap();
        let [ny, nz, nq]: [usize; 3] = sample_mask.dim().into();
        assert_eq!(nq, self.cfl_files.len());
        assert_eq!(ny, self.vol_dims[1]);
        assert_eq!(nz, self.vol_dims[2]);
        self.sample_mask = Some(sample_mask);
        self
    }

    pub fn full_sample_mask(&self) -> &Array3<Complex32> {
        &self.sample_mask.as_ref().expect("sample mask not specified")
    }

    pub fn dti_sample_mask(&self) -> Array3<Complex32> {
        assert!(self.b0_mask.is_some(), "you must load a b-table first");
        assert!(self.sample_mask.is_some(), "you must load a sample mask first");

        let sample_mask = self.sample_mask.as_ref().unwrap();
        let b0_mask = self.b0_mask.as_ref().unwrap();

        // get all dti indices
        let indices = b0_mask.iter().enumerate().filter(|(_, b)| !**b).map(|(i, _)| i).collect::<Vec<_>>();

        let n_dti = indices.len();

        let [ny, nz, _]: [usize; 3] = sample_mask.dim().into();

        let pe_stride = ny * nz;
        let mut mask_entries = vec![Complex32::zero(); ny * nz * n_dti];
        let all_entries = sample_mask.as_slice_memory_order().unwrap();

        indices.par_iter().zip(mask_entries.par_chunks_exact_mut(pe_stride)).for_each(|(&index, mask)| {
            let range = index * pe_stride..(index * pe_stride + pe_stride);
            mask.copy_from_slice(&all_entries[range])
        });

        Array3::from_shape_vec((ny, nz, n_dti).f(), mask_entries).unwrap()
    }

    pub fn b0_sample_mask(&self) -> Array3<Complex32> {
        assert!(self.b0_mask.is_some(), "you must load a b-table first");
        assert!(self.sample_mask.is_some(), "you must load a sample mask first");

        let sample_mask = self.sample_mask.as_ref().unwrap();
        let b0_mask = self.b0_mask.as_ref().unwrap();

        // get all b0 indices
        let indices = b0_mask.iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| i).collect::<Vec<_>>();

        let n_b0 = indices.len();

        let [ny, nz, _]: [usize; 3] = sample_mask.dim().into();

        let pe_stride = ny * nz;
        let mut mask_entries = vec![Complex32::zero(); ny * nz * n_b0];
        let all_entries = sample_mask.as_slice_memory_order().unwrap();

        indices.par_iter().zip(mask_entries.par_chunks_exact_mut(pe_stride)).for_each(|(&index, mask)| {
            let range = index * pe_stride..(index * pe_stride + pe_stride);
            mask.copy_from_slice(&all_entries[range])
        });

        Array3::from_shape_vec((ny, nz, n_b0).f(), mask_entries).unwrap()
    }

    pub fn dti_indices(&self) -> Vec<usize> {
        assert!(self.b0_mask.is_some(), "you must load a b-table first");
        let b0_mask = self.b0_mask.as_ref().unwrap();
        // get all dti indices
        b0_mask.iter().enumerate().filter(|(_, b)| !**b).map(|(i, _)| i).collect()
    }

    pub fn b0_indices(&self) -> Vec<usize> {
        assert!(self.b0_mask.is_some(), "you must load a b-table first");
        let b0_mask = self.b0_mask.as_ref().unwrap();
        // get all b0 indices
        b0_mask.iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| i).collect()
    }

    pub fn total_volumes(&self) -> usize {
        self.cfl_files.len()
    }
}


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
    let big_array = cfl::read_nifti_to_cfl(nii_4d.as_ref(), None::<&Path>);
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


pub fn under_sample_kspace(data_set: &mut Array4<Complex32>, sample_masks: &Array3<Complex32>) {
    let [nx, ny, nz, nq]: [usize; 4] = data_set.dim().into();

    let [nsy, nsz, nsq]: [usize; 3] = sample_masks.dim().into();

    assert_eq!([ny, nz], [nsy, nsz], "phase encoding dimension mismatch");
    assert_eq!(nq, nsq, "contrast dimension mismatch");

    let vol_stride = nx * ny * nz;
    let pe_stride = ny * nz;

    let mut sample_buff = data_set.as_slice_memory_order_mut().unwrap();
    let msk_buffer = sample_masks.as_slice_memory_order().unwrap();

    println!("applying sample mask to dti ...");
    sample_buff.par_chunks_exact_mut(vol_stride).zip(msk_buffer.par_chunks_exact(pe_stride)).for_each(|(x, y)| {
        x.chunks_exact_mut(nx).zip(y).for_each(|(line, msk)| {
            if msk.is_zero() {
                line.fill(Complex32::zero());
            }
        })
    });
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