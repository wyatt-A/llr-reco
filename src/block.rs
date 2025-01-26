use cfl::ndarray::parallel::prelude::*;
use std::sync::{Arc, Mutex};

/// reads a 3-D block of data from a 3-D volume using a periodic boundary condition. The block
/// coordinate specifies the lower corner of block in the volume space. This function does not
/// enforce out-of-bounds reads or writes from either the volume data or block data.
fn _read_block_unchecked<T>(block_size: &[usize; 3], block_coord: &[usize; 3], vol_size: &[usize; 3], vol_data: &[T], block_data: &mut [T])
where
    T: Sized + Copy + Send + Sync,
{
    block_data.par_iter_mut().enumerate().for_each(|(idx, block_entry)| {
        // calculate the block coordinate
        let mut iz = idx / (block_size[0] * block_size[1]);
        let rem = idx % (block_size[0] * block_size[1]);
        let mut iy = rem / block_size[0];
        let mut ix = rem % block_size[0];

        // apply block coord offset
        iz = (iz + block_coord[2]) % vol_size[2];
        iy = (iy + block_coord[1]) % vol_size[1];
        ix = (ix + block_coord[0]) % vol_size[0];

        // convert to linear volume address to perform write
        let vol_address = vol_size[0] * vol_size[1] * iz + vol_size[0] * iy + ix;
        *block_entry = vol_data[vol_address];
    });
}

/// writes a 3-D block of data to a 3-D volume using a periodic boundary condition. The block
/// coordinate specifies the lower corner of block in the volume space. This function does not
/// enforce out-of-bounds reads or writes from either the volume data or block data.
fn _write_block_unchecked<T>(block_size: &[usize; 3], block_coord: &[usize; 3], vol_size: &[usize; 3], vol_data: &mut [T], block_data: &[T])
where
    T: Sized + Copy + Send + Sync,
{
    //let vol_data_ptr = vol_data.as_mut_ptr();
    //let vol_data_cell = UnsafeCell::new(vol_data);
    let vol_data_ref = Arc::new(Mutex::new(vol_data));

    block_data.par_iter().enumerate().for_each(|(idx, &block_entry)| {
        // calculate the block coordinate
        let mut iz = idx / (block_size[0] * block_size[1]);
        let rem = idx % (block_size[0] * block_size[1]);
        let mut iy = rem / block_size[0];
        let mut ix = rem % block_size[0];

        // apply block coord offset
        iz = (iz + block_coord[2]) % vol_size[2];
        iy = (iy + block_coord[1]) % vol_size[1];
        ix = (ix + block_coord[0]) % vol_size[0];

        // convert to linear volume address to perform write
        let vol_address = vol_size[0] * vol_size[1] * iz + vol_size[0] * iy + ix;

        let mut vol_data_locked = vol_data_ref.lock().unwrap();
        vol_data_locked[vol_address] = block_entry;
    });
}

fn read_matrix<T>(vol_size: &[usize; 3], block_size: &[usize; 3], block_coord: &[usize; 3], volumes: &[&[T]], matrix_data: &mut [T])
where
    T: Copy + Send + Sync,
{
    // assert that an appropriate block size was chosen
    vol_size.iter().zip(block_size).for_each(|(&vol_dim, &block_dim)| {
        assert!(block_dim <= vol_dim, "block dimension must be less than or equal to volume dimension");
    });

    // infer column stride from block size
    let col_stride = block_size.iter().product::<usize>();

    // assert data buffer size consistency
    let matrix_size = col_stride * volumes.len();
    assert_eq!(matrix_size, matrix_data.len(), "mismatch between expected matrix size and matrix_data length");

    // read matrix from volume stack in col-maj order
    for (col, vol) in matrix_data.chunks_exact_mut(col_stride).zip(volumes) {
        _read_block_unchecked(block_size, block_coord, vol_size, vol, col);
    }
}

fn write_matrix<T>(vol_size: &[usize; 3], block_size: &[usize; 3], block_coord: &[usize; 3], volumes: &mut [&mut [T]], matrix_data: &[T])
where
    T: Copy + Send + Sync,
{
    // assert that an appropriate block size was chosen
    vol_size.iter().zip(block_size).for_each(|(&vol_dim, &block_dim)| {
        assert!(block_dim <= vol_dim, "block dimension must be less than or equal to volume dimension");
    });

    // infer column stride from block size
    let col_stride = block_size.iter().product::<usize>();

    // assert data buffer size consistency
    let matrix_size = col_stride * volumes.len();
    assert_eq!(matrix_size, matrix_data.len(), "mismatch between expected matrix size and matrix_data length");

    // write matrix columns as blocks into each volume in parallel
    volumes.par_iter_mut().enumerate().for_each(|(idx, vol)| {
        let start = idx * col_stride;
        let end = start + col_stride;
        let col = &matrix_data[start..end];
        _write_block_unchecked(block_size, block_coord, vol_size, vol, col);
    })
}

/// calculates the block grid dimensions for a volume and block size
fn grid_dim(vol_size: &[usize; 3], block_size: &[usize; 3]) -> [usize; 3] {
    let mut grid = [1usize; 3];
    grid.iter_mut().zip(vol_size).zip(block_size).for_each(|((n, vol_size), block_size)| {
        *n = vol_size / block_size;
    });
    grid
}

/// converts a block index to a block coordinate given the volume and block size. This doesn't
/// check if the block index is out of range.
fn block_coord(block_idx: usize, vol_size: &[usize; 3], block_size: &[usize; 3], shift: &[usize; 3]) -> [usize; 3] {
    let grid_size = grid_dim(vol_size, block_size);
    let mut iz = block_idx / (grid_size[0] * grid_size[1]);
    let rem = block_idx % (grid_size[0] * grid_size[1]);
    let mut iy = rem / grid_size[0];
    let mut ix = rem % grid_size[0];
    ix = ix * block_size[0] + shift[0];
    iy = iy * block_size[1] + shift[1];
    iz = iz * block_size[2] + shift[2];
    [ix, iy, iz]
}


fn matrix_size(n_volumes: usize, block_size: &[usize; 3]) -> [usize; 2] {
    [
        block_size.iter().product::<usize>(),
        n_volumes
    ]
}

fn matrix_stride(n_volumes: usize, block_size: &[usize; 3]) -> usize {
    matrix_size(n_volumes, block_size).iter().product()
}

fn lift_data_set<T>(volumes: &[&[T]], matrix_data: &mut [T], vol_size: &[usize; 3], block_size: &[usize; 3], shift: &[usize; 3], matrix_idx_start: usize, n_matrices: usize)
where
    T: Copy + Send + Sync,
{
    let mat_stride = matrix_stride(volumes.len(), block_size);
    assert_eq!(matrix_data.len(), mat_stride * n_matrices, "unexpected matrix data size");
    for i in matrix_idx_start..(matrix_idx_start + n_matrices) {
        let start = i * mat_stride;
        let end = start + mat_stride;
        let mat = &mut matrix_data[start..end];
        let block = block_coord(i, &vol_size, &block_size, &shift); // + random shift
        read_matrix(&vol_size, &block_size, &block, volumes, mat);
    }
}

fn unlift_data_set<T>(volumes: &mut [&mut [T]], matrix_data: &[T], vol_size: &[usize; 3], block_size: &[usize; 3], shift: &[usize; 3], matrix_idx_start: usize, n_matrices: usize)
where
    T: Copy + Send + Sync,
{
    let mat_stride = matrix_stride(volumes.len(), block_size);
    assert_eq!(matrix_data.len(), mat_stride * n_matrices, "unexpected matrix data size");
    for i in matrix_idx_start..(matrix_idx_start + n_matrices) {
        let start = i * mat_stride;
        let end = start + mat_stride;
        let mat = &matrix_data[start..end];
        let block = block_coord(i, &vol_size, &block_size, &shift); // + random shift
        write_matrix(&vol_size, &block_size, &block, volumes, mat);
    }
}


// fn insert_matrix(data_set: &mut [&mut [Complex32]], vol_dims: [usize; 3], matrix: &Array2<Complex32>, patch_range: &[Range<usize>; 3]) {
//     // infer matrix dims from data set slice
//     let n_columns = data_set.len();
//     let n_rows =
//         (patch_range[0].end - patch_range[0].start) *
//             (patch_range[1].end - patch_range[1].start) *
//             (patch_range[2].end - patch_range[2].start);
//
//     assert_eq!(n_rows * n_columns, matrix.len(), "n_rows and m_columns inconsistent with matrix data size");
//
//     let mut xr = vec![0usize; patch_range[0].end - patch_range[0].start];
//     let mut yr = vec![0usize; patch_range[1].end - patch_range[1].start];
//     let mut zr = vec![0usize; patch_range[2].end - patch_range[2].start];
//
//     // account for out-of-range indices by performing modulo over volume size
//     patch_range[0].clone().zip(&mut xr).for_each(|(x_idx, xr)| *xr = x_idx % vol_dims[0]);
//     patch_range[1].clone().zip(&mut yr).for_each(|(y_idx, yr)| *yr = y_idx % vol_dims[1]);
//     patch_range[2].clone().zip(&mut zr).for_each(|(z_idx, zr)| *zr = z_idx % vol_dims[2]);
//
//     let matrix_data = matrix.as_slice_memory_order().expect("matrix is not contiguous");
//     let mut mat_idx = 0;
//     for vol in data_set.iter_mut() {
//         for &x in &xr {
//             for &y in &yr {
//                 for &z in &zr {
//                     let col_maj_idx = x + y * vol_dims[0] + z * vol_dims[0] * vol_dims[1];
//                     vol[col_maj_idx] = matrix_data[mat_idx];
//                     mat_idx += 1;
//                 }
//             }
//         }
//     }
// }

// fn generate_patch_ranges(vol_size: impl Into<[usize; 3]>, patch_size: impl Into<[usize; 3]>, offset: impl Into<[usize; 3]>) -> Vec<[Range<usize>; 3]> {
//     let vol_size = vol_size.into();
//     let patch_size = patch_size.into();
//     let offset = offset.into();
//     let n_x = vol_size[0] / patch_size[0];
//     let n_y = vol_size[1] / patch_size[1];
//     let n_z = vol_size[2] / patch_size[2];
//     let mut patches = Vec::<[Range<usize>; 3]>::with_capacity(n_x * n_y * n_z);
//     for x in 0..n_x {
//         let x_start = &x * patch_size[0];
//         for y in 0..n_y {
//             let y_start = &y * patch_size[1];
//             for z in 0..n_z {
//                 let z_start = &z * patch_size[2];
//                 let xr = (x_start + offset[0])..(x_start + patch_size[0] + offset[0]);
//                 let yr = (y_start + offset[1])..(y_start + patch_size[1] + offset[1]);
//                 let zr = (z_start + offset[2])..(z_start + patch_size[2] + offset[2]);
//                 patches.push([xr, yr, zr])
//             }
//         }
//     }
//     patches
// }

#[cfg(test)]
mod tests {
    use crate::block::{_read_block_unchecked, _write_block_unchecked, grid_dim, lift_data_set, unlift_data_set};
    use cfl::ndarray::{Array3, Array4, ShapeBuilder};
    use cfl::num_complex::Complex32;

    #[test]
    fn test_read_block() {
        let vol_size = [10, 12, 14];
        let block_size = [1, 1, 4];
        let volume = Array3::<usize>::from_shape_fn(vol_size.f(), |(i, j, k)| i + j + k);
        let mut block = Array3::<usize>::from_elem(block_size.f(), 0);

        let block_data = block.as_slice_memory_order_mut().unwrap();
        let vol_data = volume.as_slice_memory_order().unwrap();

        let block_coord = [0, 0, 12];
        _read_block_unchecked(&block_size, &block_coord, &vol_size, vol_data, block_data);

        println!("{:?}", block);
    }

    #[test]
    fn test_write_block() {
        let vol_size = [256, 128, 64];
        let block_size = [32, 32, 32];
        let mut volume = Array3::from_shape_fn(vol_size.f(), |(i, j, k)| Complex32::new((i + j + k) as f32, 0.));
        let block = Array3::from_elem(block_size.f(), Complex32::ZERO);

        let block_data = block.as_slice_memory_order().unwrap();
        let vol_data = volume.as_slice_memory_order_mut().unwrap();

        let block_coord = [0, 0, 33];
        _write_block_unchecked(&block_size, &block_coord, &vol_size, vol_data, block_data);

        cfl::dump_magnitude("test_out", &volume.into_dyn());
    }


    #[test]
    fn test_lift_unlift_dataset() {

        // define the image volume size, number of volumes,
        // and matrix size based on the block size
        // we are lifting a bunch of (1000 x 10) matrices out of the data set
        // we also define an arbitrary shift to offset the matrix lifting
        let vol_size = [788, 480, 480];
        let n_vols = 5;
        let block_size = [32, 32, 32];
        let shift = [128, 64, 32];

        // the volume stride is the number of elements in a volume
        let vol_stride = vol_size.iter().product::<usize>();

        // the matrix size is the total block size by the number of volumes
        let matrix_size = [block_size.iter().product::<usize>(), n_vols];

        // determine the grid size, the number of blocks to lift in each dimension.
        // Blocks do not overlap.
        let grid_size = grid_dim(&vol_size, &block_size);

        // the number of matrices is the total size of the block grid
        let n_matrices = grid_size.iter().product::<usize>();

        // build the test data set (test gradient images)
        let data_set_shape = (vol_size[0], vol_size[1], vol_size[2], n_vols);
        let mut data_set = Array4::from_shape_fn(data_set_shape.f(), |(i, j, k, l)| {
            Complex32::new((i + j + k + l) as f32, 0.)
        });
        let orig = data_set.clone();

        // allocate the matrix data
        let matrix_batch_size = (matrix_size[0], matrix_size[1], n_matrices);
        let mut matrix_batch = Array3::from_elem(matrix_batch_size.f(), Complex32::ZERO);

        // create references to the volume data for the lifting function
        let vol_data = data_set.as_slice_memory_order().unwrap();
        let volumes: Vec<&[Complex32]> = vol_data.chunks_exact(vol_stride).collect();

        // get a mutable reference to the matrix data
        let matrix_data = matrix_batch.as_slice_memory_order_mut().unwrap();

        // perform lifting operation, writing to matrix data
        lift_data_set(&volumes, matrix_data, &vol_size, &block_size, &shift, 0, n_matrices);

        // this time, get a mutable set of references to the volume data for writing
        let vol_data = data_set.as_slice_memory_order_mut().unwrap();
        let mut volumes: Vec<&mut [Complex32]> = vol_data.chunks_exact_mut(vol_stride).collect();

        // get an immutable reference to matrix data
        let matrix_data = matrix_batch.as_slice_memory_order().unwrap();

        // perform un-lifting operation to write matrix data back into the volumes
        unlift_data_set(&mut volumes, matrix_data, &vol_size, &block_size, &shift, 0, n_matrices);

        // assert that the data set we wrote to is the same as the original, testing that
        // the lifting / un-lifting did not corrupt the data set
        assert_eq!(data_set, orig, "lift and un-lift operation is inconsistent");
    }
}