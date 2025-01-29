// Array utilities for column-major memory layouts

#[inline(always)]
pub fn index_to_subscript_col_maj3(index: usize, size: &[usize; 3]) -> [usize; 3] {
    let iz = index / (size[0] * size[1]);
    let rem = index % (size[0] * size[1]);
    let iy = rem / size[0];
    let ix = rem % size[0];
    [ix, iy, iz]
}

#[inline(always)]
pub fn subscript_to_index_col_maj3(subscript: &[usize; 3], size: &[usize; 3]) -> usize {
    let z_stride = size[0] * size[1];
    let y_stride = size[0];
    subscript[2] * z_stride + subscript[1] * y_stride + subscript[0]
}