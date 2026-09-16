use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array::{RustArray, Shape};

/// The three matmul shape combinations `ArrayLayer`'s own formulas actually use - matrix @
/// vector (`self.W @ x`), vector @ matrix (the interface subset's own "1D x 2D" case, not
/// exercised by the current class design but part of its documented contract), and matrix @
/// matrix (`X @ self.W.T`, `next_layer.delta_batch @ next_layer.W`,
/// `self.delta_batch.T @ input_activation_batch`). A naive triple loop, not a BLAS-competitive
/// routine - see docs/rust-array-core.md's own "expected performance" for why that gap is
/// expected and accepted at this stage.
pub(crate) fn matmul(a: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    match (a.shape, b.shape) {
        (Shape::Matrix(rows, cols), Shape::Vector(n)) => {
            if cols != n {
                return Err(shape_error(a.shape, b.shape));
            }
            let mut out = vec![0.0; rows];
            for row in 0..rows {
                let mut sum = 0.0;
                for k in 0..cols {
                    sum += a.data[row * cols + k] * b.data[k];
                }
                out[row] = sum;
            }
            Ok(RustArray::from_vector(out))
        }
        (Shape::Vector(n), Shape::Matrix(rows, cols)) => {
            if n != rows {
                return Err(shape_error(a.shape, b.shape));
            }
            let mut out = vec![0.0; cols];
            for col in 0..cols {
                let mut sum = 0.0;
                for k in 0..rows {
                    sum += a.data[k] * b.data[k * cols + col];
                }
                out[col] = sum;
            }
            Ok(RustArray::from_vector(out))
        }
        (Shape::Matrix(r1, c1), Shape::Matrix(r2, c2)) => {
            if c1 != r2 {
                return Err(shape_error(a.shape, b.shape));
            }
            // `row -> k -> col`, not `row -> col -> k` (phase 0a): accumulates into a whole
            // output row at a time, reading both `a` and `b` row-contiguously.
            //
            // Blocked over `row` and `k` when `b` is big enough for it to matter
            // (docs/rust-production-cutover.md's follow-on optimization work): without blocking,
            // every output row re-streams the *entire* `b` matrix once (`k` ranges over all of
            // `c1`), so if `b` doesn't fit in cache, `b` gets re-fetched from memory `r1` times
            // over. Blocking caps how much of `b` needs to stay resident at once (one
            // `K_BLOCK`-row slab) and reuses it across `ROW_BLOCK` output rows before moving on.
            // Measured, not assumed: blocking unconditionally was a *regression* at this
            // codebase's actual small layer sizes (e.g. `dimension=784, hidden=16` - `b` is only
            // ~100KB, already cache-resident, so the extra block-boundary bookkeeping was pure
            // overhead - 14% slower), but a genuine 1.3x-2.1x win once `b` exceeds a few hundred
            // KB (this codebase's own larger matmuls, e.g. `accumulate_gradient_batch`'s
            // `delta_batch.T @ input_activation_batch` at `batch_size=512`, and more so at sizes
            // well beyond anything this codebase currently trains). `BLOCKING_THRESHOLD_BYTES`
            // is set comfortably below a typical machine's L2 cache size, so blocking only
            // engages once there's real cache pressure for it to relieve. Either path produces
            // the same summation order per output row (k_block sweeps 0..c1 in increasing order,
            // and k sweeps increasing within each block, same as the unblocked loop), so results
            // are bit-identical, not just float64-close, regardless of which path runs.
            const ROW_BLOCK: usize = 64;
            const K_BLOCK: usize = 64;
            const BLOCKING_THRESHOLD_BYTES: usize = 256 * 1024;
            let b_size_bytes = c1 * c2 * std::mem::size_of::<f64>();
            let mut out = vec![0.0; r1 * c2];
            if b_size_bytes <= BLOCKING_THRESHOLD_BYTES {
                for row in 0..r1 {
                    let out_row = &mut out[row * c2..(row + 1) * c2];
                    for k in 0..c1 {
                        let a_value = a.data[row * c1 + k];
                        let b_row = &b.data[k * c2..(k + 1) * c2];
                        for col in 0..c2 {
                            out_row[col] += a_value * b_row[col];
                        }
                    }
                }
                return Ok(RustArray::from_matrix(out, r1, c2));
            }
            let mut row_block_start = 0;
            while row_block_start < r1 {
                let row_block_end = (row_block_start + ROW_BLOCK).min(r1);
                let mut k_block_start = 0;
                while k_block_start < c1 {
                    let k_block_end = (k_block_start + K_BLOCK).min(c1);
                    for row in row_block_start..row_block_end {
                        let out_row = &mut out[row * c2..(row + 1) * c2];
                        for k in k_block_start..k_block_end {
                            let a_value = a.data[row * c1 + k];
                            let b_row = &b.data[k * c2..(k + 1) * c2];
                            for col in 0..c2 {
                                out_row[col] += a_value * b_row[col];
                            }
                        }
                    }
                    k_block_start = k_block_end;
                }
                row_block_start = row_block_end;
            }
            Ok(RustArray::from_matrix(out, r1, c2))
        }
        (a_shape, b_shape) => Err(shape_error(a_shape, b_shape)),
    }
}

fn shape_error(a_shape: Shape, b_shape: Shape) -> PyErr {
    PyValueError::new_err(format!(
        "cannot matmul arrays of shape {:?} and {:?}",
        a_shape, b_shape
    ))
}

#[pymethods]
impl RustArray {
    /// Exposed as Python's `@` operator (`__matmul__`), not a free function - every real call
    /// site (`self.W @ x`, `X @ self.W.T`, ...) uses `@` syntax, so the Rust binding matches it
    /// rather than requiring a rewrite to a `matmul(a, b)` call style.
    fn __matmul__(&self, other: &RustArray) -> PyResult<RustArray> {
        matmul(self, other)
    }
}

/// The full pairwise-product matrix of two 1D vectors - `accumulate_gradient`'s own
/// `np.outer(delta, input_layer.a)`. Exposed as a free function (`perceptron_array.outer(a, b)`),
/// matching `np.outer`'s own call style rather than an operator.
#[pyfunction]
pub fn outer(a: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    match (a.shape, b.shape) {
        (Shape::Vector(m), Shape::Vector(n)) => {
            let mut out = Vec::with_capacity(m * n);
            for &a_value in &a.data {
                for &b_value in &b.data {
                    out.push(a_value * b_value);
                }
            }
            Ok(RustArray::from_matrix(out, m, n))
        }
        (a_shape, b_shape) => Err(PyValueError::new_err(format!(
            "outer requires two 1D vectors, got shapes {:?} and {:?}",
            a_shape, b_shape
        ))),
    }
}
