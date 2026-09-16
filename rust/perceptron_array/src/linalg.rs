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
fn matmul(a: &RustArray, b: &RustArray) -> PyResult<RustArray> {
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
            // `row -> k -> col`, not `row -> col -> k`: accumulates into a whole output row at a
            // time, reading both `a` and `b` row-contiguously (the previous order read `b` with a
            // stride-`c2` access on the innermost loop - see docs/rust-production-cutover.md's
            // phase 0a for the measured cost of that).
            let mut out = vec![0.0; r1 * c2];
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
