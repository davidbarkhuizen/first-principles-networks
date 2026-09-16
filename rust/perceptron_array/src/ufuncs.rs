use pyo3::prelude::*;

use crate::array::RustArray;

/// Elementwise `e^x` over a whole array - mirrors `array_layer.sigmoid`'s own reliance on
/// `exp`'s overflow behavior (a large negative `z` drives `exp(-z)` to `f64::INFINITY`, and
/// `1.0 / (1.0 + f64::INFINITY) == 0.0` under IEEE 754) instead of `math.exp`'s
/// `OverflowError`-raising behavior in the pure-Python reference. See
/// docs/vectorized-array-classes.md's own "numerical parity validation" - this is the one
/// operation with a documented overflow-boundary trap already flagged twice over, checked here
/// directly against `exp` before `sigmoid` itself is ever built on top of it.
#[pyfunction]
pub fn exp(arr: &RustArray) -> RustArray {
    RustArray {
        data: arr.data.iter().map(|value| value.exp()).collect(),
        shape: arr.shape,
    }
}
