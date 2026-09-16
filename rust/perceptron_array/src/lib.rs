use pyo3::prelude::*;

mod array;
mod linalg;
mod ops;
mod ufuncs;

use array::RustArray;
use linalg::outer;
use ufuncs::{argmax, exp, sum_axis0};

/// Proves the PyO3/maturin toolchain works end to end - importable and callable from Python,
/// nothing array-specific yet. See docs/rust-array-core.md's "PR 0" for why this stage exists
/// on its own before any real array type is built.
#[pyfunction]
fn ping() -> PyResult<String> {
    Ok("pong".to_string())
}

#[pymodule]
fn perceptron_array(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(outer, m)?)?;
    m.add_function(wrap_pyfunction!(sum_axis0, m)?)?;
    m.add_function(wrap_pyfunction!(argmax, m)?)?;
    m.add_class::<RustArray>()?;
    Ok(())
}
