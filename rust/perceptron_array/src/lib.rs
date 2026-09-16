use pyo3::prelude::*;

mod array;

use array::RustArray;

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
    m.add_class::<RustArray>()?;
    Ok(())
}
