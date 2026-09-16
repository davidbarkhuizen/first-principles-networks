use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;

/// This core only ever needs a 1D vector or a 2D matrix - see docs/numpy-interface-subset.md's
/// own "dtype and shape" section for why general N-dimensional machinery is deliberately not
/// built here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Vector(usize),
    Matrix(usize, usize),
}

impl Shape {
    fn size(&self) -> usize {
        match *self {
            Shape::Vector(n) => n,
            Shape::Matrix(rows, cols) => rows * cols,
        }
    }
}

fn parse_shape(shape: &PyAny) -> PyResult<Shape> {
    if let Ok((rows, cols)) = shape.extract::<(usize, usize)>() {
        Ok(Shape::Matrix(rows, cols))
    } else if let Ok(n) = shape.extract::<usize>() {
        Ok(Shape::Vector(n))
    } else {
        Err(PyTypeError::new_err(
            "shape must be an int (1D) or a (rows, cols) tuple (2D)",
        ))
    }
}

/// One layer's weights/activations/gradients as a flat, row-major f64 buffer plus a shape tag -
/// the Rust-side counterpart to a real numpy `ndarray` restricted to exactly
/// docs/numpy-interface-subset.md's own table. Named `Array`, not `PyArray`, to avoid colliding
/// with real numpy's own type of that name (see docs/rust-array-core.md's "crate structure").
#[pyclass(name = "Array")]
#[derive(Clone)]
pub struct RustArray {
    pub data: Vec<f64>,
    pub shape: Shape,
}

impl RustArray {
    pub fn from_vector(data: Vec<f64>) -> Self {
        let shape = Shape::Vector(data.len());
        RustArray { data, shape }
    }

    pub fn from_matrix(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        RustArray {
            data,
            shape: Shape::Matrix(rows, cols),
        }
    }
}

#[pymethods]
impl RustArray {
    /// Mirrors `np.array(data)`: a flat Python list of floats builds a 1D array, a nested list
    /// of same-length lists builds a 2D array - the two shapes this whole core ever needs, no
    /// more (see docs/numpy-interface-subset.md's own "construct from data" row).
    #[new]
    fn new(data: &PyAny) -> PyResult<Self> {
        if let Ok(rows) = data.extract::<Vec<Vec<f64>>>() {
            let n_rows = rows.len();
            if n_rows == 0 {
                return Err(PyValueError::new_err(
                    "cannot construct a 2D array from zero rows",
                ));
            }
            let n_cols = rows[0].len();
            let mut flat = Vec::with_capacity(n_rows * n_cols);
            for row in &rows {
                if row.len() != n_cols {
                    return Err(PyValueError::new_err(
                        "every row must have the same length",
                    ));
                }
                flat.extend_from_slice(row);
            }
            Ok(RustArray::from_matrix(flat, n_rows, n_cols))
        } else if let Ok(values) = data.extract::<Vec<f64>>() {
            Ok(RustArray::from_vector(values))
        } else {
            Err(PyTypeError::new_err(
                "Array() expects a flat list of numbers (1D) or a nested list of same-length lists (2D)",
            ))
        }
    }

    #[staticmethod]
    fn zeros(shape: &PyAny) -> PyResult<Self> {
        let shape = parse_shape(shape)?;
        Ok(RustArray {
            data: vec![0.0; shape.size()],
            shape,
        })
    }

    #[getter]
    fn shape(&self, py: Python<'_>) -> PyObject {
        match self.shape {
            Shape::Vector(n) => (n,).into_py(py),
            Shape::Matrix(rows, cols) => (rows, cols).into_py(py),
        }
    }

    /// `arr[i]` for a 1D array, `arr[i, j]` for a 2D array - two index shapes through the same
    /// slot, matching how Python itself dispatches `arr[i]` vs. `arr[i, j]` (a single tuple
    /// argument) through `__getitem__`. See docs/numpy-interface-subset.md's "re-checked against
    /// the built implementation" note for why both shapes are required, not just the 1D one.
    fn __getitem__(&self, index: &PyAny) -> PyResult<f64> {
        let flat_index = self.resolve_index(index)?;
        Ok(self.data[flat_index])
    }

    fn __setitem__(&mut self, index: &PyAny, value: f64) -> PyResult<()> {
        let flat_index = self.resolve_index(index)?;
        self.data[flat_index] = value;
        Ok(())
    }

    fn copy(&self) -> Self {
        RustArray {
            data: self.data.clone(),
            shape: self.shape,
        }
    }

    /// Reinterprets shape without changing data or order, matching `.reshape()`'s own contract
    /// (docs/numpy-interface-subset.md). Returns an independent array (a full copy of the data),
    /// not a numpy-style view sharing the original buffer - nothing in this core's required
    /// operation set relies on view-aliasing semantics (`load_mnist_dataset_as_array`'s own
    /// reshape-then-slice-then-astype chain already copies at the `.astype` step), so the
    /// simpler, correctness-preserving choice is made here rather than building view machinery
    /// nothing needs yet.
    fn reshape(&self, shape: &PyAny) -> PyResult<Self> {
        let new_shape = parse_shape(shape)?;
        if new_shape.size() != self.data.len() {
            return Err(PyValueError::new_err(format!(
                "cannot reshape array of size {} into shape of size {}",
                self.data.len(),
                new_shape.size()
            )));
        }
        Ok(RustArray {
            data: self.data.clone(),
            shape: new_shape,
        })
    }
}

impl RustArray {
    fn resolve_index(&self, index: &PyAny) -> PyResult<usize> {
        match self.shape {
            Shape::Vector(n) => {
                let i: usize = index.extract().map_err(|_| {
                    PyTypeError::new_err("index into a 1D array must be an int")
                })?;
                if i >= n {
                    return Err(PyIndexError::new_err("index out of range"));
                }
                Ok(i)
            }
            Shape::Matrix(rows, cols) => {
                let (row, col): (usize, usize) = index.extract().map_err(|_| {
                    PyTypeError::new_err("index into a 2D array must be a (row, col) tuple")
                })?;
                if row >= rows || col >= cols {
                    return Err(PyIndexError::new_err("index out of range"));
                }
                Ok(row * cols + col)
            }
        }
    }
}
