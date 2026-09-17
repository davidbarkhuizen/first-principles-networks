# theory

[← back to README](../../README.md)

## external references

- Shree Nayar, Computer Science Dept, School of Engineering & Applied Sciences, Columbia
  University — https://fpcv.cs.columbia.edu/
- First Principles of Computer Vision Course —
  https://fpcv.cs.columbia.edu/, https://www.youtube.com/@firstprinciplesofcomputerv3258
- Perceptron | Neural Networks — https://www.youtube.com/watch?v=OFbnpY_k7js

## example configuration (NAND gate)

    w = [-2, 2]
    b = 3

## summary of perceptron theory, per Rosenblatt (1958)

Given:

- input `x_`: a vector of `n` inputs, `x1 .. xn`
- input weights `w_`: a vector, one weight per input, `w1 .. wn`
- bias `b` (activation threshold): a scalar

the activation function `f` is

    f(w_.x_) = 0  iff  w_.x_ <= -b
    f(w_.x_) = 1  iff  w_.x_ >  -b

Defining `z = w_.x_ + b`, the activation `a` is a function of `z`:

    a = f(z) = 0  iff  z <= 0
    a = f(z) = 1  iff  z >  0

i.e. the perceptron neuron's activation function is a step function.

For a single neuron, Rosenblatt's perceptron convergence theorem guarantees the update rule
above finds a separating hyperplane in a finite number of steps, provided the training data
is linearly separable. See [structure](structure.md) for how this composes into
`LinearClassifierNetwork` and what changes once `cardinality > 1`.
