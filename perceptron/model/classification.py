from typing import Sequence


def argmax_first_occurrence(values: Sequence[float]) -> int:
    """
    The index of the largest value, ties broken by first occurrence (numpy's own np.argmax
    convention) - shared by every classify_state() that picks the highest-probability class:
    MultiClassBackpropClassifierNetwork, EnsembleBackpropClassifierNetwork.
    """
    return max(range(len(values)), key=lambda i: values[i])
