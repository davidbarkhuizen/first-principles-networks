def confusion_matrix(
    classifier,
    test_data: list[tuple[tuple[float, ...], int]],
    class_count: int,
) -> list[list[int]]:
    """
    matrix[true_label][predicted_label] = count. evaluate.py's disagreement/sampling functions
    don't generalize here - they're two-class- and geometry-specific, built to rejection-sample
    a continuously-sampleable target, whereas digit data is a fixed, finite, already-labeled
    set you iterate over directly.
    """

    matrix = [[0] * class_count for _ in range(class_count)]

    for state, true_label in test_data:
        predicted_label = classifier.classify_state(state)
        matrix[true_label][predicted_label] += 1

    return matrix


def accuracy(classifier, test_data: list[tuple[tuple[float, ...], int]]) -> float:
    """
    A public equivalent of train.py's private _training_accuracy - that one is private to
    train.py's own training loop; this runs standalone against a held-out test set once
    training has finished.
    """

    assert len(test_data) >= 1, "test_data must not be empty"

    correct = sum(1 for state, true_label in test_data if classifier.classify_state(state) == true_label)
    return correct / len(test_data)
