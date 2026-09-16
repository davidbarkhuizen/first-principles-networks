import random

from indrajala_ml.digits_data import load_digits_dataset, split_train_test
from indrajala_ml.model.softmax_multiclass_backprop_classifier_network import (
    SoftmaxMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.multiclass_evaluate import accuracy
from indrajala_ml.train import train_linear_classifier_network


def test_train_linear_classifier_network_drives_softmax_multiclass_backprop_on_real_digit_data():

    # the direct softmax/cross-entropy counterpart of
    # test_multiclass_training_pipeline.py::test_train_linear_classifier_network_drives_multiclass_backprop_on_real_digit_data
    # - same 200-row subset, same seeds, same architecture/learning_rate/epochs, so the two
    # loss functions' actual difference on real data is directly comparable, not just each
    # measured in isolation. Measured directly (not guessed): best_training_accuracy=1.0 at
    # epoch 15/15 (genuinely converged, not just plateaued), test accuracy 0.975 - both higher
    # than the one-vs-rest sibling's 0.98125 training / 0.9 test on this exact same split. Not
    # a claim that softmax is always better (200 rows is a small, easy-to-overfit sample), just
    # a real, reproducible measurement that it isn't worse here, and comes with the semantic
    # correctness benefit (probabilities that sum to 1) for free.
    random.seed(0)

    dataset = load_digits_dataset()
    subset = dataset[:200]
    train_data, test_data = split_train_test(subset, test_fraction=0.2, seed=1)

    student = SoftmaxMultiClassBackpropClassifierNetwork.randomized([16], 64, [(0.0, 1.0)] * 64, 10)
    result = train_linear_classifier_network(student, train_data, learning_rate=0.5, epochs=15)

    diagnostic = result.diagnostic
    assert diagnostic.best_training_accuracy == 1.0
    assert diagnostic.best_epoch_index == 14
    assert diagnostic.plateaued is False
    assert diagnostic.converged is True
    assert diagnostic.still_improving is False

    assert accuracy(student, test_data) == 0.975
