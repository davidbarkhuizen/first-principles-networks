import pytest

from indrajala_ml.digits_data import load_digits_dataset, split_train_test


def test_load_digits_dataset_returns_the_full_bundled_dataset():

    dataset = load_digits_dataset()

    assert len(dataset) == 1797
    assert all(len(state) == 64 for state, _ in dataset)
    assert all(0.0 <= value <= 1.0 for state, _ in dataset for value in state)
    assert all(0 <= label <= 9 for _, label in dataset)

    # not every state is the same, and not every pixel is at the normalized extreme - a
    # sanity check that parsing/normalization actually did something, not just returned zeros
    assert len(set(label for _, label in dataset)) == 10
    assert any(value not in (0.0, 1.0) for state, _ in dataset for value in state)


def test_split_train_test_sizes_and_no_overlap():

    dataset = load_digits_dataset()

    train, test = split_train_test(dataset, test_fraction=0.2, seed=0)

    assert len(train) + len(test) == len(dataset)
    assert len(test) == round(len(dataset) * 0.2)

    # states are unique enough (64-dim real pixel data) that this reliably detects overlap
    train_states = set(state for state, _ in train)
    test_states = set(state for state, _ in test)
    assert train_states.isdisjoint(test_states)


def test_split_train_test_is_reproducible_under_a_fixed_seed():

    dataset = load_digits_dataset()

    train_a, test_a = split_train_test(dataset, test_fraction=0.2, seed=42)
    train_b, test_b = split_train_test(dataset, test_fraction=0.2, seed=42)

    assert train_a == train_b
    assert test_a == test_b


def test_split_train_test_rejects_a_non_fractional_test_fraction():

    dataset = load_digits_dataset()

    with pytest.raises(AssertionError):
        split_train_test(dataset, test_fraction=0.0)

    with pytest.raises(AssertionError):
        split_train_test(dataset, test_fraction=1.0)
