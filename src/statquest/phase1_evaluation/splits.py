import numpy as np


def train_test_split(
    n_rows: int, n_test: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    shuffled_positions = rng.permutation(n_rows)

    train_positions = np.sort(shuffled_positions[n_test:])
    test_positions = shuffled_positions[:n_test]

    assert np.intersect1d(test_positions, train_positions).size == 0
    assert np.array_equal(
        np.sort(np.concatenate((test_positions, train_positions))), np.arange(n_rows)
    )

    return train_positions, test_positions


def k_fold_split(
    n_rows: int, n_folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    train_test_pairs = []

    rng = np.random.default_rng(seed)

    shuffled_positions = rng.permutation(n_rows)

    folds = np.array_split(shuffled_positions, n_folds)

    assert np.array_equal(np.sort(np.concatenate(folds)), np.arange(n_rows))

    for i in range(n_folds):
        train_folds = folds[:i] + folds[i + 1 :]
        train_positions = np.sort(np.concatenate(train_folds))
        test_positions = folds[i]

        train_test_pairs.append((train_positions, test_positions))
        assert np.array_equal(
            np.sort(np.concatenate((train_positions, test_positions))),
            np.arange(n_rows),
        )

    return train_test_pairs


def stratified_k_fold_split(
    classes: np.ndarray, n_folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    classes = np.asarray(classes)
    rng = np.random.default_rng(seed)

    training_test_pairs = []

    fold_of = np.full(len(classes), -1)
    for cls in np.unique(classes):
        class_positions = np.flatnonzero(classes == cls)
        class_pos_shuffled = rng.permutation(class_positions)

        fold_of_distribution = np.arange(len(class_pos_shuffled)) % n_folds
        fold_of[class_pos_shuffled] = fold_of_distribution

    assert (fold_of == -1).sum() == 0

    for f in range(n_folds):
        test_positions = np.flatnonzero(fold_of == f)
        train_positions = np.flatnonzero(fold_of != f)
        training_test_pairs.append((train_positions, test_positions))

    test_indices = [test_idx for _, test_idx in training_test_pairs]
    sorted_test_indices = np.sort(np.concatenate(test_indices))

    assert np.array_equal(sorted_test_indices, np.arange(len(classes)))

    for cls in np.unique(classes):
        bc = np.bincount(fold_of[np.flatnonzero(classes == cls)], minlength=n_folds)
        assert np.max(bc) - np.min(bc) <= 1

    return training_test_pairs
