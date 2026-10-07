import numpy as np


def train_test_split(
    n_rows: int, n_test: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    shuffled_positions = rng.permutation(n_rows)

    train_positions = shuffled_positions[n_test:]
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
        train_positions = np.concatenate(train_folds)
        test_positions = folds[i]

        train_test_pairs.append((train_positions, test_positions))
        assert np.array_equal(
            np.sort(np.concatenate((train_positions, test_positions))),
            np.arange(n_rows),
        )

    return train_test_pairs
