import numpy as np


def train_test_split(
    n_rows: int, n_test: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    shuffled_positions = rng.permutation(n_rows)

    test_positions = shuffled_positions[:n_test]
    train_positions = shuffled_positions[n_test:]

    assert np.intersect1d(test_positions, train_positions).size == 0
    assert np.array_equal(
        np.sort(np.concatenate((test_positions, train_positions))), np.arange(n_rows)
    )

    return train_positions, test_positions
