import numpy as np
import pandas as pd

from statquest.phase1_evaluation.distances import make_gower
from statquest.phase1_evaluation.knn import rank_neighbours, vote
from statquest.phase1_evaluation.splits import stratified_k_fold_split


def hits_per_k(
    features: pd.DataFrame,
    classes: pd.Series,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    max_k: int,
) -> np.ndarray:
    hits = np.zeros(max_k, dtype=int)

    for train, test in pairs:
        train_features = features.iloc[train]
        train_classes = classes.iloc[train]

        gower = make_gower(train_features)

        for test_position in test:
            plant = features.iloc[test_position].to_numpy()
            actual_class = classes.iloc[test_position]

            ranked_classes = rank_neighbours(
                train_features, train_classes, plant, gower
            )

            for k in range(1, max_k + 1):
                predicted_class, _ = vote(ranked_classes, k)
                if predicted_class == actual_class:
                    hits[k - 1] += 1
    return hits


if __name__ == "__main__":
    pairs = [(np.arange(266), np.arange(266))]

    from statquest.phase0_foundation.load_soyabeans import load_soyabeans_csv

    data = load_soyabeans_csv()

    all_classes = data["class"]
    all_features = data.drop("class", axis=1)

    complete_rows = all_features.notna().all(axis=1)

    classes = all_classes[complete_rows]
    features = all_features[complete_rows]

    print("One pair list: ")
    print(hits_per_k(features, classes, pairs, 50)[4])

    print("Stratified K-Fold: ")
    for seed in range(10):
        pairs = stratified_k_fold_split(classes, 10, seed)
        print(f"seed: {seed} - {hits_per_k(features, classes, pairs, 50)[4]}")
