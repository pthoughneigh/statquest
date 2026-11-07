from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from statquest.phase1_evaluation.distances import make_gower
from statquest.phase1_evaluation.knn import rank_neighbours, vote
from statquest.phase1_evaluation.splits import stratified_k_fold_split


def correct_predictions_per_k(
    features: pd.DataFrame,
    classes: pd.Series,
    train_test_pairs: list[tuple[np.ndarray, np.ndarray]],
    max_k: int,
) -> np.ndarray:
    correct_predictions = np.zeros(max_k, dtype=int)

    for train_positions, test_positions in train_test_pairs:
        train_features = features.iloc[train_positions]
        train_classes = classes.iloc[train_positions]

        gower = make_gower(train_features)

        for test_position in test_positions:
            test_plant = features.iloc[test_position].to_numpy()
            actual_class = classes.iloc[test_position]

            ranked_classes = rank_neighbours(
                train_features, train_classes, test_plant, gower
            )

            for k in range(1, max_k + 1):
                predicted_class, _ = vote(ranked_classes, k)
                if predicted_class == actual_class:
                    correct_predictions[k - 1] += 1

    return correct_predictions


if __name__ == "__main__":
    from statquest.phase0_foundation.load_soyabeans import load_soyabeans_csv

    data = load_soyabeans_csv()

    all_classes = data["class"]
    all_features = data.drop("class", axis=1)

    complete_rows = all_features.notna().all(axis=1)

    classes = all_classes[complete_rows]
    features = all_features[complete_rows]

    max_k = 50
    n_folds = 10
    seeds = range(10)

    datasets = {
        "set_307": (all_features, all_classes),
        "set_266": (features, classes),
    }

    results_per_dataset = {}

    for dataset_name, (dataset_features, dataset_classes) in datasets.items():
        n_rows = len(dataset_features)

        all_positions = np.arange(n_rows)
        resubstitution_pairs = [(all_positions, all_positions)]

        training_correct_predictions = correct_predictions_per_k(
            dataset_features, dataset_classes, resubstitution_pairs, max_k
        )
        validation_correct_predictions = np.array(
            [
                correct_predictions_per_k(
                    dataset_features,
                    dataset_classes,
                    stratified_k_fold_split(dataset_classes, n_folds, seed),
                    max_k,
                )
                for seed in seeds
            ]
        )

        training_errors = 1 - training_correct_predictions / n_rows
        validation_errors = 1 - validation_correct_predictions / n_rows

        mean_validation_errors = validation_errors.mean(axis=0)
        std_validation_errors = validation_errors.std(axis=0, ddof=1)

        results_per_dataset[dataset_name] = {
            "training_errors": training_errors,
            "validation_errors": validation_errors,
            "mean_validation_errors": mean_validation_errors,
            "std_validation_errors": std_validation_errors,
        }

        best_position = mean_validation_errors.argmin()
        print(
            f"{dataset_name}: lowest mean validation error "
            f"{mean_validation_errors[best_position]:.4f} at k = {best_position + 1}"
        )

    k = np.arange(1, max_k + 1)
    plots_path = Path(__file__).parents[3] / "figures"
    plots_path.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)

    for ax, dataset_name in zip(axes, ["set_266", "set_307"]):
        ax.plot(
            k,
            results_per_dataset[dataset_name]["training_errors"],
            label="training error",
        )
        ax.plot(
            k,
            results_per_dataset[dataset_name]["mean_validation_errors"],
            label="validation error (mean over seeds)",
        )
        ax.fill_between(
            k,
            results_per_dataset[dataset_name]["mean_validation_errors"]
            - results_per_dataset[dataset_name]["std_validation_errors"],
            results_per_dataset[dataset_name]["mean_validation_errors"]
            + results_per_dataset[dataset_name]["std_validation_errors"],
            alpha=0.3,
            label="validation ± 1 sd over seeds",
        )
        ax.set_title("266 complete rows" if dataset_name == "set_266" else "307 rows")
        ax.set_xlabel("k")

    axes[0].set_ylabel("error")
    axes[0].legend()

    fig.savefig(plots_path / "gower_knn_error_by_k.png", dpi=300, bbox_inches="tight")
