from collections.abc import Callable
from matplotlib import pyplot as plt
from pathlib import Path

import numpy as np
import pandas as pd

from statquest.phase1_evaluation.distances import (
    euclidean_distance,
    hamming_distance,
    make_gower,
    manhattan_distance,
)

Vector = np.ndarray | pd.Series


def knn(
    data_frame: pd.DataFrame,
    classes: Vector,
    plant: Vector,
    k: int,
    distance_function: Callable[[Vector, Vector], float],
) -> tuple[str, pd.Series]:
    rows = np.asarray(data_frame, dtype=float)

    distances = np.array([distance_function(plant, row) for row in rows])
    argsort_distances_positions = np.argsort(distances, kind="stable")[:k]
    nearest_classes = classes.iloc[argsort_distances_positions]

    nc_value_counts = nearest_classes.value_counts()

    tied = nc_value_counts.index[nc_value_counts == nc_value_counts.max()]
    nearest_neighbours = nearest_classes[nearest_classes.isin(tied)].iloc[0]

    return nearest_neighbours, nc_value_counts


if __name__ == "__main__":
    from statquest.phase0_foundation.load_soyabeans import load_soyabeans_csv
    from statquest.phase1_evaluation.metrics import (
        auc,
        auc_from_pairs,
        confusion_matrix,
        roc_curve,
        summarize_matrix,
        per_class_specificity,
        per_class_recall
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    data = load_soyabeans_csv()
    data_classes_per_row = data["class"]
    data_wo_classes = data.drop("class", axis=1)

    complete_rows = data_wo_classes.notna().all(axis=1)
    data_classes_per_row_complete_rows = data_classes_per_row[complete_rows]
    data_wo_classes_complete_rows = data_wo_classes[complete_rows]

    gower = make_gower(data_wo_classes_complete_rows)

    unique_classes = np.unique(data_classes_per_row_complete_rows)
    k = 5

    distances = {
        "gower": gower,
        "manhattan_distance": manhattan_distance,
        "hamming_distance": hamming_distance,
        "euclidean_distance": euclidean_distance,
    }

    rows = []
    scores_per_distance = {}

    for dist_name, dist in distances.items():
        actual_classes = []
        predicted_classes = []
        score_rows = []

        for position in range(len(data_wo_classes_complete_rows)):
            class_actual = data_classes_per_row_complete_rows.iloc[position]
            test_plant = np.array(data_wo_classes_complete_rows.iloc[position])

            keep = np.arange(len(data_wo_classes_complete_rows)) != position

            data_wo_classes_complete_rows_keep = data_wo_classes_complete_rows[keep]
            data_classes_per_row_complete_rows_keep = (
                data_classes_per_row_complete_rows[keep]
            )

            assert (
                len(data_wo_classes_complete_rows_keep)
                == len(data_classes_per_row_complete_rows_keep)
                == len(data_wo_classes_complete_rows) - 1
            )

            class_predicted, votes = knn(
                data_wo_classes_complete_rows_keep,
                data_classes_per_row_complete_rows_keep,
                test_plant,
                k,
                dist,
            )
            score_row = (votes.reindex(unique_classes, fill_value=0) / k).to_numpy()
            score_rows.append(score_row)

            actual_classes.append(class_actual)
            predicted_classes.append(class_predicted)

        scores = np.array(score_rows)

        actual = np.array(actual_classes)
        class_aucs = []

        for i, uclass in enumerate(unique_classes):
            y_true_binary = actual == uclass
            fpr, tpr, _ = roc_curve(y_true_binary, scores[:, i])

            auc_from_roc = auc(fpr, tpr)
            auc_from_scores = auc_from_pairs(y_true_binary, scores[:, i])
            assert np.isclose(auc_from_roc, auc_from_scores), (
                f"AUC from ROC and AUC from scores are not the same values on class "
                f"'{uclass}': {auc_from_roc} =/= {auc_from_scores}"
            )

            class_aucs.append(auc_from_roc)

        matrix = confusion_matrix(actual, np.array(predicted_classes), unique_classes)

        rows.append(
            {
                "distance": dist_name,
                **summarize_matrix(matrix),
                "macro_auc": np.mean(class_aucs),
            }
        )

        scores_per_distance[dist_name] = {"scores": scores, "actual": actual, "matrix": matrix}

    gower_results = scores_per_distance['gower']
    matrix = gower_results['matrix']
    class_to_check = ['brown-spot', 'frog-eye-leaf-spot']

    plots_path = Path(__file__).parents[3] / "figures"
    plots_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(nrows=1, ncols=len(class_to_check), figsize=(24, 8))

    for i, class_name in enumerate(class_to_check):
        y_true_binary = gower_results['actual'] == class_name
        class_index = np.where(unique_classes == class_name)[0][0]
        class_scores_column = gower_results['scores'][:, class_index]

        fpr, tpr, _ = roc_curve(y_true_binary, class_scores_column)
        knn_fpr = 1 - per_class_specificity(matrix)[class_index]
        knn_tpr = per_class_recall(matrix)[class_index]
        auc_from_roc = auc(fpr, tpr)

        ax[i].plot(fpr, tpr, marker="o", label="ROC curve")
        ax[i].plot([0, 1], [0, 1], "r--", label="Random classifier")
        ax[i].scatter(knn_fpr, knn_tpr, marker="x", s=100, color="green", label='KNN (Gower)')
        ax[i].set_title(f"'{class_name}' ROC curve - AUC: {auc_from_roc:.4f}")
        ax[i].set_xlabel("FPR (1 − specificity)")
        ax[i].set_ylabel("TPR (recall)")
        ax[i].set_aspect("equal")

        offsets = [
            (8, -5),
            (8, -5),
            (5, -5),
            (5, -6),
            (8, -8),
            (8, 5),
            (-44, 5)
        ]

        for (x, y), offset in zip(zip(fpr, tpr), offsets):
            ax[i].annotate(
                f"({x:.2f}, {y:.2f})",
                (x, y),
                xytext=offset,
                textcoords="offset points"
            )

        ax[i].annotate(
            f"KNN (Gower)\n({knn_fpr:.2f}, {knn_tpr:.2f})",
            xy=(knn_fpr, knn_tpr),
            xytext=(-80, -10),
            textcoords="offset points",
            fontsize=10,
            arrowprops=dict(arrowstyle="->")
        )

        ax[i].legend()

    fig.savefig(plots_path / "gower_knn_per_class_roc.png", dpi=300, bbox_inches="tight")

    summary = pd.DataFrame(rows).set_index("distance")
    print(summary.round(4))
