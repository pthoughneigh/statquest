from collections.abc import Callable

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

    summary = pd.DataFrame(rows).set_index("distance")
    print(summary.round(4))
