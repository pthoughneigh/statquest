from collections.abc import Callable

import numpy as np
import pandas as pd

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
    from statquest.phase1_evaluation.distances import (
        # hamming_distance,
        # manhattan_distance,
        euclidean_distance,
        make_gower,
    )
    from statquest.phase1_evaluation.metrics import confusion_matrix

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    data = load_soyabeans_csv()
    data_classes_per_row = data["class"]
    data_wo_classes = data.drop("class", axis=1)

    complete_rows = data_wo_classes.notna().all(axis=1)
    data_classes_per_row_complete_rows = data_classes_per_row[complete_rows]
    data_wo_classes_complete_rows = data_wo_classes[complete_rows]

    gower = make_gower(data_wo_classes_complete_rows)
    distance = euclidean_distance

    actual_classes = []
    predicted_classes = []
    for position in range(len(data_wo_classes_complete_rows)):
        class_actual = data_classes_per_row_complete_rows.iloc[position]
        test_plant = np.array(data_wo_classes_complete_rows.iloc[position])

        keep = np.arange(len(data_wo_classes_complete_rows)) != position

        data_wo_classes_complete_rows_keep = data_wo_classes_complete_rows[keep]
        data_classes_per_row_complete_rows_keep = data_classes_per_row_complete_rows[
            keep
        ]

        assert (
            len(data_wo_classes_complete_rows_keep)
            == len(data_classes_per_row_complete_rows_keep)
            == len(data_wo_classes_complete_rows) - 1
        )

        class_predicted, _ = knn(
            data_wo_classes_complete_rows_keep,
            data_classes_per_row_complete_rows_keep,
            test_plant,
            5,
            distance,
        )

        actual_classes.append(class_actual)
        predicted_classes.append(class_predicted)

    unique_classes = np.unique(data_classes_per_row_complete_rows)
    matrix = confusion_matrix(
        np.array(actual_classes), np.array(predicted_classes), unique_classes
    )

    num_of_cases_per_class = (
        data_classes_per_row_complete_rows.value_counts()
        .sort_index(ascending=True)
        .to_numpy()
    )
    num_of_cases_per_row = np.sum(matrix, axis=1)
    print(f"Confusion matrix: \n{matrix}")
    print(f"Sum of all cells: {np.sum(matrix)}")
    print(f"Sum of diagonal elements: {np.trace(matrix)}")
    print(f"Sum through rows: {num_of_cases_per_row}")
    print(f"Number of cases per class: {num_of_cases_per_class}")
    print(
        "sum through rows == number of cases per class:",
        np.array_equal(num_of_cases_per_class, num_of_cases_per_row),
    )
    print(matrix.shape)
