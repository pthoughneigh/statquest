from collections import defaultdict
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
    from statquest.phase1_evaluation.distances import make_gower

    data = load_soyabeans_csv()
    data_classes = data["class"]
    data_wo_classes = data.drop("class", axis=1)

    complete = data.notna().all(axis=1)
    data_classes = data_classes[complete]
    data_wo_classes = data_wo_classes[complete]

    gower = make_gower(data_wo_classes)
    support_per_class = defaultdict(int)
    correct_per_class = defaultdict(int)
    correct_predictions = 0
    shared_maximum = 0
    accurate_shared_maximum = 0

    for position in range(len(data_wo_classes)):
        class_actual = data_classes.iloc[position]
        support_per_class[class_actual] += 1
        plant_new = np.array(data_wo_classes.iloc[position])
        keep = np.arange(len(data_wo_classes)) != position

        _data_wo_classes = data_wo_classes.iloc[keep]
        _data_classes = data_classes.iloc[keep]

        assert len(_data_wo_classes) == len(_data_classes) == len(data_wo_classes) - 1

        class_predicted, vote_counts = knn(
            _data_wo_classes, _data_classes, plant_new, 5, gower
        )

        if (vote_counts == vote_counts.max()).sum() > 1:
            if class_predicted == class_actual:
                accurate_shared_maximum += 1
            shared_maximum += 1

        if class_predicted == class_actual:
            correct_per_class[class_actual] += 1
            correct_predictions += 1

    print(f"Correct predictions: {correct_predictions}")
    print()
    print(f"Support per class: {dict(support_per_class)}")
    print(f"Support total: {sum(support_per_class.values())}")
    print()
    print(f"Correct per class: {dict(correct_per_class)}")
    print(f"Correct total: {sum(correct_per_class.values())}")
    print()
    recall_per_class = {
        k: correct_per_class.get(k, 0) / support_per_class[k]
        for k in sorted(support_per_class, key=lambda k: (-support_per_class[k], k))
    }
    print("Recall per class (sorted by support, desc):")
    for cls, recall in recall_per_class.items():
        print(
            f"{cls:<25} support={support_per_class[cls]:>3}"
            f"  correct={correct_per_class.get(cls, 0):>3}"
            f"  recall={recall:.3f}"
        )
    print()
    print(f"Macro recall: {sum(recall_per_class.values()) / len(recall_per_class)}")
    print(f"Shared: {shared_maximum}")
    print(f"Accuracy on shared: {accurate_shared_maximum}")
