from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from statquest.phase0_foundation.load_soyabeans import load_soyabeans_csv
from statquest.phase1_evaluation.metrics import (
    confusion_matrix,
    per_class_precision,
    per_class_recall,
)


def majority_class_baseline(classes: pd.Series) -> dict:
    """Always predict the most frequent class; score it from the confusion matrix."""
    majority_class = classes.value_counts().idxmax()
    # From the same series that is scored, so every class in the matrix has
    # at least one row and recall is never 0/0.
    unique_classes = np.unique(classes)

    predictions = np.full(len(classes), majority_class)
    cm = confusion_matrix(classes, predictions, unique_classes)
    # A matrix that lost rows would still give a plausible accuracy, because
    # numerator and denominator would come from the same broken matrix.
    assert cm.sum() == len(classes)

    # Macro recall equals 1 / n_classes here only because one class gets
    # recall 1 and every other class 0; for any other model it would not.
    recall = per_class_recall(cm)
    macro_precision = per_class_precision(cm).mean()
    accuracy = np.trace(cm) / cm.sum()
    macro_recall = recall.mean()

    return {
        "rows": len(classes),
        "classes": len(unique_classes),
        "majority_class": majority_class,
        "accuracy": accuracy,
        "precision": macro_precision,
        "macro_recall": macro_recall,
    }


if __name__ == "__main__":
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    plots_path = Path(__file__).parents[3] / "figures"
    plots_path.mkdir(parents=True, exist_ok=True)

    data = load_soyabeans_csv()
    class_column_value_counts = data["class"].value_counts()
    print(f"\nColumn 'class' value counts: \n{class_column_value_counts}\n")

    fig, ax = plt.subplots(figsize=(25, 6))
    b = ax.barh(class_column_value_counts.index, class_column_value_counts)
    ax.set_title("Disease distribution")
    ax.set_xlabel("Number of cases")
    ax.set_ylabel("Disease")
    ax.bar_label(b)
    fig.savefig(plots_path / "baseline.png", bbox_inches="tight")

    # Rows with no missing feature values.
    complete_rows = data.drop("class", axis=1).notna().all(axis=1)
    complete_classes = data.loc[complete_rows, "class"]

    summary = pd.DataFrame(
        [
            majority_class_baseline(data["class"]),
            majority_class_baseline(complete_classes),
        ],
        index=["all rows", "complete rows"],
    )
    print(f"Majority-class baselines:\n{summary.round(4)}")
