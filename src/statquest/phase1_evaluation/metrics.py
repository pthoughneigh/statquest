import numpy as np
import pandas as pd

Vector = np.ndarray | pd.Series


def confusion_matrix(y_true: Vector, y_pred: Vector, classes: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(classes), len(classes)), dtype="int64")
    class_to_index = {name: index for index, name in enumerate(classes)}

    for actual, predicted in zip(y_true, y_pred, strict=True):
        matrix[class_to_index[actual], class_to_index[predicted]] += 1

    return matrix


def per_class_recall(conf_matrix: np.ndarray) -> np.ndarray:
    diagonal = np.diag(conf_matrix)
    row_sum = conf_matrix.sum(axis=1)
    return diagonal / row_sum


def per_class_precision(conf_matrix: np.ndarray) -> np.ndarray:
    diagonal = np.diag(conf_matrix)
    column_sum = conf_matrix.sum(axis=0)
    column_sum = np.where(column_sum != 0, column_sum, np.nan)
    return diagonal / column_sum


def per_class_specificity(conf_matrix: np.ndarray) -> np.ndarray:
    TN = (
        conf_matrix.sum()
        - conf_matrix.sum(axis=0)
        - conf_matrix.sum(axis=1)
        + np.diag(conf_matrix)
    )
    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    return TN / (TN + FP)


def per_class_f1(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    denom = precision + recall
    # The guard covers two different classes, and returns 0 for both.
    # denom == 0: the name was used but never correctly, so P = R = 0. F1 is 0,
    # not undefined: as 2TP / (2TP + FP + FN) the denominator is FP + FN > 0.
    # denom is nan: the column is empty, so precision is nan from
    # per_class_precision. nan > 0 is False, so this branch takes it too, and 0
    # is again the right answer: TP = FP = 0 leaves the support in the
    # denominator. Macro precision stays nan there, macro F1 does not.
    return np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom > 0,
    )


def summarize_matrix(conf_matrix: np.ndarray) -> dict:
    total = conf_matrix.sum()
    if total == 0:
        raise ValueError("Empty confusion matrix")

    accuracy = np.trace(conf_matrix) / total

    precision = per_class_precision(conf_matrix)
    recall = per_class_recall(conf_matrix)

    f1 = per_class_f1(precision, recall)

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }
