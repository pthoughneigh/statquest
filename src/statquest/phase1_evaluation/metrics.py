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


def roc_curve(
    y_true_binary: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    negatives = (y_true_binary == 0).sum()
    positives = y_true_binary.sum()

    fpr = []
    tpr = []

    thresholds = np.concatenate(([np.inf], np.unique(scores)[::-1]))

    for t in thresholds:
        scores_mask = scores >= t

        TP = y_true_binary[scores_mask].sum()
        FP = (y_true_binary[scores_mask] == 0).sum()

        tpr.append(TP / positives)
        fpr.append(FP / negatives)

    return np.array(fpr), np.array(tpr), thresholds


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    widths = np.diff(fpr)
    heights = (tpr[:-1] + tpr[1:]) / 2

    return float(np.sum(widths * heights))


def auc_from_pairs(y_true_binary: np.ndarray, scores: np.ndarray) -> float:
    positive = y_true_binary == 1
    negative = y_true_binary == 0

    positive_scores = scores[positive][:, np.newaxis]
    negative_scores = scores[negative]

    wins = positive_scores > negative_scores
    ties = positive_scores == negative_scores

    return float((wins.sum() + ties.sum() / 2) / wins.size)
