import numpy as np
import pandas as pd

Vector = np.ndarray | pd.Series


def confusion_matrix(y_true: Vector, y_pred: Vector, classes: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(classes), len(classes)), dtype="int64")
    class_to_index = {name: index for index, name in enumerate(classes)}

    for actual, predicted in zip(y_true, y_pred, strict=True):
        matrix[class_to_index[actual], class_to_index[predicted]] += 1

    return matrix
