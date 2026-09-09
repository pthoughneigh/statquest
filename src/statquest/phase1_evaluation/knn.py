from collections.abc import Callable

import numpy as np
import pandas as pd

Vector = np.ndarray | pd.Series


def knn(
    plant: Vector,
    data_frame: pd.DataFrame,
    k: int,
    distance_function: Callable[[Vector, Vector], float],
) -> str:
    pass
