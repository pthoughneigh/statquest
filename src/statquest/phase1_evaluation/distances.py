from collections.abc import Callable

import numpy as np
import pandas as pd

from statquest.phase0_foundation.load_soyabeans import (
    NOMINAL,
    ORDINAL,
    load_soyabeans_csv,
)

Vector = np.ndarray | pd.Series


def euclidean_distance(a: Vector, b: Vector) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def hamming_distance(a: Vector, b: Vector) -> float:
    return (a != b).sum()


def manhattan_distance(a: Vector, b: Vector) -> float:
    return np.sum(np.abs(a - b))


def make_gower(data_frame: pd.DataFrame) -> Callable[[Vector, Vector], float]:
    data_frame_columns = data_frame.columns

    ordinal_ranges = {c: data_frame[c].max() - data_frame[c].min() for c in ORDINAL}

    def gower(a: Vector, b: Vector) -> float:
        score, n = 0.0, 0
        for i, col in enumerate(data_frame_columns):
            x, y = a[i], b[i]
            if pd.isna(x) or pd.isna(y):
                continue
            if col in NOMINAL:
                score += 0.0 if x == y else 1.0
            elif col in ORDINAL:
                rng = ordinal_ranges[col]
                if rng == 0:
                    continue
                score += abs(x - y) / rng
            else:
                continue
            n += 1
        return score / n if n else np.nan

    return gower


if __name__ == "__main__":
    data = load_soyabeans_csv().drop("class", axis=1)

    r1 = np.asarray(data.iloc[1], dtype=float)
    r2 = np.asarray(data.iloc[175], dtype=float)

    mask = ~(np.isnan(r1) | np.isnan(r2))

    print(f"Mask sum: {mask.sum()}")
    print()
    print(f"Distance without mask (hamming): {hamming_distance(r1, r2)}")
    print(f"Distance with mask (hamming): {hamming_distance(r1[mask], r2[mask])}")

    print()
    print(f"Distance without mask (euclidean_distance): {euclidean_distance(r1, r2)}")
    print(
        f"Distance with mask (euclidean_distance): {euclidean_distance(r1[mask], r2[mask])}"
    )

    print()
    print(f"Distance without mask (manhattan_distance): {manhattan_distance(r1, r2)}")
    print(
        f"Distance with mask (manhattan_distance): {manhattan_distance(r1[mask], r2[mask])}"
    )

    print()
    print(f"Distance gower: {make_gower(data)(r1, r2)}")
