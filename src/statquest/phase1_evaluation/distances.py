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


def hamming_distance(a: Vector, b: Vector) -> int:
    return (a != b).sum()


def manhattan_distance(a: Vector, b: Vector) -> float:
    return np.sum(np.abs(a - b))


def gower(data_frame: pd.DataFrame, a: Vector, b: Vector):
    score, n = 0.0, 0
    nominal, ordinal, rest = [], [], []
    for i, col in enumerate(data_frame.columns):
        x, y = a[i], b[i]
        if pd.isna(x) or pd.isna(y):
            continue
        if col in NOMINAL:
            nominal.append(col)
            score += 0.0 if x == y else 1.0
        elif col in ORDINAL:
            ordinal.append(col)
            rng = data_frame[col].max() - data_frame[col].min()
            if rng == 0:
                continue
            score += abs(x - y) / rng
        else:
            rest.append(col)
            continue
        n += 1
    return score / n if n else np.nan, nominal, ordinal, rest, n


if __name__ == "__main__":
    data = load_soyabeans_csv().drop("class", axis=1)

    r1 = np.asarray(data.iloc[1], dtype=float)
    r2 = np.asarray(data.iloc[301], dtype=float)

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
    score, nominal, ordinal, rest, n = gower(data, r1, r2)
    print(f"Distance gower: {score}")
    print(f"Nominal in gower: {nominal}")
    print(f"Ordinal in gower: {ordinal}")
    print(f"Rest in gower: {rest}")
    print(f"Number of columns: {n}")
