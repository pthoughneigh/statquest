import numpy as np


def generate_yields(
    doses: np.ndarray, a: float, b: float, sigma: float, seed: int
) -> np.ndarray:
    return (
        a * doses
        + b
        + np.random.default_rng(seed).normal(loc=0, scale=sigma, size=doses.shape[0])
    )


if __name__ == "__main__":
    x = np.repeat([1, 2, 3, 4, 5], 200)
    a = 2
    b = 1
    y = generate_yields(doses=x, a=a, b=b, sigma=1, seed=42)

    noise = y - (a * x + b)
    assert abs(np.mean(noise)) < 0.1
    assert abs(np.std(noise, ddof=1) - 1) < 0.1
