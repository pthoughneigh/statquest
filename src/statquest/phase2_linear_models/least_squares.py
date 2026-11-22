from pathlib import Path

import numpy as np


def sum_of_squared_residuals(y_true: np.ndarray, y_pred: np.ndarray | float) -> float:
    return float(np.pow((y_true - y_pred), 2).sum())


def predict_line(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


def fit_linear_regression_grid(
    x: np.ndarray, y: np.ndarray, slopes: np.ndarray
) -> tuple[float, float]:
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    sses = []

    for slope in slopes:
        intercept = y_mean - slope * x_mean
        y_pred = predict_line(x, slope, intercept)

        sses.append(sum_of_squared_residuals(y, y_pred))

    sses = np.array(sses)
    best_index = np.argmin(sses)

    assert best_index != 0, "Minimum is at the lower slope boundary."
    assert best_index != len(slopes) - 1, "Minimum is at the upper slope boundary."

    a_min = slopes[best_index]
    b_min = y_mean - a_min * x_mean

    return a_min, b_min


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    a = numerator / denominator

    b = y_mean - a * x_mean

    return a, b


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_mean = np.mean(y_true)

    ss_fit = sum_of_squared_residuals(y_true, y_pred)
    ss_mean = sum_of_squared_residuals(y_true, y_mean)

    return 1 - ss_fit / ss_mean


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    from statquest.phase2_linear_models.data_generator import generate_yields

    toy_x = np.array([1, 2, 3, 4, 5])
    toy_y = np.array([2, 4, 5, 4, 5])

    toy_slopes = np.arange(-0.5, 1.51, 0.01)

    for a, b, expected_sse in zip(
        [0, 0.5, 1, 0.6], [4, 2.5, 1, 2.2], [6, 2.5, 4, 2.4], strict=True
    ):
        y_pred = predict_line(toy_x, a, b)
        assert np.isclose(sum_of_squared_residuals(toy_y, y_pred), expected_sse), (
            f"Failed for a={a}, b={b}"
        )

    assert np.isclose(
        fit_linear_regression_grid(toy_x, toy_y, toy_slopes), (0.6, 2.2)
    ).all()
    assert np.isclose(r_squared(toy_y, predict_line(toy_x, 0.6, 2.2)), 0.6)
    assert np.isclose(r_squared(toy_y, predict_line(toy_x, 1.5, -0.5)), -0.75)

    doses = {
        "wide_doses": np.repeat([1, 2, 3, 4, 5], 3),
        "narrow_doses": np.repeat([2.8, 2.9, 3.0, 3.1, 3.2], 3),
    }

    results_per_dataset = {"wide_doses": [], "narrow_doses": []}

    wide_slopes = np.arange(-10.0, 14.0, 0.01)

    true_a = 2
    true_b = 1
    sigma = 1

    for dose, values in doses.items():
        for seed in range(1000):
            y = generate_yields(values, a=true_a, b=true_b, sigma=sigma, seed=seed)
            a_hat, _ = fit_linear_regression_grid(values, y, wide_slopes)
            results_per_dataset[dose].append(a_hat)

        dose_mean = np.mean(results_per_dataset[dose])
        dose_std = np.std(results_per_dataset[dose], ddof=1)

        assert np.isclose(
            dose_std,
            sigma / np.sqrt(sum_of_squared_residuals(values, values.mean())),
            rtol=0.02,
        )

        print(f"{dose}: ")
        print("Slopes mean: ", dose_mean)
        print("Slopes std: ", dose_std)

        print("-" * 100)

    a, b = fit_linear_regression(toy_x, toy_y)

    assert np.isclose(a, 0.6)
    assert np.isclose(b, 2.2)

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --------------------------------------------------
    # Left panel: Data, mean line and best-fit line
    # --------------------------------------------------

    y_mean = np.mean(toy_y)
    best_a, best_b = fit_linear_regression_grid(toy_x, toy_y, toy_slopes)

    best_line = predict_line(toy_x, best_a, best_b)

    axes[0].scatter(toy_x, toy_y, label="Observed values")
    axes[0].plot(
        toy_x,
        np.repeat(y_mean, len(toy_y)),
        label=f"Mean line (SSE={sum_of_squared_residuals(toy_y, y_mean)})",
    )
    axes[0].plot(
        toy_x,
        best_line,
        label=f"Best-fit line (SSE={np.round(sum_of_squared_residuals(toy_y, best_line), 2)})",
    )
    axes[0].vlines(
        toy_x, toy_y, best_line, linestyles="dashed", colors="C2", label="Residuals"
    )

    axes[0].set_xlabel("Doses")
    axes[0].set_ylabel("Yields")
    axes[0].set_title("Observed data and fitted line")
    axes[0].legend(loc="lower right")

    # --------------------------------------------------
    # Right panel: Sum of squared errors for each slope
    # --------------------------------------------------

    x_mean = np.mean(toy_x)

    sse_values = []

    for slope in toy_slopes:
        intercept = y_mean - slope * x_mean
        y_pred = predict_line(toy_x, slope, intercept)

        sse = sum_of_squared_residuals(toy_y, y_pred)
        sse_values.append(sse)

    sse_values = np.array(sse_values)

    axes[1].plot(toy_slopes, sse_values, label="SSE for each slope")

    # Three manually selected slopes
    axes[1].scatter([0, 0.5, 1], [6, 2.5, 4], label="Selected slopes")

    # Minimum SSE from the fitted model
    best_sse = sum_of_squared_residuals(toy_y, best_line)

    axes[1].scatter(
        best_a, best_sse, color="red", marker="*", s=150, label="Minimum SSE"
    )

    axes[1].set_xlabel("Slope (a)")
    axes[1].set_ylabel("Sum of squared errors (SSE)")
    axes[1].set_title("SSE as a function of slope")
    axes[1].legend()

    # --------------------------------------------------
    # Save and display figure
    # --------------------------------------------------
    plots_path = Path(__file__).parents[3] / "figures"
    plots_path.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(
        plots_path / "least_squares_visualization.png", dpi=300, bbox_inches="tight"
    )
