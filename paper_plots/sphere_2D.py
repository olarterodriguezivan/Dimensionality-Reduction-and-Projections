import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Any

import numpy as np
from pathlib import Path
from ioh import get_problem
from scipy.stats.qmc import LatinHypercube, scale



plt.rcParams.update({
    "legend.fontsize": 9,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "pgf.rcfonts": False,
})


# -----------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------

COLORMAP = "cividis" # or cividis or viridis_r or summer
MARKER_COLOR = "black"

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

FORMAT = "pdf"  # or "pdf"

FUNCTION_ID = 5
INSTANCE_ID = 1


# -----------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------

def uplift_points(points: np.ndarray, angle: float) -> np.ndarray:
    """
    Uplifts 1D points to 2D along a line defined by an angle.
    """

    if points.ndim != 2 or points.shape[1] != 1:
        raise ValueError("Points must have shape (n_samples,1)")

    direction = np.array([[np.cos(angle), np.sin(angle)]])
    return points @ direction


def base_contour(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, ax: Axes,
                 set_ylabel: bool = True) -> Any:
    """Draw contour background."""
    contour = ax.contourf(X, Y, Z, levels=25, cmap=COLORMAP)

    ax.set_xlabel(r"$x_1$")

    if set_ylabel:
        ax.set_ylabel(r"$x_2$")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")

    return contour


def make_full_sampling_plot(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, ax: Axes) -> Any:

    contour = base_contour(X, Y, Z, ax, set_ylabel=True)

    ax.set_title("Full Sampling")

    sampler = LatinHypercube(d=2, seed=42, optimization="lloyd")
    samples = scale(sampler.random(n=20), -5, 5)

    ax.scatter(samples[:, 0], samples[:, 1],
               color=MARKER_COLOR, marker="x", label="Samples")

    ax.legend()

    return contour


def make_two_slices_plot(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, ax: Axes, seed: int = 42):

    contour = base_contour(X, Y, Z, ax, set_ylabel=False)
    ax.set_title("$k$ Embeddings")

    rng = np.random.default_rng(seed)

    angles = rng.uniform(0, 2 * np.pi, 2)
    line_length = np.sqrt(200)
    colors = ["magenta", "cyan"]
    linestyles = ["--", "-."]

    for i, angle in enumerate(angles):

        x_line = np.linspace(-line_length * np.cos(angle),
                             line_length * np.cos(angle), 100)

        y_line = np.linspace(-line_length * np.sin(angle),
                             line_length * np.sin(angle), 100)

        points = LatinHypercube(d=1, seed=seed+i).random(n=10).flatten()
        points = points * 10 - 5

        points_2d = uplift_points(points[:, None], angle)

        ax.plot(x_line, y_line, linestyle=linestyles[i],
                    label=f"Embedding {i+1}",
                    color=colors[i])

        if i == 1:
            ax.scatter(points_2d[:, 0], points_2d[:, 1],
                   marker="x", color=MARKER_COLOR,
                   label="Samples")
        else:
            ax.scatter(points_2d[:, 0], points_2d[:, 1],
                   marker="x", color=MARKER_COLOR)

    ax.legend()

    return contour


def make_one_slice_plot(X, Y, Z, ax: Axes, seed: int = 42,
                        highlight: bool = False):

    contour = base_contour(X, Y, Z, ax, set_ylabel=False)
    ax.set_title("Single Embedding")

    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2*np.pi)

    line_length = np.sqrt(200)

    x_line = np.linspace(-line_length*np.cos(angle),
                         line_length*np.cos(angle), 100)

    y_line = np.linspace(-line_length*np.sin(angle),
                         line_length*np.sin(angle), 100)

    points = LatinHypercube(d=1, seed=seed).random(n=20).flatten()
    points = points * 10 - 5

    points_2d = uplift_points(points[:, None], angle)

    if highlight:
        ax.plot(x_line, y_line, color="red", linewidth=4.5)

    ax.plot(x_line, y_line, linestyle="--", color ="magenta", label="Embedding")

    ax.scatter(points_2d[:, 0], points_2d[:, 1],
               marker="x", color=MARKER_COLOR, label="Samples")

    ax.legend()

    return contour


def main():

    problem = get_problem(fid=FUNCTION_ID, instance=INSTANCE_ID, dimension=2)

    x = np.linspace(-5, 5, 100)
    y = x.copy()

    X, Y = np.meshgrid(x, y)

    inputs = np.column_stack([X.ravel(), Y.ravel()])
    Z = np.array([problem(p) for p in inputs]).reshape(X.shape)

    fig, axes = plt.subplots(1, 4, figsize=(14, 6), sharey=True)

    contour = make_full_sampling_plot(X, Y, Z, axes[0])
    make_two_slices_plot(X, Y, Z, axes[1])
    make_one_slice_plot(X, Y, Z, axes[2])
    make_one_slice_plot(X, Y, Z, axes[3], highlight=True)

    for ii, ax in enumerate(axes):
        if ii < 3:
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(2)

    fig.colorbar(
    contour,
    ax=axes,
    orientation="horizontal",
    location="top",
    pad=0.07,
    shrink=0.3,
    label=r"$f(\mathbf{x})$"
)

    #plt.tight_layout()
    plt.show()

    # Save the figure
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    fig.savefig(OUTPUT_DIR / f"sphere_2D_plot.{FORMAT}", transparent=True, dpi=300)

    fig.savefig(OUTPUT_DIR / f"sphere_2D_plot.pgf", transparent=True)

    #import tikzplotlib

    #tikzplotlib.save(OUTPUT_DIR / f"sphere_2D_plot.pgf")


if __name__ == "__main__":
    main()