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

COLORMAP = "viridis" # or cividis or viridis_r or summer
MARKER_COLOR = "white"

EMBEDDING_1_COLOR = "gold"
EMBEDDING_2_COLOR = "cyan"

SPACE_OF_INTEREST_COLOR = "red"
SPACE_OF_INTEREST_LINEWIDTH = 2.5
LAST_SPACE_OF_INTEREST_LINEWIDTH = 7.5


SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

FORMAT = "pdf"  # or "pdf"

FUNCTION_ID = 16
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
    contour = ax.contourf(X, Y, Z, levels=25, cmap=COLORMAP, zorder=0)

    ax.set_xlabel(r"$x_1$")

    if set_ylabel:
        ax.set_ylabel(r"$x_2$")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")

    return contour


def make_full_sampling_plot(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, ax: Axes, seed: int = 42, 
                            title: str = "Full Sampling", set_ylabel: bool = True) -> Any:

    contour = base_contour(X, Y, Z, ax, set_ylabel=set_ylabel)

    ax.set_title(title)

    sampler = LatinHypercube(d=2, seed=seed, optimization="random-cd")
    samples = scale(sampler.random(n=20), -5, 5)

    samples_object = ax.scatter(samples[:, 0], samples[:, 1],
               color=MARKER_COLOR, marker="x", label="Samples", zorder=1)

    #ax.legend()

    return (contour, samples_object)


def make_two_slices_plot(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, ax: Axes, seed: int = 42,
                         title: str = "$K$ Embeddings", set_ylabel: bool = True)-> Any:

    contour = base_contour(X, Y, Z, ax, set_ylabel=set_ylabel)
    ax.set_title(title)

    rng = np.random.default_rng(seed)

    angles = rng.uniform(0, 2 * np.pi, 2)
    line_length = np.sqrt(200)
    colors = [EMBEDDING_1_COLOR, EMBEDDING_2_COLOR]
    linestyles = ["--", "-."]

    embedding_object_list = []
    points_object_list = []

    zorders = np.linspace(1, 2, len(angles))

    for i, angle in enumerate(angles):

        x_line = np.linspace(-line_length * np.cos(angle),
                             line_length * np.cos(angle), 100)

        y_line = np.linspace(-line_length * np.sin(angle),
                             line_length * np.sin(angle), 100)

        points = LatinHypercube(d=1, seed=seed+i).random(n=10).flatten()
        points = points * 10 - 5

        points_2d = uplift_points(points[:, None], angle)

        embedding_object = ax.plot(x_line, y_line, linestyle=linestyles[i],
                    label=f"Embedding {i+1}",
                    color=colors[i], zorder=zorders[i])

        if i == 1:
            points_object = ax.scatter(points_2d[:, 0], points_2d[:, 1],
                   marker="x", color=MARKER_COLOR,
                   label="Samples", zorder=zorders[i])
        else:
            points_object = ax.scatter(points_2d[:, 0], points_2d[:, 1],
                   marker="x", color=MARKER_COLOR, zorder=zorders[i])
        
        embedding_object_list.append(embedding_object)
        points_object_list.append(points_object)

    #ax.legend(facecolor="lightgray", framealpha=0.9)

    return (contour, embedding_object_list, points_object_list)


def make_one_slice_plot(X, Y, Z, ax: Axes, seed: int = 42,
                        highlight: bool = False, title: str = "Single Embedding", 
                        set_ylabel: bool = True) -> Any:

    contour = base_contour(X, Y, Z, ax, set_ylabel=set_ylabel)
    ax.set_title(title)

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

    
    interest_space = None

    if highlight:
        interest_space = ax.plot(x_line, y_line, color=SPACE_OF_INTEREST_COLOR, linewidth=LAST_SPACE_OF_INTEREST_LINEWIDTH, 
                                 zorder=1, label="ELA Computation Space")

    embedding_object = ax.plot(x_line, y_line, linestyle="--", color = EMBEDDING_1_COLOR, label="Embedding", zorder=2)

    points_object = ax.scatter(points_2d[:, 0], points_2d[:, 1],
               marker="x", color=MARKER_COLOR, label="Samples", zorder=3)


    #ax.legend(facecolor="lightgray", framealpha=0.9)
    if highlight:
        return (contour, embedding_object, points_object, interest_space)
    return (contour, embedding_object, points_object)


def main():

    problem = get_problem(fid=FUNCTION_ID, instance=INSTANCE_ID, dimension=2)

    x = np.linspace(-5, 5, 500)
    y = x.copy()

    X, Y = np.meshgrid(x, y)

    inputs = np.column_stack([X.ravel(), Y.ravel()])
    Z = np.array([problem(p) for p in inputs]).reshape(X.shape)

    fig, axes = plt.subplots(1, 4, figsize=(10, 10), sharey=True, sharex=True)



    contour, points_obj1 = make_full_sampling_plot(X, Y, Z, axes[0], title=r"\texttt{Full/ELA$_{\texttt{A}}$}")
    contour2, emb_obj1, points_obj2 = make_two_slices_plot(X, Y, Z, axes[1], title=r"\texttt{Sliced/ELA$_{\texttt{A}}$}")
    contour3, emb_obj2, points_obj3 = make_one_slice_plot(X, Y, Z, axes[2], title=r"\texttt{All\_in/ELA$_{\texttt{A}}$}", set_ylabel=True)
    contour4, emb_obj3, points_obj4, interest_space = make_one_slice_plot(X, Y, Z, axes[3], highlight=True, title=r"\texttt{All\_in/ELA$_{\texttt{R}}$}", set_ylabel=True)

    for ii, ax in enumerate(axes.flatten()):
        if ii < 3:
            for spine in ax.spines.values():
                spine.set_edgecolor(SPACE_OF_INTEREST_COLOR)
                spine.set_linewidth(SPACE_OF_INTEREST_LINEWIDTH)

    cbar = fig.colorbar(
    contour,
    ax=axes,
    orientation="horizontal",
    location="top",
    pad=0.07,
    shrink=0.6,
    label=r"$f(\mathbf{x})$")

    # Set the legend for the last subplot
    handles = [emb_obj1[0][0], emb_obj1[1][0], points_obj4, interest_space[0]]
    labels = ["Embedding 1", "Embedding 2", "Samples", "ELA Computation Space"]

    axes[1].legend(handles=handles, labels=labels, bbox_to_anchor=(-0.71, -0.25), loc='upper left', facecolor="gainsboro", framealpha=0.9,
                   edgecolor="black",fontsize=10, ncol=4)

    #plt.tight_layout()
    plt.show()

    # Save the figure
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    fig.savefig(OUTPUT_DIR / f"sphere_2D_plot.{FORMAT}", transparent=True, dpi=300, bbox_inches="tight")

    fig.savefig(OUTPUT_DIR / f"sphere_2D_plot.pgf", transparent=True, dpi=300, bbox_inches="tight")

    #import tikzplotlib

    #tikzplotlib.save(OUTPUT_DIR / f"sphere_2D_plot.pgf")


if __name__ == "__main__":
    main()