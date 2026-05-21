"""
Visualisation code for manifolds.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap


def visualise_manifold(data: np.ndarray, n_components: int = 2) -> Isomap.fit_transform:
    """Visualise the manifold of the data using Isomap and return a projection."""
    iso = Isomap(n_components=n_components)
    projection = iso.fit_transform(data)
    return projection


def visualise_projection(
    projection: np.ndarray, target: np.ndarray, save_plot: str | None
) -> None:
    """Plot the projection of the data with a color map based on the target labels. If
    given a save directory, save the plot to that directory, otherwise show the plot"""
    plt.scatter(
        projection[:, 0],
        projection[:, 1],
        lw=0.1,
        c=target,
        cmap=plt.cm.get_cmap("cubehelix", len(np.unique(target))),
    )
    plt.colorbar(ticks=range(len(np.unique(target))), label="digit value")
    plt.clim(-0.5, len(np.unique(target)) - 0.5)
    plt.axis("off")
    if save_plot is not None:
        plt.savefig(save_plot, bbox_inches="tight")
    else:
        plt.show()
