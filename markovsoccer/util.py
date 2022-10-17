import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotsoccer.fns as fns
import socceraction.spadl.config as spadlconfig
from markovsoccer.config import *
from scipy.interpolate import interp2d  # type: ignore


def get_cell_indices_from_flat(flat_cell_index: int) -> (int, int):
    x = flat_cell_index % LENGTH  # length index
    y = flat_cell_index // LENGTH  # width index
    return y, x


def heatmap_interpolated_visualization(
        heatmap: np.ndarray,
        ax=None,
        cbar=True,
        show=True,
        norm_min=0.,
        norm_max=1.
):

    (width, length) = heatmap.shape

    cell_length = spadlconfig.field_length / length
    cell_width = spadlconfig.field_width / width

    x = np.arange(0.0, spadlconfig.field_length, cell_length) + 0.5 * cell_length
    y = np.arange(0.0, spadlconfig.field_width, cell_width) + 0.5 * cell_width

    interp = interp2d(x=x, y=y, z=heatmap, kind="linear", bounds_error=False)

    x = np.linspace(0, 105, 1050)
    y = np.linspace(0, 68, 680)
    return heatmap_visualization(interp(x, y), ax=ax, figsize=None, alpha=1, cmap="hot", linecolor="white",
                                 cbar=cbar, show=show, norm_min=norm_min, norm_max=norm_max)


def convert_dictionary_to_heatmap(results: dict, width: int, length: int) -> np.ndarray:
    heatmap = np.zeros((width, length))
    for row in range(width):
        for col in range(length):
            index = row*length + col
            heatmap[row, col] = results[index]
    return heatmap


# Modification to the heatmap function of matplotsoccer
def heatmap_visualization(
    matrix,
    ax=None,
    figsize=None,
    alpha=1,
    cmap="Blues",
    linecolor="black",
    cbar=False,
    show=True,
    norm_min=0.,
    norm_max=1.,
):
    if ax is None:
        ax = fns._field(
            figsize=figsize, linecolor=linecolor, fieldcolor="white", show=False
        )

    cfg = fns.spadl_config
    zheatmap = fns.zheatmap
    x1, y1, x2, y2 = (
        cfg["origin_x"],
        cfg["origin_y"],
        cfg["origin_x"] + cfg["length"],
        cfg["origin_y"] + cfg["width"],
    )
    extent = (x1, x2, y1, y2)

    norm = mpl.colors.Normalize(vmin=norm_min, vmax=norm_max)

    limits = ax.axis()
    imobj = ax.imshow(
        matrix, norm=norm, extent=extent, aspect="auto", alpha=alpha, cmap=cmap, zorder=zheatmap
    )
    ax.axis(limits)

    if cbar:
        # dirty hack
        # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        colorbar = plt.gcf().colorbar(
            imobj, ax=ax, fraction=0.035, aspect=15, pad=-0.05
        )
        colorbar.minorticks_on()

    plt.axis("scaled")
    if show:
        plt.show()
    return ax
