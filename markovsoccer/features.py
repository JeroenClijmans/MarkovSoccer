import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotsoccer.fns as fns
import socceraction.spadl.config as spadlconfig
from abc import ABC, abstractmethod
from markovsoccer.team_model import TeamModel
from markovsoccer.config import *
from scipy.interpolate import interp2d  # type: ignore


class Feature(ABC):

    @staticmethod
    @abstractmethod
    def calculate(team_model: TeamModel) -> any:
        pass


class SideUsage(Feature):

    def __init__(self):
        return

    @staticmethod
    def calculate(team_model: TeamModel) -> dict[str, float]:
        left = team_model.average_number_visits_in(fromm=INITIAL_STATE, states=LEFT_STATES)
        center = team_model.average_number_visits_in(fromm=INITIAL_STATE, states=CENTER_STATES)
        right = team_model.average_number_visits_in(fromm=INITIAL_STATE, states=RIGHT_STATES)
        total = left + center + right
        d = {
            'left': left / total,
            'right': right / total,
            'center': center / total
        }
        return d

    @staticmethod
    def visualize(team_model: TeamModel):
        zone_usages = SideUsage.calculate(team_model)
        left_val, right_val, center_val = zone_usages['left'], zone_usages['right'], zone_usages['center']
        avgs = dict()
        for i in range(0, 48):
            avgs[i] = left_val
        for i in range(48, 144):
            avgs[i] = center_val
        for i in range(144, 192):
            avgs[i] = right_val
        matrix = _convert_dictionary_to_heatmap(avgs, 12, 16)
        ax = _heatmap_interpolated_visualization(matrix, show=False, cbar=False, norm_min=0.1, norm_max=0.6)

        cfg = fns.spadl_config
        cell_width = (cfg["width"] - cfg["origin_x"]) / WIDTH
        cell_length = (cfg["length"] - cfg["origin_y"]) / LENGTH
        base_y = cfg["origin_y"]
        base_x = cfg["origin_x"]
        left_x = base_x + 8 * cell_length
        left_y = base_y + 10.5 * cell_width
        center_x = base_x + 8 * cell_length
        center_y = base_y + 6 * cell_width
        right_x = base_x + 8 * cell_length
        right_y = base_y + 1.5 * cell_width
        ax.text(left_x, left_y, '{:.1%} (L)'.format(left_val), horizontalalignment='center',
                verticalalignment='center',
                zorder=10000, fontsize=18)
        ax.text(center_x, center_y, '{:.1%} (C)'.format(center_val), horizontalalignment='center',
                verticalalignment='center',
                zorder=10000, fontsize=18)
        ax.text(right_x, right_y, '{:.1%} (R)'.format(right_val), horizontalalignment='center',
                verticalalignment='center',
                zorder=10000, fontsize=18)
        plt.axis("scaled")
        plt.show()


def _heatmap_interpolated_visualization(
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
    return _heatmap_visualization(interp(x, y), ax=ax, figsize=None, alpha=1, cmap="hot", linecolor="white",
                                  cbar=cbar, show=show, norm_min=norm_min, norm_max=norm_max)


def _convert_dictionary_to_heatmap(results: dict, width:int, length:int) -> np.ndarray:
    heatmap = np.zeros((width, length))
    for row in range(width):
        for col in range(length):
            index = row*length + col
            heatmap[row, col] = results[index]
    return heatmap


# Modification to the heatmap function of matplotsoccer
def _heatmap_visualization(
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
