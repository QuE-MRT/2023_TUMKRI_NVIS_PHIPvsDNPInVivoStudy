# Authors: Andre Wendlinger, andre.wendlinger@tum.de
#          Luca Nagel, luca.nagel@tum.de
#          Wolfgang Gottwald, wolfgang.gottwald@tum.de


# Institution: Technical University of Munich
# Date of last Edit (dd.mm.yyyy): 24.05.2023

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_colorbar(figure, axis, data, **kwargs):
    """Appends colorbar to axis and scales it according to min and max of data.

    Requires the following imports:

        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable

    'cmap' arg. deprecated, use arg. 'mappable' instead:
        img = ax.imshow(...)
        plot_colorbar(..., mappable=img)
    """
    # sort out arrangement of colorbar and plot
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # normalize internal data form 0 to 1
    norm = plt.Normalize(np.nanmin(data), np.nanmax(data))

    # try to get corresponding cmap from axis in case none was specified
    cmap = kwargs.get("cmap", False)
    if not cmap:
        try:
            cmap = axis.get_children()[0].get_cmap()
        except AttributeError as e:
            print("Failed to retrieve cmap from axis.get_children()[0]")
            print("Children are:")
            [print(a) for a in axis.get_children()]
            raise e

    # plot the colorbar
    figure.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="vertical",
        **kwargs,
    )


def plot_image_series(
    arrays,
    label_list,
    nrows=1,
    cmap="plasma",
    plot_func=None,
    normalize=False,
    **subplot_kwrags,
):
    """Plots series of images into subplots, optionally into multiple rows"""

    def pad_or_truncate(some_list, target_len):
        return some_list[:target_len] + [""] * (target_len - len(some_list))

    ncols = len(arrays) // nrows

    if len(label_list) != nrows * ncols:
        label_list = pad_or_truncate(label_list, nrows * ncols)

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, **subplot_kwrags)

    if normalize:
        global_min, global_max = np.min(arrays), np.max(arrays)

    for ax, arr, label in zip(axs.flat, arrays, label_list):
        if plot_func:
            plot_func(ax, arr)
            plot_colorbar(fig, ax, arr)
        else:
            if normalize:
                ax.imshow(arr, cmap=cmap, vmin=global_min, vmax=global_max)
                plot_colorbar(fig, ax, [global_min, global_max])
            else:
                ax.imshow(arr, cmap=cmap)
                plot_colorbar(fig, ax, arr)
        ax.axis("off")
        ax.set_title(label)

    fig.tight_layout()


def plot_3d(data, **scatter_kwargs):
    """3D colorcoded matrix plot.

    https://stackoverflow.com/questions/14995610/how-to-make-a-4d-plot-with-matplotlib-using-arbitrary-data
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    mask = data < 1e5
    idx = np.arange(int(np.prod(data.shape)))
    x, y, z = np.unravel_index(idx, data.shape)
    ax.scatter(
        x,
        y,
        z,
        c=data.flatten(),
        s=10.0 * mask,
        edgecolor="face",
        alpha=0.5,
        marker="o",
        cmap="magma",
        linewidth=0,
    )
    plt.tight_layout()
    return ax
