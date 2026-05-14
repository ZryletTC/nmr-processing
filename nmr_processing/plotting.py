"""
Functions for plotting data from raw Bruker NMR files.

All functions return a dictionary containing data and/or figures with appropriate keys.

TODO: sim_diffusion, plot_slice, diff_plot, T2_plot, T1_plot
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nmr_processing.processing import get_1d_data, get_2d_data
from nmr_processing.utils import nucleus_label

# Plot parameters
FONT_SIZE = 28
params = {
    "legend.fontsize": "large",
    # 'figure.figsize': (15,18),
    "font.family": "Arial",
    "font.weight": "normal",
    "figure.titlesize": FONT_SIZE + 2,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE * 0.8,
    "xtick.major.width": 2,
    "xtick.major.size": 10,
    "xtick.minor.visible": True,
    "xtick.minor.width": 1.5,
    "xtick.minor.size": 5,
    "ytick.labelsize": FONT_SIZE * 0.8,
    "ytick.major.width": 2,
    "ytick.major.size": 10,
    "ytick.minor.visible": True,
    "ytick.minor.width": 1.5,
    "ytick.minor.size": 5,
    "axes.labelweight": "normal",
    "axes.linewidth": 2,
    "axes.titlepad": 25,
}
plt.rcParams.update(params)


def plot_1d_nmr(
    arg,
    proc_num=1,
    mass=1,
    f1p=0,
    f2p=0,
    plwidth=15,
    plheight=12,
    normalize=False,
):
    """
    Plot 1D NMR data from raw Bruker files.

    The first argument must be either:

    bundle: Dictionary containing the data used for plotting.
    data_path: Top-level experiment folder containing the 1D NMR data.

    proc_num: Process number of data to be plotted (default = 1). If bundle provided,
              this does nothing.
    mass: Mass of sample for comparison to others (default = 1).
    normalize: Normalize max intensities if true. If False, normalize by mass if list of
               masses provided, otherwise do not normalize. (default = False).
    f1p/f2p: Left and right limits of x-axis, order agnostic.
    plheight/plwidth: Plot height/width in inches (default = 15x18).
    """

    if isinstance(arg, dict):
        bundle = arg
    elif isinstance(arg, str):
        data_path = arg
        bundle = get_1d_data(data_path, proc_num=proc_num)
    else:
        raise TypeError(
            "The first argument of this function must be a string of the path to the "
            "experiment folder containing the 1D NMR data or a dictionary containing "
            "the data to be plot."
        )

    y_data = bundle["y_data"]
    x_vals_ppm = bundle["x_vals_ppm"]
    x_label_text = nucleus_label(bundle["nucleus"])

    if f1p or f2p:
        x_low = np.abs(x_vals_ppm - f1p).argmin()
        x_high = np.abs(x_vals_ppm - f2p).argmin()
        if x_low > x_high:
            x_low, x_high = x_high, x_low
        x_vals_ppm = x_vals_ppm[x_low:x_high]
        y_data = y_data[x_low:x_high]

    if normalize:
        # y_data -= min(y_data)
        y_data /= max(y_data)
    else:
        y_data = y_data / mass

    fig, ax = plt.subplots()
    plt.plot(x_vals_ppm, y_data, "k", linewidth=3)
    plt.xlabel(x_label_text)
    if f1p + f2p != 0:
        if f2p > f1p:
            plt.xlim(f1p, f2p)
        else:
            plt.xlim(f2p, f1p)

    # ax.spines[['top','right','left']].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.yticks([])
    ax.invert_xaxis()

    fig.set_figheight(plheight)
    fig.set_figwidth(plwidth)

    bundle["fig"] = fig
    bundle["ax"] = ax

    return bundle


def plot_folder(
    arg,
    exp_num_list=None,
    mass=1,
    normalize=False,
    f1p=0,
    f2p=0,
    plwidth=15,
    plheight=18,
    stacking_factor=0,
):
    """
    Function to plot stacked all 1D NMR data from a folder of raw Bruker files.

    The first argument must be either:

    bundle: Dictionary containing the data used for plotting. This must be of the form
            {'exp_1': exp_bundle, 'exp_5': exp_bundle, 'exp_6': exp_bundle, etc.}.
    dir_path: Top-level data directory containing all 1D NMR experiment folders.

    exp_num_list: List of experiment numbers (e.g., [1, 5, 6, 10]). If bundle provided,
                 this does nothing. Providing an empty list (default) will plot all
                 experiments in the folder.
    mass: List of masses for each experiment for normalization.
    normalize: Normalize max intensities if true. If False, normalize by mass if list of
               masses provided, otherwise do not normalize. (default = False).
    f1p/f2p: Left and right limits of x-axis, order agnostic.
    plheight/plwidth: Plot height/width in inches (default = 15x18).
    stacking_factor: The amount of space between stacked spectra, scaled by spectrum
                     intensity. A value of 0 will overlay spectra. A value of 1 will
                     line up spectra baselines with the previous spectrum's maximum.
    """

    # Get bundle of experiment bundles
    if isinstance(arg, dict):
        bundle = arg
    elif isinstance(arg, str):
        dir_path = arg
        bundle = {}

        # If exp_num_list not provided, use all experiment-numbered folders in dir_path
        if exp_num_list is None:
            exp_num_list = [
                item
                for item in os.listdir(dir_path)
                if os.path.isdir(os.path.join(dir_path, item)) and item.isnumeric()
            ]

        for exp_num in exp_num_list:
            data_path = os.path.join(dir_path, str(exp_num))
            bundle[f"exp_{exp_num}"] = get_1d_data(data_path)
    else:
        raise TypeError(
            "The first argument of this function must be a string of the path to a "
            "folder containing experiment folders or a dictionary containing the data "
            "bundles to be plot."
        )

    fig, ax = plt.subplots()
    y_offset = 0

    for exp_num, exp_bundle in bundle.items():
        x_data = exp_bundle["x_vals_ppm"]
        y_data = exp_bundle["y_data"]

        if normalize:
            # y_data -= min(y_data)
            y_data /= max(y_data)
        else:
            y_data = y_data / mass

        plt.plot(x_data, y_data + y_offset)
        y_offset = y_offset + max(y_data) * stacking_factor

    if f1p + f2p != 0:
        if f2p > f1p:
            plt.xlim(f1p, f2p)
        else:
            plt.xlim(f2p, f1p)

    # ax.spines[['top','right','left']].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.yticks([])
    ax.invert_xaxis()

    fig.set_figheight(plheight)
    fig.set_figwidth(plwidth)

    # If all nuclei are the same, use specific label, otherwise just use 'shift / ppm'
    if len(set([exp_bundle["nucleus"] for exp_bundle in bundle])) == 1:
        plt.xlabel(nucleus_label(bundle))
    else:
        plt.xlabel("shift / ppm")

    bundle["fig"] = fig
    bundle["ax"] = ax

    return bundle


def plot_2d_nmr(
    arg,
    proc_num=1,
    f1l=0,
    f1r=0,
    f2l=0,
    f2r=0,
    factor=0.02,
    clevels=10,
    color=True,
    plheight=12,
    plwidth=12,
    show_projections=True,
):
    """
    Function to plot 2D NMR data from raw Bruker files.

    The first argument must be either:

    bundle: Dictionary containing the data used for plotting.
    data_path: Top-level experiment folder containing the 2D NMR data.

    proc_num: Process number of data to be plotted (default = 1). If bundle provided,
              this does nothing.
    f1l/f1r: Left and right limits of F1 (vertical) dimension.
    f2l/f2r: :eft and right limits of F2 (horizontal) dimension.
    factor: Minimum value for the contours (factor*max value).
            (default = 0.02, 2% of max signal)
    clevels: Number of contour levels for plot (default = 10).
    color: If True, plot with log-scaled color contours. Otherwise, just use black
            lines (default = True, color on).
    plheight/plwidth: Plot height/width in inches (default = 12x12).
    show_projections: If False, hide projections along each axis. Default = True.
    """

    if isinstance(arg, dict):
        bundle = arg
    elif isinstance(arg, str):
        data_path = arg
        bundle = get_2d_data(data_path, proc_num=proc_num)
    else:
        raise TypeError(
            "The first argument of this function must be a string of the path to the "
            "experiment folder containing the 2D NMR data or a dictionary containing "
            "the data to be plot."
        )

    x = bundle["x_vals_ppm"]
    y = bundle["y_vals_ppm"]
    z = bundle["z_data"]

    ####################################
    # Index limits of plot
    f2l_temp = max(x)
    f2r_temp = min(x)
    f1l_temp = max(y)
    f1r_temp = min(y)

    if f2l < f2r:
        f2l, f2r = f2r, f2l
    if f1l < f1r:
        f1l, f1r = f1r, f1l

    xlow = np.argmax(x < f2l)
    xhigh = np.argmax(x < f2r)

    ylow = np.argmax(y < f1l)
    yhigh = np.argmax(y < f1r)

    if xlow > xhigh:
        xlow, xhigh = xhigh, xlow

    if ylow > yhigh:
        ylow, yhigh = yhigh, ylow

    if f2l == 0:
        xlow = np.argmax(x == f2l_temp)
    if f2r == 0:
        xhigh = np.argmax(x == f2r_temp)
    if f1l == 0:
        ylow = np.argmax(y == f1l_temp)
    if f1r == 0:
        yhigh = np.argmax(y == f1r_temp)

    x = x[xlow:xhigh]
    y = y[ylow:yhigh]
    z = z[ylow:yhigh, xlow:xhigh]

    threshmin = factor * np.amax(z)
    threshmax = np.amax(z)
    cc2 = np.linspace(1, clevels, clevels)
    thresvec = [
        threshmin * ((threshmax / threshmin) ** (1 / clevels)) ** (1.25 * i)
        for i in cc2
    ]

    fig, ax = plt.subplots()

    if color:
        z = np.ma.masked_where(z <= 0, z)
        ax.contour(
            x,
            y,
            z,
            levels=thresvec,
            cmap=cm.seismic,
            norm=LogNorm(),
        )
    else:
        ax.contour(x, y, z, colors="black", levels=thresvec)

    plt.xlabel(nucleus_label(bundle["x_nucleus"]))
    plt.ylabel(nucleus_label(bundle["y_nucleus"]))

    if f1l + f1r != 0:
        if f1l > f1r:
            plt.ylim(f1r, f1l)
        else:
            plt.ylim(f1l, f1r)

    if f2l + f2r != 0:
        if f2l > f2r:
            plt.xlim(f2r, f2l)
        else:
            plt.xlim(f2l, f2r)

    ax.invert_xaxis()
    ax.invert_yaxis()

    if show_projections:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.tick_params(pad=6)

        divider = make_axes_locatable(ax)
        ax_f2 = divider.append_axes("top", 2, pad=0.1, sharex=ax)
        ax_f1 = divider.append_axes("left", 2, pad=0.1, sharey=ax)
        ax_f2.plot(x, z.sum(axis=0), "k")
        ax_f1.plot(-z.sum(axis=1), y, "k")

        # make projection axes invisible
        ax_f2.tick_params(axis="both", which="both", bottom=False, left=False)
        ax_f1.tick_params(axis="both", which="both", bottom=False, left=False)
        ax_f2.xaxis.set_tick_params(labelbottom=False)
        ax_f1.yaxis.set_tick_params(labelleft=False)
        ax_f2.spines["left"].set_visible(False)
        ax_f2.spines["right"].set_visible(False)
        ax_f2.spines["bottom"].set_visible(False)
        ax_f2.spines["top"].set_visible(False)
        ax_f1.spines["left"].set_visible(False)
        ax_f1.spines["right"].set_visible(False)
        ax_f1.spines["bottom"].set_visible(False)
        ax_f1.spines["top"].set_visible(False)
        ax_f2.yaxis.set_ticklabels([])
        ax_f1.xaxis.set_ticklabels([])

    fig.set_figheight(plheight)
    fig.set_figwidth(plwidth)

    plt.tight_layout()

    bundle["fig"] = fig
    bundle["ax"] = ax

    return bundle
