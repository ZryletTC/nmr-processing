"""
Functions for reading data from raw Bruker NMR files.

All functions return a dictionary containing data with appropriate keys.

TODO: diff_params_import, xf2, xf2_peak_pick
"""

import os

import numpy as np
from nmrglue.fileio import bruker


def get_1d_data(data_path, proc_num=1, include_md=False):
    """
    Get 1D NMR data from raw Bruker files. Returns a dictionary bundling the data.

    data_path: Top-level experiment folder containing the 2D NMR data.
    proc_num: Process number of data to be plotted (default = 1).
    include_md: If true, include procs and acqus dictionaries in the output bundle.
    """

    pdata_path = os.path.join(data_path, "pdata", str(proc_num))
    metadata, data = bruker.read_pdata(pdata_path)

    # Determine x axis values
    offset_freq = metadata["acqus"]["O1"]
    basic_field = metadata["acqus"]["BF1"]
    nucleus = metadata["acqus"]["NUC1"]

    pdata_size = metadata["procs"]["SI"]
    spectral_ref_freq = metadata["procs"]["SF"]
    spectral_width_hz = metadata["procs"]["SW_p"]

    spectral_ref = (spectral_ref_freq - basic_field) * 1e6
    true_center = offset_freq - spectral_ref
    x_min = true_center - spectral_width_hz / 2
    x_max = true_center + spectral_width_hz / 2
    x_vals_hz = np.linspace(x_max, x_min, num=int(pdata_size))
    x_vals_ppm = x_vals_hz / spectral_ref_freq

    bundle = {
        "data_type": "1d",
        "nucleus": nucleus,
        "x_vals_hz": x_vals_hz,
        "x_vals_ppm": x_vals_ppm,
        "y_data": data,
    }

    if include_md:
        bundle.update(metadata)

    return bundle


def get_2d_data(data_path, proc_num=1):
    """
    Get 2D NMR data from raw Bruker files.

    data_path: Top-level experiment folder containing the 2D NMR data.
    proc_num: Process number of data to be plotted (default = 1).
    """

    pdata_path = os.path.join(data_path, "pdata", str(proc_num))
    metadata, data = bruker.read_pdata(pdata_path)

    bundle = {"data_type": "2d", "z_data": data}

    # Determine values for x and y axes
    for dim in ["x", "y"]:
        if dim == "x":
            acqus = "acqus"
            procs = "procs"
        else:
            acqus = "acqu2s"
            procs = "proc2s"

        offset_freq = metadata[acqus]["O1"]
        basic_field = metadata[acqus]["BF1"]
        nucleus = metadata[acqus]["NUC1"]

        pdata_size = metadata[procs]["SI"]
        spectral_ref_freq = metadata[procs]["SF"]
        spectral_width_hz = metadata[procs]["SW_p"]

        spectral_ref = (spectral_ref_freq - basic_field) * 1e6
        center = offset_freq - spectral_ref
        min_hz = center - spectral_width_hz / 2
        max_hz = center + spectral_width_hz / 2
        vals_hz = np.linspace(max_hz, min_hz, num=int(pdata_size))

        bundle[f"{dim}_nucleus"] = nucleus
        bundle[f"{dim}_vals_hz"] = vals_hz
        bundle[f"{dim}_vals_ppm"] = vals_hz / spectral_ref_freq

    # TODO: Add ability to crop data using f1l,f1r or xmin, xmax

    return bundle
