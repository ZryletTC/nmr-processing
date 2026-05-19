"""
Utilities for reading and processing SIR-specific NMR intensity files.

This module includes helpers for reading T1 intensity files and variable delay lists
used in SIR/T1 relaxation experiments.

TODO: Reorganize between process_sir.py and sirtools
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Model
from lmfit.models import PseudoVoigtModel
from nmrglue.fileio import bruker

from nmr_processing.plotting import plot_1d
from nmr_processing.processing import (
    get_1d_data,
    get_data_from_folder,
    get_peak_slice_intensities,
    get_pseudo2d_data,
)


def read_t1ints(exp_path, proc_num=1, delay_offset=True, normalize=True):

    t1ints_path = os.path.join(exp_path, "pdata", str(proc_num), "t1ints.txt")

    times = []
    ints = []

    with open(t1ints_path, "r") as file:
        line = file.readline()  # skip first line
        line = file.readline()
        while not line.startswith("-"):
            times.append(float(line.split(" ")[0]))  # in seconds

            line = file.readline()
            n_sites = int(line.split(" ")[2])
            if not ints:
                ints = [[] for i in range(n_sites)]  # gotta be a better way
            for i in range(n_sites):
                line = file.readline()
                integral = float(line.split(" ")[1])  # integral is middle number
                ints[i].append(integral)

            line = file.readline()  # start next line

    # print('finished reading')
    times = np.array(times)
    ints = np.array(ints).T
    if normalize:
        # ints = (ints - np.min(ints))/(np.max(ints) - np.min(ints))
        ints = ints / np.max(ints)
    # print('finished np')

    if delay_offset:
        pulse_lengths = bruker.read_acqus_file(exp_path)["acqus"]["P"]
        p1_and_p2 = float(pulse_lengths[1]) + float(pulse_lengths[2])
        times += (p1_and_p2 * 1e-6) / 2

    # TODO: Convert t1ints indices to ppm positions
    positions = [0 for i in range(n_sites)]  # for now just doing the right len

    return times, ints, positions


def load_vdlist(
    exp_path,
    delay_offset=True,
):

    vdlist_path = os.path.join(exp_path, "vdlist")

    str_arr = np.loadtxt(vdlist_path, dtype=str)
    delays = []
    for val in str_arr:
        # if val.isnumeric():
        #     delays.append(float(val))
        # else:
        try:
            delays.append(float(val))
        except ValueError:
            num = float(val.rstrip("pnum"))
            c = val[-1]
            if c == "p":
                num *= 1e-12
            elif c == "n":
                num *= 1e-9
            elif c == "u":
                num *= 1e-6
            elif c == "m":
                num *= 1e-3
            else:
                raise ValueError(f"Unexpected value in vdlist: {val}")
            delays.append(num)
    delays = np.array(delays)

    if delay_offset:
        pulse_lengths = bruker.read_acqus_file(exp_path)["acqus"]["P"]
        p1_and_p2 = float(pulse_lengths[1]) + float(pulse_lengths[2])
        delays += (p1_and_p2 * 1e-6) / 2
    return delays


def process_sir(
    exp_path,
    proc_num=1,
    peak_pos=None,
    delay_offset=True,
    regions=None,
):
    """
    Process a Selective Inversion Recovery experiment.
    Pass in the path to the experiment and this returns the data.

    Also works for T1 measurements or other pseudo-2D exps that use vdlist.
    """

    # Load vdlist and interpret suffixes
    vdlist = os.path.join(exp_path, "vdlist")
    delays = load_vdlist(vdlist, delay_offset=delay_offset)

    bundle = get_pseudo2d_data(exp_path, proc_num=proc_num)
    x_vals_ppm = bundle["x_vals_ppm"]
    y_data = bundle["y_data"]

    if peak_pos:
        intensities = get_peak_slice_intensities(x_vals_ppm, y_data, peak_pos=peak_pos)
    elif regions:
        ints = []
        # FIXME: region subset not used
        for f2l, f2r in regions:
            ints.append(np.trapz(x_vals_ppm, y_data))
        intensities = np.concatenate(ints, axis=1)
    else:
        peak_pick_bundle = get_peak_slice_intensities(
            x_vals_ppm, y_data, prominence=[0.9, 1]
        )
        intensities = peak_pick_bundle["peak_ints_norm"]

    return delays, intensities


def make_cifit_files(
    filename,
    delays,
    intensities,
    title=None,
    names=[],
    R1_guesses=[],
    tp2=False,
    k_guesses=[],
    matrix=[],
    M_0_guesses=[],
    M_f_guesses=[],
):
    make_dat_file(
        filename, delays=delays, intensities=intensities, title=title, names=names
    )

    intensities = np.array(intensities)
    M_0_guesses = M_0_guesses if M_0_guesses else list(intensities[0])

    make_mch_file(
        filename,
        title=title,
        sites=intensities.shape[1],
        M_0_guesses=M_0_guesses,
        M_f_guesses=M_f_guesses,
        R1_guesses=R1_guesses,
        tp2=tp2,
        k_guesses=k_guesses,
        matrix=matrix,
    )


def make_mch_file(
    filename,
    sites=2,
    processes=1,
    R1_guesses=[],
    k_guesses=[],
    M_f_guesses=[],
    M_0_guesses=[],
    matrix=[],
    title="TEST",
    tp2=False,
):
    """
    Make .mch file describing the mechanism for CIFIT fitting,

    tp2 hard codes varying rate and M0 but keeping M_inf and T1s constant,
    for use with cifit2.1 aka cifit2 aka cifit_tp2

    TODO: Rename tp2
    """
    if matrix:
        matrix = np.array(matrix)
        if matrix.shape != (sites, sites):
            raise ValueError(
                "matrix must be a square array of shape `sites` by `sites`!"
            )
    else:
        if processes == 1:
            matrix = np.ones((sites, sites)) - np.diag(np.ones(sites))
        else:
            raise NotImplementedError(
                "This function can't currently handle "
                "the off-diagonals for more than one "
                "process."
            )

    mch_lines = [title]

    mch_lines.extend(["", f"{sites} {processes}"])

    if not R1_guesses:
        R1_guesses = [1 / 0.462, 1 / 2.141]  # Example reciprocal of LPSC and LZC T1s

    if len(R1_guesses) != sites:
        raise ValueError("`r1_guesses` must be of length `sites`!")

    mch_lines.extend(["", " ".join(map(str, R1_guesses))])
    if tp2:
        mch_lines.append(" ".join(map(str, np.ones(sites, dtype=np.int8))))

    if M_f_guesses:
        if len(M_f_guesses) != sites:
            raise ValueError("`M_f_guesses` must be of length `sites`!")
    else:
        M_f_guesses = np.ones(sites)  # Final intensities are normalized, so 1
    mch_lines.extend(["", " ".join(map(str, M_f_guesses))])
    if tp2:
        mch_lines.append(" ".join(map(str, np.ones(sites, dtype=np.int8))))

    if M_0_guesses:
        if len(M_0_guesses) != sites:
            raise ValueError("`M_0_guesses` must be of length `sites`!")
    else:
        # This is a bad guess, it should be more like 1, -1
        M_0_guesses = np.ones(sites)
    mch_lines.extend(["", " ".join(map(str, M_0_guesses))])
    if tp2:
        mch_lines.append(" ".join(map(str, np.ones(sites, dtype=np.int8))))

    if k_guesses:
        if len(k_guesses) != sites:
            raise ValueError("`k_guesses` must be of length `sites`!")
    else:
        k_guesses = np.ones(processes)  # This is a bad guess

    for i in range(processes):
        if tp2:
            # rate guess for the mechanism, with variation specified
            mch_lines.extend(["", str(k_guesses[i]) + " 1"])
        else:
            # rate guess for the mechanism
            mch_lines.extend(["", str(k_guesses[i])])
        mch_lines.append(str(sites * (sites - 1)))  # number of off-diagonals

        for j in range(sites):
            for k in range(sites):
                if matrix[j][k] != 0:
                    mch_lines.append(f"{j} {k} {matrix[j][k]}")

    with open(filename + ".mch", "w") as file:
        file.writelines(s + "\n" for s in mch_lines)


def make_dat_file(filename, delays, intensities, title="TEST", names=[]):
    data_lines = [title]

    numpoints = len(delays)
    intensities = np.array(intensities)
    if intensities.shape[0] != numpoints:
        raise ValueError("`delays` and `intensities` must have the same length!")

    data_lines.extend(["", str(numpoints), ""])

    if list(names):
        comment = "# Tmix, " + ", ".join(map(str, names))
    else:
        comment = "# Tmix, unknown intensities..."
    data_lines.append(comment)

    delays = delays.reshape(numpoints, 1)
    data = np.concatenate((delays, intensities), axis=1)

    def format_array(arr):
        return np.array2string(arr, precision=6, suppress_small=True)

    for line in data:
        data_lines.append("\t".join(map(format_array, line)))
    # data_str = np.array_str(data, precision=5, suppress_small=True)
    # data_lines.extend(data_str.strip('[] ').split(']\n ['))

    with open(filename + ".dat", "w") as file:
        file.writelines(s + "\n" for s in data_lines)


def exp_to_cifit(
    exp_path,
    outfile=None,
    proc_num=1,
    peak_pos=[],
    peak_names=[],
    T1_values=[],
    tp2=False,
    k_guesses=[],
    matrix=[],
    M_0_guesses=[],
    M_f_guesses=[],
    ints=True,
):
    # Add plotting back into sir functions

    if ints:
        t1ints = os.path.join(exp_path, f"pdata/{proc_num}/t1ints.txt")
        delays, ints, positions = read_t1ints(t1ints)
    else:
        delays, ints = process_sir(exp_path, proc_num=proc_num, peak_pos=peak_pos)
        positions = None

    exp_name = os.path.basename(os.path.dirname(exp_path))
    exp_no = int(os.path.basename(exp_path))
    title = f"Extracted from {exp_name} exp no {exp_no}"

    if peak_names and positions:
        if len(peak_names) != len(positions):
            raise ValueError(f"Wrong number of peak names, should be {len(positions)}")
    else:
        peak_names = [f"{s:.2f} ppm" for s in positions]

    R1_guesses = [1 / T1 for T1 in T1_values]

    if outfile is None:
        outfile = exp_path

    make_cifit_files(
        outfile,
        delays,
        ints,
        title=title,
        names=peak_names,
        R1_guesses=R1_guesses,
        tp2=tp2,
        k_guesses=k_guesses,
        matrix=matrix,
        M_0_guesses=M_0_guesses,
        M_f_guesses=M_f_guesses,
    )


def plot_cifit_csv(
    filepath, nsites=2, names=[], data_rows=16, fit_rows=101, savepath=""
):
    cols = ["delay"]
    cols.extend(map(str, range(nsites * 3)))
    data_df = pd.read_csv(
        filepath,
        sep=None,
        nrows=data_rows,
        skiprows=1,
        header=None,
        index_col=False,
        names=cols,
    )
    # print('DATA_DF')
    # print(data_df)

    cols = ["delay"]
    cols.extend(map(str, range(nsites)))
    fit_df = pd.read_csv(
        filepath,
        sep=None,
        nrows=fit_rows,
        skiprows=3 + data_rows,
        header=None,
        index_col=False,
        names=cols,
    )

    # print('FIT_DF')
    # print(fit_df)

    if len(names) == 0:
        names = [f"Site {i+1}" for i in range(nsites)]

    _, ax = plt.subplots()
    for i in range(nsites):
        pts = ax.plot(
            data_df["delay"], data_df[str(nsites + i)], ".", label=names[i] + " Data"
        )
        ax.plot(
            fit_df["delay"],
            fit_df[str(i)],
            label=names[i] + " Calc",
            color=pts[0].get_color(),
        )

    outpath = filepath.replace(".csv", ".out")
    if os.path.exists(outpath):
        with open(outpath, "r") as f:
            outtext = f.read()
        outtext = outtext[outtext.find("Final") :]
        # print(outtext)
        # TODO: Check if regex still work with double quotes and raw string
        match = re.search('No. \d+=\s*([\.\d]+)\nChi', outtext)  # fmt: skip # noqa
        # print(match)
        if match:
            rate = match.group(1)
            k = float(rate)
            # print(k)
            ax.text(
                0.9,
                0.5,
                f"Rate = {k} Hz",
                horizontalalignment="right",
                transform=ax.transAxes,
            )
        else:
            print("Rate not found in .out file")
            return

    ax.legend()

    plt.tight_layout()
    if not savepath:
        savepath = filepath.replace(".csv", ".pdf")
    plt.savefig(savepath)
    plt.show()


# exp_path = '/Users/tylerpennebaker/BoxSync/wp6_exsy/EXSYstudy/500.TP-2024.10.31_7Li_LZC+LPSC/219'  # noqa
# exp_to_cifit(exp_path, 'test', peak_pos=[1.46, -0.92])


########################
# Functions from leonmr:
########################
def get_1d_exsy_data(datapath, exp_nums, peak_pos=None, plot=False):
    # TODO: Use nmrglue to get D15 value
    d15s = []
    for exp_num in exp_nums:
        Dstr = "##$D= (0..63)"
        acqus = os.path.join(datapath, str(exp_num), "acqus")
        with open(acqus, "rb") as input:
            for line in input:
                if Dstr in line.decode():
                    line = next(input)
                    linestr = line.decode()
                    D = linestr.strip("\n").split(" ")
                    D15 = float(D[15])
                    break
        d15s.append(D15)
    d15s = np.array(d15s)

    data_bundle = get_data_from_folder(datapath, exp_nums)
    if "x_vals_ppm" not in data_bundle:
        raise ValueError("x_vals_ppm don't match for all experiments! Can not proceed.")
    x_vals_ppm = data_bundle["x_vals_ppm"]
    # CHECK: order might be messed up in dict?
    y_data = [exp_bundle["y_data"] for _, exp_bundle in data_bundle.items()]

    proc_bundle = get_peak_slice_intensities(x_vals_ppm=x_vals_ppm, y_data=y_data)
    peak_ints_norm = proc_bundle["peak_ints_norm"]

    return d15s, peak_ints_norm


def analyze_lpsc_1d_exsys(datapath, exp_nums, plot=False):
    first_path = os.path.join(datapath, str(exp_nums[0]))

    if plot:
        bundle = plot_1d(first_path, f1p=3, f2p=-3.2)
        ax = bundle["ax"]
    else:
        bundle = get_1d_data(first_path)
        _, ax = plt.subplots()

    x_vals_ppm = bundle["x_vals_ppm"]
    y_data = bundle["y_data"]

    amplitude = -np.trapz(y_data, x=x_vals_ppm)

    lpsc_peak1 = PseudoVoigtModel(prefix="p1_")
    init_pars = lpsc_peak1.make_params(
        center=dict(value=1.2, min=-10, max=10),
        amplitude=dict(value=0.6 * amplitude, min=0),
        sigma=dict(value=0.1, min=0.001, max=3),
    )
    lpsc_peak2 = PseudoVoigtModel(prefix="p2_")
    init_pars.update(
        lpsc_peak2.make_params(
            center=dict(value=1, min=-10, max=10),
            amplitude=dict(value=0.4 * amplitude, min=0),
            sigma=dict(value=0.1, min=0.001, max=3),
        )
    )

    lpsc_model = lpsc_peak1 + lpsc_peak2
    first_fit = lpsc_model.fit(y_data, init_pars, x=x_vals_ppm)
    first_fit.plot_fit(ax=ax, numpoints=100, fitfmt="r-")
    lpsc_fits = [first_fit]
    lpsc_pars = lpsc_model.make_params(**first_fit.best_values)

    lpsc_pars["p1_center"].set(
        max=first_fit.best_values["p1_center"] + 0.2,
        min=first_fit.best_values["p1_center"] - 0.2,
    )
    lpsc_pars["p2_center"].set(
        max=first_fit.best_values["p2_center"] + 0.2,
        min=first_fit.best_values["p2_center"] - 0.2,
    )

    d15s = []
    for exp_num in exp_nums:
        Dstr = "##$D= (0..63)"
        acqus = os.path.join(datapath, str(exp_num), "acqus")
        with open(acqus, "rb") as input:
            for line in input:
                if Dstr in line.decode():
                    line = next(input)
                    linestr = line.decode()
                    D = linestr.strip("\n").split(" ")
                    D15 = float(D[15])
                    break
        d15s.append(D15)
    d15s = np.array(d15s)

    bundle = get_data_from_folder(datapath, exp_nums[1:])

    x_vals_ppm = bundle["x_vals_ppm"]
    spectra = bundle["y_data"]

    for y_data in spectra:
        new_fit = lpsc_model.fit(y_data, lpsc_pars, x=x_vals_ppm)
        new_fit.plot_fit(ax=ax, numpoints=100, fitfmt="r-")
        lpsc_fits.append(new_fit)

    p1_ints = [
        fit.best_values["p1_amplitude"] / first_fit.best_values["p1_amplitude"]
        for fit in lpsc_fits
    ]
    p2_ints = [
        fit.best_values["p2_amplitude"] / first_fit.best_values["p2_amplitude"]
        for fit in lpsc_fits
    ]

    return d15s, [p1_ints, p2_ints]


def fit_1d_exsys(mixtimes, intensities, savename=None, fixed_t1=None, plot=True):
    default_k = 8
    default_t1 = 0.2

    def exsy1dfit(x, k, t1):
        return np.multiply(
            np.exp(np.multiply(-1 / t1, x)),
            np.divide(1 + np.exp(np.multiply(2 * k, x)), 2),
        )

    exsy_model = Model(exsy1dfit)
    params = exsy_model.make_params(k=default_k, t1=default_t1)
    if fixed_t1 is not None:
        params["t1"].set(value=fixed_t1, vary=False)
    fit_result = exsy_model.fit(intensities, params, x=mixtimes)
    print(fit_result.fit_report())

    # def exsy1dfit_fixedt1(t1_fixed):
    #     def wrapped(x, k, t1=t1_fixed):
    #         return exsy1dfit(x, k, t1)
    #     return wrapped

    # if fixed_t1 is None:
    #     model = exsy1dfit
    #     popt, pconv = curve_fit(exsy1dfit, mixtimes, intensities,
    #                             p0=[DEFAULT_K, DEFAULT_T1])
    # else:
    #     model = exsy1dfit_fixedt1(fixed_t1)
    #     popt, pconv = curve_fit(exsy1dfit, mixtimes, intensities, p0=[DEFAULT_K])

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(mixtimes, intensities)

        xfit = np.linspace(min(mixtimes), max(mixtimes), 100)
        ax.plot(xfit, fit_result.eval(x=xfit), "r-")
        # fit_result.plot_fit(ax=ax, numpoints=100, fitfmt='r-')

        ax.set_xlabel("Mixing Time (s)", fontname="Arial", fontsize=16)
        ax.set_ylabel("Normalized Peak Intensity", fontname="Arial", fontsize=16)

        if savename is not None:
            plt.savefig(savename, bbox_inches="tight", dpi=300)

        return fit_result, fig, ax

    return fit_result
