# pylint: skip-file
# flake8: noqa
"""
NMR Import/Plotting Functions

NMR1D(datapath, procno=1, showplot=True, f1p=0, f2p=0,plwidth=15,plheight=12,
      normalise=False)
NUC_label(NUC)
stackplot(datadir,Expt_no, nuc, f1p=0, f2p=0, plwidth=15,plheight=18,
          normalise=False)
NMR2D(datapath, procno=1, mass=1, f1l=0, f1r=0, f2l=0, f2r=0, factor = 0.02,
    clevels = 6, frame=False, homonuclear=False, plheight =18, plwidth = 18)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from lmfit import Model
from lmfit.models import PseudoVoigtModel
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.metrics import r2_score

# TODO: remove dependence on ssNMR
from ssNMR.formatting import format_plot
from ssNMR.fitting import fit

from nmr_processing.utils import find_gamma


def sim_diffusion(NUC, delta=1, DELTA=20, maxgrad=17, D=0):
    """
    fig, ax = sim_diffusion(NUC, delta=1, DELTA = 20, maxgrad = 17, D = 0)

    Function to help estimate appropriate diffusion experiment parameters. Can set the
    maximum gradient, little delta, and big DELTA to understand the level of
    attenuation/shape of the curve for whatever nuclide.

    D = 0 is a placeholder, if left as 0, D will be a range of 1e-7 to 1e-15 stepping
    by order of magnitude, if a value for D is set, only one line will be plotted.
    """

    from matplotlib import cm

    switch = 1
    delta = delta / 1000
    DELTA = DELTA / 1000
    if D == 0:
        D = np.logspace(-8, -15, 8)
        switch = 0
    gamma = find_gamma(NUC)  # [10^7 1/T/s]
    G = np.arange(0, maxgrad + (maxgrad / 100.0), maxgrad / 99.0)
    B = [(2 * np.pi * gamma * delta * i) ** 2 * (DELTA - (delta / 3)) for i in G]

    if switch == 0:
        intensities = np.zeros(shape=(len(D), len(G)))
        cnt = 0
        for j in D:
            intensities[cnt] = np.exp(np.multiply(-j, B))
            cnt += 1
    else:
        intensities = np.exp(np.multiply(-D, B))

    fig, ax = plt.subplots()
    if switch == 0:
        colmap = cm.seismic(np.linspace(0, 1, len(D)))
        [
            plt.plot(
                G,
                intensities[k, :],
                color=c,
                linewidth=2,
                label=str(D[k]) + r" $\mathregular{m^2 s^{–1}}$",
            )
            for k, c in zip(range(len(D)), colmap)
        ]
    else:
        plt.plot(
            G,
            intensities,
            linewidth=2,
            color="r",
            label=str(D) + r" $\mathregular{m^2 s^{–1}}$",
        )
    ax.set_xlim(0, maxgrad * 1.25)
    plt.legend(loc="upper right", frameon=False)
    plt.xlabel(r"Gradient Strength, g / $\mathregular{T m^{–1}}$")
    plt.ylabel(r"Intensity, $\mathregular{I/I_0}$")
    plt.show()
    return fig, ax


def xf2(datapath, procno=1, mass=1, f2l=10, f2r=0):
    """Return (xAxppm, real_spectrum, expt_parameters)."""

    real_spectrum_path = os.path.join(datapath, "pdata", str(procno), "2rr")
    procs = os.path.join(datapath, "pdata", str(procno), "procs")
    acqus = os.path.join(datapath, "acqus")
    proc2s = os.path.join(datapath, "pdata", str(procno), "proc2s")
    acqu2s = os.path.join(datapath, "acqu2s")

    ########################################################################

    # Bruker file format information
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Bruker binary files (ser/fid) store data as an array of numbers whose
    # endianness is determined by the parameter BYTORDA (1 = big endian, 0 = little
    # endian), and whose data type is determined by the parameter DTYPA (0 = int32,
    # 2 = float64). Typically the direct dimension is digitally filtered. The exact
    # method of removing this filter is unknown but an approximation is available.
    # Bruker JCAMP-DX files (acqus, etc) are text file which are described by the
    # `JCAMP-DX standard <http://www.jcamp-dx.org/>`_.  Bruker parameters are
    # prefixed with a '$'.

    ####################################

    # Get aqcus
    O1str = "##$O1= "
    OBSstr = "##$BF1= "
    NUCstr = "##$NUC1= <"
    Lstr = "##$L= (0..31)"
    CNSTstr = "##$CNST= (0..63)"
    TDstr = "##$TD= "

    O1 = float("NaN")
    OBS = float("NaN")
    NUC = ""
    L1 = float("NaN")
    L2 = float("NaN")
    CNST31 = float("NaN")
    TD = float("NaN")

    with open(acqus, "rb") as input:
        for line in input:
            #         print(line.decode())
            if O1str in line.decode():
                linestr = line.decode()
                O1 = float(linestr[len(O1str) : len(linestr) - 1])
            if OBSstr in line.decode():
                linestr = line.decode()
                OBS = float(linestr[len(OBSstr) : len(linestr) - 1])
            if NUCstr in line.decode():
                linestr = line.decode()
                NUC = str(linestr[len(NUCstr) : len(linestr) - 2])
            if TDstr in line.decode():
                linestr = line.decode()
                TD = float(linestr.strip(TDstr))
            if Lstr in line.decode():
                line = next(input)
                linestr = line.decode()
                L = linestr.strip("\n").split(" ")
                L1 = float(L[1])
                L2 = float(L[2])
            if CNSTstr in line.decode():
                CNST = []
                line = next(input)
                while "##$CPDPRG=" not in str(line):
                    linestr = line.decode()
                    CNST.extend(linestr.strip("\n").split(" "))
                    line = next(input)
                CNST31 = float(CNST[31])
            if (
                ~np.isnan(O1)
                and ~np.isnan(OBS)
                and ~np.isnan(L1)
                and ~np.isnan(TD)
                and ~np.isnan(CNST31)
                and not len(NUC) == 0
            ):
                break

    ####################################

    # Get procs

    SWstr = "##$SW_p= "
    SIstr = "##$SI= "
    SFstr = "##$SF= "
    NCstr = "##$NC_proc= "
    XDIM_F2str = "##$XDIM= "

    SW = float("NaN")
    SI = int(0)
    SF = float("NaN")
    NC_proc = float("NaN")
    XDIM_F2 = int(0)

    with open(procs, "rb") as input:
        for line in input:
            if SWstr in line.decode():
                linestr = line.decode()
                SW = float(linestr[len(SWstr) : len(linestr) - 1])
            if SIstr in line.decode():
                linestr = line.decode()
                SI = int(linestr[len(SIstr) : len(linestr) - 1])
            if SFstr in line.decode():
                linestr = line.decode()
                SF = float(linestr[len(SFstr) : len(linestr) - 1])
            if NCstr in line.decode():
                linestr = line.decode()
                NC_proc = float(linestr[len(NCstr) : len(linestr) - 1])
            if XDIM_F2str in line.decode():
                linestr = line.decode()
                XDIM_F2 = int(linestr[len(XDIM_F2str) : len(linestr) - 1])
            if (
                ~np.isnan(SW)
                and SI != int(0)
                and ~np.isnan(NC_proc)
                and ~np.isnan(SF)
                and XDIM_F2 != int(0)
            ):
                break

    ####################################

    # Get aqcu2s for indirect dimension
    O1str_2 = "##$O1= "
    OBSstr_2 = "##$BF1= "
    NUCstr_2 = "##$NUC1= <"
    TDstr_2 = "##$TD= "

    O1_2 = float("NaN")
    OBS_2 = float("NaN")
    NUC_2 = ""
    TD_2 = float("NaN")

    with open(acqu2s, "rb") as input:
        for line in input:
            #         print(line.decode())
            if O1str_2 in line.decode():
                linestr = line.decode()
                O1_2 = float(linestr[len(O1str_2) : len(linestr) - 1])
            if OBSstr_2 in line.decode():
                linestr = line.decode()
                OBS_2 = float(linestr[len(OBSstr_2) : len(linestr) - 1])
            if NUCstr_2 in line.decode():
                linestr = line.decode()
                NUC_2 = str(linestr[len(NUCstr_2) : len(linestr) - 2])
            if TDstr_2 in line.decode():
                linestr = line.decode()
                TD_2 = float(linestr.strip(TDstr_2))
            if (
                ~np.isnan(O1_2)
                and ~np.isnan(OBS_2)
                and ~np.isnan(TD_2)
                and not len(NUC_2) == 0
            ):
                break

    ####################################

    # # Get proc2s for indirect dimension

    SIstr_2 = "##$SI= "
    XDIM_F1str = "##$XDIM= "

    XDIM_F1 = int(0)
    SI_2 = int(0)

    with open(proc2s, "rb") as input:
        for line in input:
            if SIstr_2 in line.decode():
                linestr = line.decode()
                SI_2 = int(linestr[len(SIstr_2) : len(linestr) - 1])
            if XDIM_F1str in line.decode():
                linestr = line.decode()
                XDIM_F1 = int(linestr[len(XDIM_F1str) : len(linestr) - 1])
            if SI_2 != int(0) and XDIM_F1 != int(0):
                break

    ####################################

    # Determine x axis values
    SR = (SF - OBS) * 1000000
    true_centre = O1 - SR
    xmin = true_centre - SW / 2
    xmax = true_centre + SW / 2
    xAxHz = np.linspace(xmax, xmin, num=int(SI))
    xAxppm = xAxHz / SF

    real_spectrum = np.fromfile(real_spectrum_path, dtype="<i4", count=-1)
    if not bool(real_spectrum.any()):
        print(real_spectrum)
        print("Error: Spectrum not read.")

    # print(np.shape(real_spectrum),int(XDIM_F1), int(XDIM_F2), int(SI_2), int(SI))
    if XDIM_F1 == 1:
        real_spectrum = real_spectrum.reshape([int(SI_2), int(SI)])
    else:
        # to shape the column matrix according to Bruker's format, matrices are broken
        # into (XDIM_F1,XDIM_F2) submatrices, so reshaping where XDIM_F1!=1 requires
        # this procedure.
        column_matrix = real_spectrum
        submatrix_rows = int(SI_2 // XDIM_F1)
        submatrix_cols = int(SI // XDIM_F2)
        submatrix_number = submatrix_cols * submatrix_rows

        blocks = np.array(
            np.array_split(column_matrix, submatrix_number)
        )  # Split into submatrices
        blocks = np.reshape(
            blocks, (submatrix_rows, submatrix_cols, -1)
        )  # Reshape these submatrices so each has its own 1D array
        real_spectrum = np.vstack(
            [np.hstack([np.reshape(c, (XDIM_F1, XDIM_F2)) for c in b]) for b in blocks]
        )  # Concatenate submatrices in the correct orientation

    f2l_temp = max(xAxppm)
    f2r_temp = min(xAxppm)

    if f2l < f2r:
        f2l, f2r = f2r, f2l

    xlow = np.argmax(xAxppm < f2l)
    xhigh = np.argmax(xAxppm < f2r)

    if xlow == 0:
        xlow = np.argmax(xAxppm == f2l_temp)
    if xhigh == 0:
        xhigh = np.argmax(xAxppm == f2r_temp)

    if xlow > xhigh:
        xlow, xhigh = xhigh, xlow
    xAxppm = xAxppm[xlow:xhigh]
    real_spectrum = real_spectrum[: int(SI_2), xlow:xhigh]
    # real_spectrum = real_spectrum[:,xlow:xhigh]

    expt_parameters = {"NUC": NUC, "L1": L1, "L2": L2, "CNST31": CNST31, "TD_2": TD_2}

    return xAxppm, real_spectrum, expt_parameters


def diff_params_import(datapath, NUC):
    """
    delta, DELTA, expectedD, Gradlist = diff_params_import(datapath, NUC)
    obtains delta, DELTA, a guess for D, and the gradient list from the diff.xml file
    """
    import xml.etree.ElementTree as ET

    diff_params_path = os.path.join(datapath, "diff.xml")

    tree = ET.parse(diff_params_path)
    root = tree.getroot()

    delta = float(root.find(".//delta").text)  # [ms]
    delta = delta / 1000  # [s]
    DELTA = float(root.find(".//DELTA").text)  # [ms]
    DELTA = DELTA / 1000  # [s]
    exD = float(root.find(".//exDiffCoff").text)  # [m2/s]
    x_values_element = root.find(".//xValues/List")
    if x_values_element:
        x_values_list = x_values_element.text.split()
        x_values = [float(value) for value in x_values_list]
        Gradlist = x_values[1::4]  # [G/cm]
    else:
        x_values_list = root.findall(".//X")
        Gradlist = [float(i.attrib["g"]) for i in x_values_list]
    Gradlist = [x / 100 for x in Gradlist]  # [T/m]

    gamma = find_gamma(NUC)  # [10^7 1/T/s]
    # gamma = gamma  # [1/T/s]

    return delta, DELTA, exD, Gradlist, gamma


def plot_slice(
    datapath, slice_idx=0, procno=1, mass=1, f2l=1, f2r=-1, savefile="", normalize=True
):
    """
    Plot a single slice from a pseudo-2D experiment. Defaults to first slice.
    Will save plot to savefile path if specified.
    """

    xAxppm, real_spectrum, expt_parameters = xf2(
        datapath=datapath, procno=procno, mass=mass, f2l=f2l, f2r=f2r
    )

    if normalize:
        first_slice = real_spectrum[0, :]
        last_slice = real_spectrum[-1, :]
        best_slice = (
            first_slice if (np.sum(first_slice) > np.sum(last_slice)) else last_slice
        )

        min_best_slice = min(best_slice)
        best_slice = best_slice - min_best_slice
        max_best_slice = max(best_slice)
        best_slice = best_slice / max_best_slice
        real_spectrum = real_spectrum - min_best_slice
        real_spectrum = real_spectrum / max_best_slice

    fig, ax = plt.subplots()
    plt.plot(xAxppm, real_spectrum[slice_idx])

    ax.invert_xaxis()
    # ax.set_xlabel("Shift / ppm")
    # ax.set_ylabel("Normalized Intensity")

    if f2l or f2r:
        if f2l < f2r:
            plt.xlim(f2r, f2l)
        else:
            plt.xlim(f2l, f2r)

    plt.tight_layout()
    if savefile:
        plt.savefig(savefile)

    return fig, ax


def xf2_peak_pick(
    xAxppm,
    real_spectrum,
    prominence=[0.001, 1],
    peak_pos=float("NaN"),
    f1p=0,
    f2p=0,
    plot=True,
):
    """
    peak finder from the xf2 function, if peak_pos is defined, the fit for only that x
    value (in ppm) will be shown. Return peak_ints_norm.
    """

    first_slice = real_spectrum[0, :]
    last_slice = real_spectrum[-1, :]
    best_slice = (
        first_slice if (np.sum(first_slice) > np.sum(first_slice)) else last_slice
    )

    min_best_slice = min(best_slice)
    best_slice = best_slice - min_best_slice
    max_best_slice = max(best_slice)
    best_slice = best_slice / max_best_slice
    real_spectrum = real_spectrum - min_best_slice
    real_spectrum = real_spectrum / max_best_slice

    if np.isnan(peak_pos):
        pl = find_peaks(best_slice, prominence=prominence)
        pl = pl[0]  # indices of picked peaks
        peak_positions = xAxppm[pl]  # ppm values of picked peaks
        # cols = [str(round(items, 2)) for items in peak_positions]
    else:
        pl = [np.where(xAxppm <= peak_pos)[0][0]]  # broken!
        # cols = ["peak"]

    # best_slice_pl = real_spectrum[0,pl]
    # peak_slices = real_spectrum[:,pl]

    if plot:
        fig, ax = plt.subplots()

    # All Slices
    peak_ints = []
    for slices in real_spectrum:
        current_slice = slices
        peak_ints_now = [float(current_slice[i]) for i in pl]
        peak_ints.append(peak_ints_now)
        if plot:
            plt.plot(xAxppm, current_slice)

    if plot:
        ax.vlines(x=peak_pos, ymin=-0.075, ymax=0.0, color="r")
        ax.invert_xaxis()
        ax.set_xlabel("Shift / ppm")
        ax.set_ylabel("Normalized Intensity")

        if f1p != 0 and f2p != 0:
            if f2p < f1p:
                plt.xlim(f1p, f2p)
            else:
                plt.xlim(f2p, f1p)

    max_peak_ints = np.amax(peak_ints, axis=0)
    peak_ints_norm = []
    for slices2 in peak_ints:
        current_slice2 = np.divide(slices2, max_peak_ints)
        peak_ints_norm.append(current_slice2)

    # peak_intensity = pd.DataFrame(np.array(peak_ints_norm), columns=cols)
    # print(peak_intensity)

    if np.isnan(peak_pos):
        return np.array(peak_ints_norm), peak_positions
    else:
        return np.array(peak_ints_norm)


def read_1d_exsys(datapath, expnos, peak_pos=None, plot=False):
    d15s = []
    for expno in expnos:
        Dstr = "##$D= (0..63)"
        acqus = os.path.join(datapath, str(expno), "acqus")
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

    ppm, spectra = readfolder(datapath, expnos)
    peak_ints = xf2_peak_pick(
        ppm, spectra, prominence=[0.1, 1], f1p=4, f2p=-1.5, plot=plot
    )
    peak_ints = peak_ints.transpose()

    return d15s, peak_ints


def analyze_lpsc_1d_exsys(datapath, expnos, peak_pos=None, plot=False):
    first_path = os.path.join(datapath, str(expnos[0]))
    ppm, intensity, fig, ax = NMR1D(first_path, f1p=3, f2p=-3.2)

    amplitude = -np.trapz(intensity, x=ppm)

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
    first_fit = lpsc_model.fit(intensity, init_pars, x=ppm)
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
    for expno in expnos:
        Dstr = "##$D= (0..63)"
        acqus = os.path.join(datapath, str(expno), "acqus")
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

    ppm, spectra = readfolder(datapath, expnos[1:])

    for intensity in spectra:
        new_fit = lpsc_model.fit(intensity, lpsc_pars, x=ppm)
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
    DEFAULT_K = 8
    DEFAULT_T1 = 0.2

    def exsy1dfit(x, k, t1):
        return np.multiply(
            np.exp(np.multiply(-1 / t1, x)),
            np.divide(1 + np.exp(np.multiply(2 * k, x)), 2),
        )

    exsy_model = Model(exsy1dfit)
    params = exsy_model.make_params(k=DEFAULT_K, t1=DEFAULT_T1)
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


def T2_plot(peak_ints_norm, L1, L2, CNST31):
    """
    T2 plotting function, uses data read from the xf2 function
    T2_plot(peak_ints_norm, L1, CNST31)
    """
    echo_delay = np.arange(
        (2 * L1 / CNST31),
        (2 * ((L1) + (L2 * (len(peak_ints_norm[:, 0])))) / CNST31),
        2 * L2 / CNST31,
    )
    echo_delay *= 1000  # unit [=] ms
    fig2, ax2 = plt.subplots()
    plt.plot(echo_delay, peak_ints_norm)
    ax2.set_xlabel("Echo delay / ms")
    ax2.set_ylabel("Normalized Intensity")


def diff_plot(peak_ints_norm, datapath, NUC):
    """
    Diffusion plotting function, uses data read from the xf2 function
    G, grad_params = diff_plot(peak_ints_norm, datapath, NUC)
    """
    delta, DELTA, expD, G, gamma = diff_params_import(datapath, NUC)
    # print(delta,DELTA,expD,G, gamma)
    fig2, ax2 = plt.subplots()
    plt.plot(G, peak_ints_norm, "o")  # , c='red', mfc='blue', mec='blue')
    ax2.set_xlabel(r"Gradient Strength / G cm$\mathregular{^{-1}}$")
    ax2.set_ylabel("Normalized Intensity")
    grad_params = {"delta": delta, "DELTA": DELTA, "gamma": gamma, "expD": expD}
    return G, grad_params


def T2_Fit(x, y, t0=0.5, c0=1, beta0=0.5, showall=False, fittype="default"):

    def monoExp_t(x, t):
        result = []
        for i in x:
            result.append(np.exp(-(1 / t) * i))
        return result

    def doubleExp(x, m1, t1, t2):
        result = []
        for i in x:
            result.append(m1 * np.exp(-(1 / t1) * i) + (1 - m1) * np.exp(-(1 / t2) * i))
            # result.append(m1*np.exp(-(1/t1) * i)+m2*np.exp(-(1/t2) * i))

        return result

    def stretchExp(x, t, beta):
        result = []
        for i in x:
            result.append(np.exp(-(i / t)))  # **beta))
        return result

    method_str = ["Mono-exponential", "Bi-exponential", "Stretched Exponential"]

    params, cv = scipy.optimize.curve_fit(monoExp_t, x, y, t0)
    mono_t = params
    monoT2 = monoExp_t(x, mono_t)
    R_sq_Mono = r2_score(y, monoT2)

    param_bounds2 = ([0, 0, 0], [1, 1000000, 1000000])
    p0bi = (0.5, t0, t0)  # start with values near those we expect
    params, cv = scipy.optimize.curve_fit(doubleExp, x, y, p0bi, bounds=param_bounds2)
    m1, t1, t2 = params
    biexpT2 = doubleExp(x, m1, t1, t2)
    R_sq_Bi = r2_score(y, biexpT2)
    m2 = 1 - m1

    # start with values near those we expect --> c is near 1, T2 is close to to 4ms,
    # use a beta of 0.5
    p0str = (t0, beta0)
    params, cv = scipy.optimize.curve_fit(stretchExp, x, y, p0str, maxfev=1000)
    str_t, beta = params
    stretchT2 = stretchExp(x, str_t, beta)
    R_sq_Stretch = r2_score(y, stretchT2)

    All_R = [R_sq_Mono, R_sq_Bi, R_sq_Stretch]
    fit_types = ["Mono-exponential", "Bi-exponential", "Stretched exponential"]
    R_max = max(All_R)
    method_choice = All_R.index(R_max)
    if fittype != "default":
        method_choice = fit_types.index(fittype)
    # print(All_R, method_choice)

    if fittype == "default":
        if method_choice == 0:
            YY = monoT2

        elif method_choice == 1:
            YY = biexpT2

        elif method_choice == 2:
            YY = stretchT2
    else:
        if method_choice == 0:
            YY = monoT2
        elif method_choice == 1:
            YY = biexpT2
        elif method_choice == 2:
            YY = stretchT2

    if YY == monoT2:
        txt_disp = f"T$_2$ = {round(float(mono_t),6)}"
    elif YY == biexpT2:
        txt_disp = (
            f"Component 1: T$_2$ = {round(t1,6)} ms, w = {round(m1,3)}\n"
            f"Component 2: T$_2$ = {round(t2,6)} ms, w = {round(m2,3)}"
        )
    elif YY == stretchT2:
        txt_disp = (
            f"T$_2$ = {round(str_t,6)} ms\nβ = {round(beta,3)}"  # \nc = {round(c,3)}'
        )

    fig, ax = plt.subplots()

    if showall:
        plt.plot(x, y, "o", color="black", label="Experimental Data")
        plt.plot(x, monoT2, "--", color="teal", label=method_str[0] + " fit")
        plt.plot(x, biexpT2, "-.", color="orange", label=method_str[1] + " fit")
        plt.plot(x, stretchT2, ":", color="green", label=method_str[2] + " fit")
        plt.xlabel("Echo delay / ms")
        plt.ylabel("Normalized intensity")
        plt.legend(loc="right")
        plt.show()
    else:
        plt.plot(x, y, "o", color="blue", label="Experimental Data")
        plt.plot(x, YY, "--", color="red", label=method_str[method_choice] + " fit")
        plt.xlabel("Echo delay / ms")
        plt.ylabel("Normalized intensity")
        plt.text(0.95, 0.95, txt_disp, transform=ax.transAxes, ha="right", va="top")
        plt.legend(loc="right")
        plt.ylim(-0.05, max(y) * 1.1)
        plt.show()

    print(f"R² = {R_max}")


def T1_IR_func(time, T1, init_intensity, A):
    # fit T1 for inversion recovery measurement
    time = np.array(time, dtype=np.longdouble)
    T1 = np.array(T1, dtype=np.longdouble)
    return init_intensity * (1 - 2 * A * np.exp(-1 * time / T1))


def T1_SR_func(time, T1, init_intensity, A):
    # fit T1 for saturation recovery measurement
    time = np.array(time, dtype=np.longdouble)
    T1 = np.array(T1, dtype=np.longdouble)
    return init_intensity * (1 - A * np.exp(-1 * time / T1))


# Eventually want to make these functions fit in here better
def fit_T1_IR(
    save_dir,
    save_name,
    delay_data,
    intensity_data,
    labels=None,
    normalize=False,
    show_plot=True,
    colors=["red", "blue", "green"],
):
    """
    DESCRIPTION:
    Given delay and intensity data for a T1 inversion recovery experiment, extract out
    the T1 time constant in s

    PARAMETERS:
        save_dir: string
            Directory to save plot figure
        save_name: string
            Figure save file name
        delay_data: array of arrays
            List of delay data each acquistion was run at, for each resonance,
            i.e. [delay_para, delay_dia], where delay_para = [1,3,5,7,10,30,50,80]
        intensity_data: array of arrays
            List of the intensity values extracted from a component/ group of
            components after fitting the spectra
        labels: array of strings
            Label names for each component/ group of components
        normalize: boolean
            Whether or not to normalize the plot values.
    RETURNS: [T2_list, unscaled_percentages, scaled_percentages]
        T2_list: array of floats
            list of T1 constants corresponding to each component/ group of components
            specified in intensity_data, index-matched
        unscaled_percentages: array of floats
            list of unscaled molar percentages of each component/ group of components
            specified in intensity_data, index-matched
        scaled_percentages: array of floats
            list of T1 scaled molar percentages of each component/ group of components
            specified in intensity_data, index-matched
    """

    extracted_intensities = []
    initial_intensities = []
    T1_list = []
    # print('delay_data: {}'.format(delay_data))
    # print('intensity_data: {}'.format(intensity_data))
    # print('labels: {}'.format(labels))
    norm_factor = [intensity[-1] for intensity in intensity_data]

    plt, ax = format_plot(fig_size=(8, 8))

    if not labels:
        labels = [f"Feature {i+1}" for i in range(len(intensity_data))]

    for i in range(len(intensity_data)):
        label = labels[i]

        delay = np.array(delay_data)  # delay times in s
        intensity = np.array(intensity_data[i])
        if normalize:
            initial_intensities.append(intensity[-1])
            intensity = intensity / intensity[-1]
        else:
            initial_intensities.append(intensity[-1])

        plt.plot(delay, intensity, "o", color=colors[i], label=label)

        popt, pcov = curve_fit(
            T1_IR_func,
            delay,
            intensity,
            p0=[delay[-1], intensity[-1], 1],
            maxfev=5000,
            bounds=(0, [np.inf, np.inf, 2]),
        )
        T1 = popt[0]
        init_intensity = popt[1]
        A = popt[2]

        std_dev = np.sqrt(np.diag(pcov))
        T1_std_dev = std_dev[0]
        init_intensity_std_dev = std_dev[1]
        # A_std = std_dev[2]

        if normalize:
            abs_init_intensity = init_intensity * norm_factor[i]
            abs_init_intensity_std_dev = init_intensity_std_dev * norm_factor[i]
        else:
            abs_init_intensity = init_intensity
            abs_init_intensity_std_dev = init_intensity_std_dev

        extracted_intensities.append(abs_init_intensity)
        T1_list.append(T1)

        print("-----------------------------------------------")
        print("*****{} fitting results*****".format(label))
        print("-----------------------------------------------")
        print("T1 constant: {} s".format(np.round(T1, 6)))
        print("T1 constant std dev: {} s".format(np.round(T1_std_dev, 4)))
        print("Initial intensity: {}".format(np.round(abs_init_intensity, 0)))
        print(
            "Initial intensity std dev: {}".format(
                np.round(abs_init_intensity_std_dev, 0)
            )
        )
        print("A: {}".format(np.round(A, 4)))
        print("A std dev: {}".format(np.round(A, 4)))

        xfit = np.linspace(min(delay), max(delay))
        plt.plot(xfit, T1_IR_func(xfit, T1, init_intensity, A), "-", color="black")

    plt.xlabel("Time (s)")
    if normalize:
        plt.ylabel("Normalized Intensity (a.u.)")
    else:
        plt.ylabel("Intensity (a.u.)")
    plt.legend(prop={"size": 22}, frameon=False).set_draggable(True)

    text = "\n".join([f"{label} T1 = {t1:.3f} s" for label, t1 in zip(labels, T1_list)])
    ax.text(
        0.9,
        0.5,
        text,
        horizontalalignment="right",
        verticalalignment="center",
        transform=ax.transAxes,
    )

    plt.savefig(save_dir + save_name + ".png", bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show()
    plt.close()

    return T1_list


def fit_T1_SR(
    save_dir,
    save_name,
    delay_data,
    intensity_data,
    labels=None,
    normalize=False,
    show_plot=True,
    colors=["red", "blue", "green"],
):
    """
    Given delay and intensity data for a T1 saturation recovery experiment, extract out
    the T1 time constant in s

    PARAMETERS:
        save_dir: string
            Directory to save plot figure
        save_name: string
            Figure save file name
        delay_data: array of arrays
            List of delay data each acquistion was run at, for each resonance,
            i.e. [delay_para, delay_dia], where delay_para = [1,3,5,7,10,30,50,80]
        intensity_data: array of arrays
            List of the intensity values extracted from a component or group of
            components after fitting the spectra
        labels: array of strings
            Label names for each component/ group of components
        normalize: boolean
            Whether or not to normalize the plot values.
    RETURNS: [T2_list, unscaled_percentages, scaled_percentages]
        T2_list: array of floats
            list of T1 constants corresponding to each component or group of components
            specified in intensity_data, index-matched
        unscaled_percentages: array of floats
            list of unscaled molar percentages of each component or group of components
            specified in intensity_data, index-matched
        scaled_percentages: array of floats
            list of T1 scaled molar percentages of each component or group of components
            specified in intensity_data, index-matched
    """

    extracted_intensities = []
    initial_intensities = []
    T1_list = []
    # print('delay_data: {}'.format(delay_data))
    # print('intensity_data: {}'.format(intensity_data))
    # print('labels: {}'.format(labels))
    norm_factor = [intensity[-1] for intensity in intensity_data]

    plt, ax = format_plot(fig_size=(8, 8))

    if not labels:
        labels = [f"Feature {i+1}" for i in range(len(intensity_data))]

    for i in range(len(intensity_data)):
        label = labels[i]

        delay = np.array(delay_data)  # delay times in s
        intensity = np.array(intensity_data[i])
        if normalize:
            initial_intensities.append(intensity[-1])
            intensity = intensity / intensity[-1]
        else:
            initial_intensities.append(intensity[-1])

        plt.plot(delay, intensity, "o", color=colors[i], label=label)

        popt, pcov = curve_fit(
            T1_SR_func,
            delay,
            intensity,
            p0=[delay[-1], intensity[-1], 1],
            maxfev=5000,
            bounds=(0, [np.inf, np.inf, 1]),
        )
        T1 = popt[0]
        init_intensity = popt[1]
        A = popt[2]

        std_dev = np.sqrt(np.diag(pcov))
        T1_std_dev = std_dev[0]
        init_intensity_std_dev = std_dev[1]
        # A_std = std_dev[2]

        if normalize:
            abs_init_intensity = init_intensity * norm_factor[i]
            abs_init_intensity_std_dev = init_intensity_std_dev * norm_factor[i]
        else:
            abs_init_intensity = init_intensity
            abs_init_intensity_std_dev = init_intensity_std_dev

        extracted_intensities.append(abs_init_intensity)
        T1_list.append(T1)

        print("-----------------------------------------------")
        print("*****{} fitting results*****".format(label))
        print("-----------------------------------------------")
        print("T1 constant: {} s".format(np.round(T1, 6)))
        print("T1 constant std dev: {} s".format(np.round(T1_std_dev, 4)))
        print("Initial intensity: {}".format(np.round(abs_init_intensity, 0)))
        print(
            "Initial intensity std dev: {}".format(
                np.round(abs_init_intensity_std_dev, 0)
            )
        )
        print("A: {}".format(np.round(A, 4)))
        print("A std dev: {}".format(np.round(A, 4)))

        xfit = np.linspace(min(delay), max(delay))
        plt.plot(xfit, T1_SR_func(xfit, T1, init_intensity, A), "-", color="black")

    plt.xlabel("Time (s)")
    if normalize:
        plt.ylabel("Normalized Intensity (a.u.)")
    else:
        plt.ylabel("Intensity (a.u.)")
    plt.legend(prop={"size": 22}, frameon=False).set_draggable(True)

    text = "\n".join([f"{label} T1 = {t1:.3f} s" for label, t1 in zip(labels, T1_list)])
    ax.text(
        0.9,
        0.5,
        text,
        horizontalalignment="right",
        verticalalignment="center",
        transform=ax.transAxes,
    )

    if save_name:
        plt.savefig(save_name, bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show()
    plt.close()

    return T1_list


def fit_T1_spectra(
    data_files,
    delays,
    fit_range,
    components_list=None,
    comp_constraints=None,
    comp_names=None,
    normalize=False,
    comp_groups=[],
    group_names=[],
    fit_ssb=False,
    ssb_list=[],
    mas_freq=30000,
    print_results=True,
    show_plot=True,
    plot_init_fit=True,
    show_lgd=True,
    lgd_loc=0,
    lgd_fsize=22,
    save_name=None,
    summary_save_dir=None,
    fig_save_dir=None,
    data_color="black",
    fit_color="red",
    init_fit_color="green",
    comp_colors=None,
    group_comp_colors=["blue", "red"],
    saturation=False,
):
    """
    Given a set of T1 relaxation data, automatically fit all spectra, and extract of T1
    constants and scaled intensity values for all components

    PARAMETERS:
        data_files: list of strings
            List of files containing T1 relaxation experiments, with varying interpulse
            delays
        delays: array of floats
            List of delays for each of the spectra in data_files, index-matched
        normalize: boolean
            Whether or not to normalize the plot for T1 intensity decay
        **kwargs: key-word arguments
            key-word arguments corresponding to the 'fit' function. See 'fit' function
            for details
    RETURNS: [T1_list, unscaled_percentages, scaled_percentages]
        T1_list: array of floats
            list of T1 constants (in s) corresponding to each component or group of
            components specified in intensity_data, index-matched
        unscaled_percentages: array of floats
            list of unscaled molar percentages of each component or group of components
            specified in intensity_data, index-matched
        scaled_percentages: array of floats
            list of T2 scaled molar percentages of each component or group of components
            specified in intensity_data, index-matched
    """
    amplitudes = []
    comp_group_index = []
    comp_labels = []
    plt, ax = format_plot(
        fig_size=(8, 8),
    )

    for comp_name in comp_names:
        assigned_group = False
        for i, group in enumerate(comp_groups):
            if comp_name in group:
                comp_group_index.append(i)
                assigned_group = True
                if group_names[i] not in comp_labels:
                    comp_labels.append(group_names[i])
                else:
                    comp_labels.append(None)
        if not assigned_group:
            comp_group_index.append(-1)
            comp_labels.append(comp_name)
    # assigning colors to components
    colors = []
    default_colors = []
    for index in comp_group_index:
        if index != -1:
            colors.append(group_comp_colors[index])
        else:
            color = next(ax._get_lines.prop_cycler)["color"]
            colors.append(color)
            default_colors.append(color)
    if len(comp_groups) > 0:
        for i in range(len(comp_groups)):
            amplitudes.append([])
    else:
        for i in range(len(components_list)):
            amplitudes.append([])
    if len(comp_groups) > 0:
        colors = group_comp_colors
    plt.close()
    for i, data_file in enumerate(data_files):
        save_name = os.path.splitext(os.path.basename(data_file))[0].replace(".txt", "")

        print(data_file)
        print(delays[i])
        (
            freq_ppm_data,
            intensity_data,
            model_result,
            groupless_amplitudes,
            group_amplitudes,
        ) = fit(
            data_file=data_file,
            fit_range=fit_range,
            components_list=components_list,
            comp_constraints=comp_constraints,
            comp_names=comp_names,
            comp_groups=comp_groups,
            group_names=group_names,
            fit_ssb=fit_ssb,
            ssb_list=ssb_list,
            mas_freq=mas_freq,
            print_results=print_results,
            show_plot=show_plot,
            plot_init_fit=plot_init_fit,
            show_lgd=show_lgd,
            lgd_loc=lgd_loc,
            lgd_fsize=lgd_fsize,
            save_name=save_name,
            summary_save_dir=summary_save_dir,
            fig_save_dir=fig_save_dir,
            data_color=data_color,
            fit_color=fit_color,
            init_fit_color=init_fit_color,
            comp_colors=comp_colors,
            group_comp_colors=group_comp_colors,
        )
        if len(comp_groups) > 0:
            for i in range(len(comp_groups)):
                amplitudes[i].append(group_amplitudes[i])
        else:
            for i in range(len(components_list)):
                amplitudes[i].append(groupless_amplitudes[i])
    if len(comp_groups) > 0:
        delay_data = len(comp_groups) * [delays]
        labels = group_names
    else:
        delay_data = len(components_list) * [delays]
        labels = comp_names

    print(np.array(amplitudes).shape)
    print(amplitudes)

    func = fit_T1_SR if saturation else fit_T1_IR

    T1_list, unscaled_percentages, scaled_percentages = func(
        save_dir=fig_save_dir,
        save_name=save_name,
        delay_data=delay_data,
        intensity_data=amplitudes,
        labels=labels,
        normalize=normalize,
        colors=colors,
        show_plot=True,
    )
    return [T1_list, unscaled_percentages, scaled_percentages]
