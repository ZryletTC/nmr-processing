import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nmr_processing.leonmr import xf2, xf2_peak_pick


def read_t1ints(path, delay_offset=True, normalize=True):
    times = []
    ints = []

    with open(path, "r") as file:
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
        acqus = path.replace("t1ints.txt", "../../acqus")
        # print(f'acqus file: {acqus}')
        with open(acqus, "rb") as input:
            for line in input:
                # print(line.decode())
                if "##$P= (0..63)" in line.decode():
                    line = next(input)
                    linestr = line.decode()
                    P = linestr.strip("\n").split(" ")
                    P2 = float(P[2])
        times += (P2 * 1e-6) / 2
    # print('finished offset')

    # TODO: Convert t1ints indices to ppm positions
    positions = [0 for i in range(n_sites)]  # for now just doing the right len

    return times, ints, positions


def load_vdlist(path, delay_offset=True):
    str_arr = np.loadtxt(path, dtype=str)
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
        acqus = path.replace("vdlist", "acqus")
        with open(acqus, "rb") as input:
            for line in input:
                if "##$P= (0..63)" in line.decode():
                    line = next(input)
                    linestr = line.decode()
                    P = linestr.strip("\n").split(" ")
                    P2 = float(P[2])
        delays += (P2 * 1e-6) / 2
    return delays


def process_sir(
    exp_path,
    procno=1,
    peak_pos=[],
    f2l=3,
    f2r=-3,
    plot=True,
    delay_offset=True,
    regions=[],
):
    """
    Process a Selective Inversion Recovery experiment.
    Pass in the path to the experiment and this returns the data.

    Also works for t1 measurements or other pseudo-2D exps that use vdlist.
    """

    # Load vdlist and interpret suffixes
    vdlist = os.path.join(exp_path, "vdlist")
    delays = load_vdlist(vdlist, delay_offset=delay_offset)

    xAxppm, real_spectrum, params = xf2(exp_path, f2l=f2l, f2r=f2r)
    if peak_pos:
        ints = []
        for peak in peak_pos:
            ints.append(xf2_peak_pick(xAxppm, real_spectrum, peak_pos=peak, plot=plot))
        intensities = np.concatenate(ints, axis=1)
        positions = peak_pos
    elif regions:
        ints = []
        for f2l, f2r in regions:
            ints.append(np.trapz(xAxppm, real_spectrum))
        intensities = np.concatenate(ints, axis=1)
        positions = peak_pos
    else:
        intensities, positions = xf2_peak_pick(
            xAxppm, real_spectrum, prominence=[0.9, 1], plot=plot
        )

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
    """
    if matrix:
        matrix = np.array(matrix)
        assert matrix.shape == (sites, sites)
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
    assert len(R1_guesses) == sites
    mch_lines.extend(["", " ".join(map(str, R1_guesses))])
    if tp2:
        mch_lines.append(" ".join(map(str, np.ones(sites, dtype=np.int8))))

    if M_f_guesses:
        assert len(M_f_guesses) == sites
    else:
        M_f_guesses = np.ones(sites)  # Final intensities are normalized, so 1
    mch_lines.extend(["", " ".join(map(str, M_f_guesses))])
    if tp2:
        mch_lines.append(" ".join(map(str, np.ones(sites, dtype=np.int8))))

    if M_0_guesses:
        assert len(M_0_guesses) == sites
    else:
        # This is a bad guess, it should be more like 1, -1
        M_0_guesses = np.ones(sites)
    mch_lines.extend(["", " ".join(map(str, M_0_guesses))])
    if tp2:
        mch_lines.append(" ".join(map(str, np.ones(sites, dtype=np.int8))))

    if k_guesses:
        assert len(k_guesses) == processes
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
    assert intensities.shape[0] == numpoints

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
    procno=1,
    peak_pos=[],
    peak_names=[],
    plot=False,
    T1_values=[],
    tp2=False,
    k_guesses=[],
    matrix=[],
    M_0_guesses=[],
    M_f_guesses=[],
    ints=True,
):

    if ints:
        t1ints = os.path.join(exp_path, f"pdata/{procno}/t1ints.txt")
        delays, ints, positions = read_t1ints(t1ints)
    else:
        delays, ints, positions = process_sir(
            exp_path, procno=procno, peak_pos=peak_pos, plot=plot
        )

    exp_name = os.path.basename(os.path.dirname(exp_path))
    exp_no = int(os.path.basename(exp_path))
    title = f"Extracted from {exp_name} exp no {exp_no}"
    if peak_names:
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

    fig, ax = plt.subplots()
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
