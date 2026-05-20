# pylint: skip-file
# flake8: noqa
"""
NMR data fitting utilities.

This module is reserved for fitting functions for NMR data.

TODO: Make better fitting, compare with Vincent's and Leo's code. Leo's code is
      currently in fitting.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from ssNMR.fitting import fit

# TODO: remove dependence on ssNMR
from ssNMR.formatting import format_plot


def T2_Fit(x, y, t0=0.5, c0=1, beta0=0.5, fit_type="default", show_all=False):
    """
    Fit T2 relaxation decay data using monoexponential, biexponential,
    or stretched exponential models.

    Parameters
    ----------
    x : array-like
        Echo delay times.
    y : array-like
        Normalized intensity measurements.
    t0 : float, default: 0.5
        Initial guess for the relaxation time constant.
    c0 : float, default: 1
        Initial amplitude guess for the stretched exponential model.
    beta0 : float, default: 0.5
        Initial stretching exponent guess.
    fit_type : str, optional
        If set to a specific model name, that model is used. Otherwise the highest-R²
        model is selected.
    show_all : bool, default: False
        If True, plot all three candidate models. Otherwise, plot only the selected
        model.
    """

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
    if fit_type != "default":
        method_choice = fit_types.index(fit_type)
    # print(All_R, method_choice)

    if fit_type == "default":
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

    if show_all:
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


def T1_IR_func(time, T1, full_intensity, A):
    """
    T1 relaxation model for inversion recovery experiments.

    Parameters
    ----------
    time : array-like
        Delay times in seconds.
    T1 : float or array-like
        Longitudinal relaxation time constant(s) in seconds.
    full_intensity : float
        Initial signal intensity.
    A : float
        Inversion factor.

    Returns
    -------
    np.ndarray
        Modeled intensity values.
    """

    time = np.array(time, dtype=np.longdouble)
    T1 = np.array(T1, dtype=np.longdouble)
    return full_intensity * (1 - 2 * A * np.exp(-1 * time / T1))


def T1_SR_func(time, T1, full_intensity, A):
    """
    T1 relaxation model for saturation recovery experiments.

    Parameters
    ----------
    time : array-like
        Delay times in seconds.
    T1 : float or array-like
        Longitudinal relaxation time constant(s) in seconds.
    full_intensity : float
        Initial signal intensity.
    A : float
        Saturation factor.

    Returns
    -------
    np.ndarray
        Modeled intensity values.
    """

    time = np.array(time, dtype=np.longdouble)
    T1 = np.array(T1, dtype=np.longdouble)
    return full_intensity * (1 - A * np.exp(-1 * time / T1))


# Eventually want to make these functions fit in here better
def fit_T1_IR(
    delay_data,
    intensity_data,
    labels=None,
    normalize=False,
    show_plot=True,
    colors=["red", "blue", "green"],
    save_path=None,
):
    """
    Fit inversion recovery data and extract T1 time constants.

    Parameters
    ----------
    delay_data : list of array-like
        Delay times for each component in seconds.
    intensity_data : list of array-like
        Intensity values to fit for each component.
    labels : list of str, optional
        Labels for each component. If not specified, components will be labeled as
        numbered features.
    normalize : bool, default: False
        If True, normalize intensity values before fitting.
    show_plot : bool, default: True
        If True, display the fit plot.
    colors : list, default: ["red", "blue", "green"]
        Plot colors for each component, as specified by `matplotlib`.
    save_path : str, optional
        Path to save the plot figure. If not specified, the figure is not saved.

    Returns
    -------
    list
        List of T1 relaxation times in seconds.
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
        full_intensity = popt[1]
        A = popt[2]

        std_dev = np.sqrt(np.diag(pcov))
        T1_std_dev = std_dev[0]
        full_intensity_std_dev = std_dev[1]
        # A_std = std_dev[2]

        if normalize:
            abs_full_intensity = full_intensity * norm_factor[i]
            abs_full_intensity_std_dev = full_intensity_std_dev * norm_factor[i]
        else:
            abs_full_intensity = full_intensity
            abs_full_intensity_std_dev = full_intensity_std_dev

        extracted_intensities.append(abs_full_intensity)
        T1_list.append(T1)

        print("-----------------------------------------------")
        print("*****{} fitting results*****".format(label))
        print("-----------------------------------------------")
        print("T1 constant: {} s".format(np.round(T1, 6)))
        print("T1 constant std dev: {} s".format(np.round(T1_std_dev, 4)))
        print("Full intensity: {}".format(np.round(abs_full_intensity, 0)))
        print(
            "Initial intensity std dev: {}".format(
                np.round(abs_full_intensity_std_dev, 0)
            )
        )
        print("A: {}".format(np.round(A, 4)))
        print("A std dev: {}".format(np.round(A, 4)))

        xfit = np.linspace(min(delay), max(delay))
        plt.plot(xfit, T1_IR_func(xfit, T1, full_intensity, A), "-", color="black")

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
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show()
    plt.close()

    return T1_list


def fit_T1_SR(
    delay_data,
    intensity_data,
    labels=None,
    normalize=False,
    show_plot=True,
    colors=["red", "blue", "green"],
    save_path=None,
):
    """
    Fit saturation recovery data and extract T1 time constants.

    Parameters
    ----------
    delay_data : list of array-like
        Delay times for each component in seconds.
    intensity_data : list of array-like
        Intensity values to fit for each component.
    labels : list of str, optional
        Labels for each component. If not specified, components will be labeled as
        numbered features.
    normalize : bool, default: False
        If True, normalize intensity values before fitting.
    show_plot : bool, default: True
        If True, display the fit plot.
    colors : list, default: ["red", "blue", "green"]
        Plot colors for each component, as specified by `matplotlib`.
    save_path : str, optional
        Path to save the plot figure. If not specified, the figure is not saved.

    Returns
    -------
    list
        List of T1 relaxation times in seconds.
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
        full_intensity = popt[1]
        A = popt[2]

        std_dev = np.sqrt(np.diag(pcov))
        T1_std_dev = std_dev[0]
        full_intensity_std_dev = std_dev[1]
        # A_std = std_dev[2]

        if normalize:
            abs_full_intensity = full_intensity * norm_factor[i]
            abs_full_intensity_std_dev = full_intensity_std_dev * norm_factor[i]
        else:
            abs_full_intensity = full_intensity
            abs_full_intensity_std_dev = full_intensity_std_dev

        extracted_intensities.append(abs_full_intensity)
        T1_list.append(T1)

        print("-----------------------------------------------")
        print("*****{} fitting results*****".format(label))
        print("-----------------------------------------------")
        print("T1 constant: {} s".format(np.round(T1, 6)))
        print("T1 constant std dev: {} s".format(np.round(T1_std_dev, 4)))
        print("Full intensity: {}".format(np.round(abs_full_intensity, 0)))
        print(
            "Initial intensity std dev: {}".format(
                np.round(abs_full_intensity_std_dev, 0)
            )
        )
        print("A: {}".format(np.round(A, 4)))
        print("A std dev: {}".format(np.round(A, 4)))

        xfit = np.linspace(min(delay), max(delay))
        plt.plot(xfit, T1_SR_func(xfit, T1, full_intensity, A), "-", color="black")

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

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
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
    Fit a series of T1 spectra and extract relaxation times for all components.

    Parameters
    ----------
    data_files : list of strings
        List of files containing T1 relaxation experiments, with varying interpulse
        delays
    delays : array of floats
        List of delays for each of the spectra in data_files, index-matched
    normalize : boolean
        Whether or not to normalize the plot for T1 intensity decay
    **kwargs : key-word arguments
        key-word arguments corresponding to the `fit` function. See `fit` function
        for details
    Returns
    -------
    T1_list : array of floats
        list of T1 constants (in s) corresponding to each component or group of
        components specified in intensity_data, index-matched
    unscaled_percentages : array of floats
        list of unscaled molar percentages of each component or group of components
        specified in intensity_data, index-matched
    scaled_percentages : array of floats
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
