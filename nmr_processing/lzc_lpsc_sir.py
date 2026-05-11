import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nmr_processing.leonmr import xf2
from nmr_processing.process_sir import exp_to_cifit, plot_cifit_csv

# folder = '/Users/tylerpennebaker/BoxSync/wp6_exsy/EXSYstudy/500.TP-2024.12.02_7Li_LZC+LPSC'
# exp_nums = [12, 20, 30, 40]
# T1_exps = [11, 21, 31, 41]

folder = '/Users/tylerpennebaker/Library/CloudStorage/Box-Box/LGES subgroup/FRL II/Raw data/Selective Inversion/NMR Data/data/500.TP-2025.03.26_6Li_LZC+LPSC'
exp_nums = [32]
T1_exps=[16]

for exp_num, T1_exp in zip(exp_nums, T1_exps):
    exp_path = os.path.join(folder, str(exp_num))
    T1_path = os.path.join(folder, str(T1_exp))
    out_path = exp_path

    # ppm, spectra, _ = xf2(exp_path, f2l=3, f2r=-3)

    # df = pd.DataFrame.from_dict({'ppm': ppm, 'spec': spectra[-1]})

    # idx = df[df['ppm'] > 0].idxmax()['spec']
    # lpsc_pos = df['ppm'][idx]

    # idx = df[df['ppm'] < 0].idxmax()['spec']
    # lzc_pos = df['ppm'][idx]

    # with open(T1_path+'/LPSC_T1.txt', 'r') as f:
    #     lpsc_t1 = float(f.read())
    # with open(T1_path+'/LZC_T1.txt', 'r') as f:
    #     lzc_t1 = float(f.read())

    # # exp_to_cifit(exp_path, out_path, peak_pos=[lpsc_pos, lzc_pos],
    # #              peak_names=['LPSC', 'LZC'], tp2=True,
    # #              T1_values=[lpsc_t1, lzc_t1], k_guesses=[0.5],
    # #              matrix=[[0, .24], [1, 0]])
    # exp_to_cifit(exp_path, out_path, peak_pos=[lpsc_pos, lzc_pos],
    #              peak_names=['LPSC', 'LZC'], tp2=True,
    #              T1_values=[lpsc_t1, lzc_t1], k_guesses=[0.5],
    #              matrix=[[0, 1], [1, 0.83]], M_0_guesses=[1, -1])

    args = ['/Users/tylerpennebaker/bin/cifit2', str(exp_num)]
    subprocess.run(args, cwd=folder)  # capture and check for "not converged"
    # stdout=subprocess.DEVNULL)

    plot_cifit_csv(out_path+'.csv', names=['LPSC', 'LZC'])


# T1s from exp
# populations imbalance
