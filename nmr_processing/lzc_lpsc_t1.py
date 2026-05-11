import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nmr_processing.leonmr import fit_T1_IR, xf2
from nmr_processing.process_sir import process_sir

# folder = '/Users/tylerpennebaker/BoxSync/wp6_exsy/EXSYstudy/500.TP-2024.12.02_7Li_LZC+LPSC'
# T1_exps = [11, 21, 31, 41]

folder = '/Users/tylerpennebaker/Library/CloudStorage/Box-Box/LGES subgroup/FRL II/Raw data/Selective Inversion/NMR Data/data/500.TP-2025.03.26_6Li_LZC+LPSC'
T1_exps=[16]

for T1_exp in T1_exps:
    T1_path = os.path.join(folder, str(T1_exp))

    ppm, spectra, _ = xf2(T1_path, f2l=3, f2r=-3)

    df = pd.DataFrame.from_dict({'ppm': ppm, 'spec': spectra[-1]})  # last spectrum

    idx = df[df['ppm'] > 0].idxmax()['spec']
    lpsc_pos = df['ppm'][idx]

    idx = df[df['ppm'] < 0].idxmax()['spec']
    lzc_pos = df['ppm'][idx]

    delays, ints, positions = process_sir(T1_path, peak_pos=[lpsc_pos, lzc_pos], plot=False)

    T1_list = fit_T1_IR(
        save_dir=T1_path,
        save_name='/t1_fit',
        delay_data=delays,
        intensity_data=ints.transpose(),
        labels=['LPSC', 'LZC']
    )

    print(T1_list)
    with open(T1_path+'/LPSC_T1.txt', 'w') as f:
        f.write(str(T1_list[0]))
    with open(T1_path+'/LZC_T1.txt', 'w') as f:
        f.write(str(T1_list[1]))
