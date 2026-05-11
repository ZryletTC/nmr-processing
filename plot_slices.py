import matplotlib.pyplot as plt

from nmr_processing.leonmr import plot_slice

filepath = '/Users/tylerpennebaker/BoxSync/wp6_exsy/EXSYstudy/500.TP-2024.12.02_7Li_LZC+LPSC/12'

def custom_plot(sl):
    fig, ax = plot_slice(filepath, slice_idx=sl, f2l=3, f2r=-3, procno=2)
    ax.set_ylim(-0.2,1.1)
    ax.set_yticks([])
    ax.set_xlabel('$^7$Li shift / ppm')
    plt.tight_layout()
    plt.savefig(filepath+f'_{sl}_2.pdf')

for i in range(16):
    custom_plot(i)
