import numpy as np
import matplotlib.pyplot as plt

from .helpers import calc_rho_f, seconds_2_day

units = {
    'z': (1, 'm'),
    'R_z': (1, '$m~s^{-1}$'),
    'r': (1e6, '$\mu m$'),
    'P': (1, 'bar'),
    'DIC': (1, '$\mu mol~kg_{SW}^{-1}$'),
    'TA': (1, '$\mu mol~kg_{SW}^{-1}$'),
    'O2': (1, '$\mu mol~kg_{SW}^{-1}$'),
    'Fc': (100, '%'),
    'rho_p': (1, '$kg~m^{-3}$'),
    'POC': (1, '$\mu mol~kg_{SW}^{-1}$'),
    'PIC': (1, '$\mu mol~kg_{SW}^{-1}$'),
    'PC': (1, '$\mu mol~kg_{SW}^{-1}$'),
    'CO3': (1, '$\mu mol~kg_{SW}^{-1}$'),
    'Omega': (1, '$\Omega$'),
    'R_diss': (1, '$\mu mol~kg_{SW}^{-1}~s^{-1}$'),
    'R_remin': (1, '$\mu mol~kg_{SW}^{-1}~s^{-1}$'),
    't': (seconds_2_day, 'days'),
    'rho_sw': (1, '$kg~m^{-3}$'),
}



def models(ms, pvars=['r', 'R_z', 'Fc', 'rho_p', 'R_remin', 'DIC', 'O2', 'Omega', 'R_diss', 'TA', 'PIC', 'POC'], model_labels=None):
    
    if isinstance(ms, dict):
        ms = [ms]
    if model_labels is None:
        model_labels = [f'{i+1:.0f}' for i in range(len(ms))]

    fig, axs = plt.subplots(1, len(pvars), figsize=[len(pvars) * 2, 4], sharey=True, constrained_layout=True)
    
    for v, ax in zip(pvars, axs):
        mult, unit = units[v]
        for lab, m in zip(model_labels, ms):
            ax.plot(m[v] * mult, m['z'], label=lab)    
            ax.axvline(m[v][1] * mult, ls='dashed', alpha=0.4)
    
        ax.set_xlabel(f'{v}\n{unit}')

    ax.invert_yaxis()
    axs[0].set_ylabel('z (m)')
    axs[0].legend()
    
    if 'Omega' in pvars:
        i = np.argwhere(np.array(pvars) == 'Omega')[0,0]
        axs[i].axvline(1, color=(1,0,0,0.3))
        
    if 'rho_p' in pvars:
        i = np.argwhere(np.array(pvars) == 'rho_p')[0,0]
        axs[i].set_ylim(axs[i].get_ylim())
        zn = np.linspace(*axs[i].get_ylim())
        axs[i].plot(calc_rho_f(zn), zn, color=(1,0,0,0.3))
        
    return fig, ax