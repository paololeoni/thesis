import numpy as np
import matplotlib.pyplot as plt
import argparse
import jsonpickle
from scipy.special import erfcinv

au = 1.496e13
msun = 1.989e33 
mu = 2.35  # Peso molecolare medio del disco
k_B = 1.380649e-16      
G = 6.67430e-8          
m_H = 1.6735575e-24     
delta = lambda T: 3e-6 * T**(-0.5)  
m0_values = [1e-3, 1e-2, 1e-1]
zeta_vals = [1e-19, 1e-17]

nr = 10000
nz = 10000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Given a surface density and temperature profile, compute the expected height of the emission surface")
    parser.add_argument("paramfile", help='JSON file with the parameters of the surface density and temperature profile')
    args = parser.parse_args()
    
    with open(args.paramfile, 'r') as json_file:
        json_string = json_file.read()
        params = jsonpickle.decode(json_string, keys=True)

    r = np.linspace(1 * au, 1000 * au, nr)
    fig, axes = plt.subplots(len(zeta_vals), 1, figsize=(12, 12), sharex=True)
    fig.suptitle(r"$z_{em}$ in funzione del raggio: caso isotermo", fontsize=25)
    Omega = np.sqrt(params["mstar"] * msun * G / r ** 3)
    z_em = np.zeros_like(r)

    hr = params["h0"] * (r / params["rpivot"] / au) ** params['plh']
    H = hr * r
    cs = H * Omega
    temp = mu * m_H * cs * cs / k_B
    temp = np.maximum(temp, 10)
    colors = ['b', 'g', 'r']

    for idx, (ax, zeta) in enumerate(zip(axes, zeta_vals)):
        for color, m0 in zip(colors, m0_values):

            exponent = int(np.log10(m0))
            sigma = 1. / r * np.exp(-r / (params["rc"] * au))
            mass = np.trapz(2 * np.pi * r * sigma, r)
            sigma *= m0 * msun / mass

            for transition, linestyle in zip(params["transitions"], ['-', '--']):
                T_crit, crit_N = np.loadtxt(f'Ntau1_T_HCOP_J{transition}.txt', unpack=True)
                crit_N *= np.cos(np.deg2rad(params["inclination"]))

                rho_0 = sigma / H / np.sqrt(2 * np.pi)

                crit_N_r = np.interp(temp, T_crit, crit_N, left=np.nan, right=np.nan)
                arg = (crit_N_r / H) * np.sqrt((delta(temp) * mu * m_H) / (2 * zeta * rho_0 * np.pi))
                z_em = 2 * H * erfcinv(arg)

                ax.semilogx(
                    r / au, z_em / r,
                    linestyle=linestyle,
                    linewidth=3,
                    color=color
                )

        # Imposta i testi descrittivi per ogni sottoplot
        ax.text(0.02, 0.95 if idx > 0 else 0.05, rf"$\zeta = 10^{{{int(np.log10(zeta))}}} \, \mathrm{{s}}^{{-1}}$", 
                transform=ax.transAxes, fontsize=25,
                verticalalignment='top' if idx > 0 else 'bottom', horizontalalignment='left', 
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.2))

        ax.set_ylabel(r'$z_{\mathrm{em}} / r$', fontsize=30)
        ax.set_ylim(0, 0.7)
        ax.tick_params(axis='y', labelsize=25)

        if idx == len(zeta_vals) - 1:
            ax.set_xlabel('r [au]', fontsize=30)
            ax.tick_params(axis='x', labelsize=25)
        else:
            ax.label_outer()
        # ax.set_xlabel('r [au]', fontsize=25)
        # ax.tick_params(axis='x', labelsize=20)
        # ax.set_ylabel(r'$z_{\mathrm{em}} / r$', fontsize=25)
        # ax.set_ylim(0, 0.7)
        # ax.tick_params(axis='y', labelsize=20)
        # if idx == len(zeta_vals)-2:
        #     ax.set_ylabel(r'$z_{\mathrm{em}} / r$', fontsize=25)
        #     ax.set_ylim(0, 0.7)
        #     ax.tick_params(axis='y', labelsize=20)
        # else:
        #     ax.label_outer()
    # Crea manualmente la legenda
    legend_elements = []
    for color, m0 in zip(colors, m0_values):
        exponent = int(np.log10(m0))
        legend_elements.append(plt.Line2D([], [], color=color, linestyle='-', label=rf"$m_0 = 10^{{{exponent}}}M_\odot: J=4-3\,(-), J=3-2\,(--)$"))

    axes[0].legend(handles=legend_elements, title=r"$m_0$ e transizioni", title_fontsize=17, fontsize=17, loc="upper left")

    plt.tight_layout()
    plt.savefig("slides_100.png")
    plt.show()
