import numpy as np
import matplotlib.pyplot as plt
import argparse
import jsonpickle
from scipy.integrate import cumtrapz

# Constants
au = 1.496e13  # Astronomical Unit in cm
msun = 1.989e33  # Solar mass in grams
mu = 2.35  # Mean molecular weight of the disk
k_B = 1.380649e-16  # Boltzmann constant in erg/K
G = 6.67430e-8  # Gravitational constant in cm^3/g/s^2
m_H = 1.6735575e-24  # Mass of hydrogen atom in grams
N_dissoc = 0
zeta_vals = [1e-19, 1e-17]  # Three zeta values for separate plots
m0_values = [1e-3, 1e-2, 1e-1]  # Three m0 values for separate curves
delta = lambda T: 3e-6 * T**(-0.5)  # Function of temperature

nr = 1000
nz = 1000

def temp_dartois(R,z,params_dartois):
    '''Compute temperature using the Dartois formulation
    Expect R and z to be in cgs'''

    temp_midplane = params_dartois["tmid_100"]*(R/100/au)**params_dartois["q_mid"] # midplane

    temp_atm = params_dartois["tatm_100"]*(R/100/au)**params_dartois["q_atm"] # atmosfera: strati superiori

    zq = params_dartois["z0"]*(R/100/au)**params_dartois["beta_z"]*au

    temp = temp_midplane + (temp_atm-temp_midplane)*np.sin(np.pi*np.abs(z)/2/zq)**2

    try:
        temp[np.abs(z)>zq]=temp_atm[np.abs(z)>zq]
    except IndexError:
        temp[np.abs(z)>zq]=temp_atm

    return temp

def temp_law(R, z, params_law):
    temp_midplane = params_law["tmid_100"]*(R/100/au)**params_law["q_mid"]
    temp_atm = params_law["tatm_100"]*(R/100/au)**params_law["q_atm"] # atmosfera: strati superiori
    zq = params_law["z0"]*(R/100/au)**params_law["beta"]*au
    temp = (temp_midplane**4 + 0.5*(1+np.tanh((z-params_law["alpha"]*zq)/zq))*temp_atm**4)**1/4

    try:
        temp[np.abs(z)>zq]=temp_atm[np.abs(z)>zq]
    except IndexError:
        temp[np.abs(z)>zq]=temp_atm

    return temp

def T_to_cs(T):
    return np.sqrt(k_B * T / mu / m_H)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the expected height of the emission surface")
    parser.add_argument("paramfile", help="JSON file with parameters for surface density and temperature profile")
    args = parser.parse_args()
    
    # Load parameters from JSON file
    with open(args.paramfile, 'r') as json_file:
        json_string = json_file.read()
        params = jsonpickle.decode(json_string, keys=True)

    r = np.linspace(1 * au, 1100 * au, nr)

    Omega = np.sqrt(params["mstar"] * msun * G / r ** 3)
    temp_midplane = temp_dartois(r, 0, params["params_dartois"])
    cs_midplane = T_to_cs(temp_midplane)
    H_midplane = cs_midplane / Omega

    fig, axes = plt.subplots(len(zeta_vals), 1, figsize=(12, 12), sharex=True)
    fig.suptitle(r"$z_{em}$ in funzione del raggio: caso NON isotermo", fontsize=25)


    colors = ['b', 'g', 'r']  # Blu, verde, rosso

    for idx, (ax, zeta) in enumerate(zip(axes, zeta_vals)):
        for color, m0 in zip(colors, m0_values):

            exponent = int(np.log10(m0))
            sigma_base = 1. / r * np.exp(-r / (params["rc"] * au))
            mass = np.trapz(2 * np.pi * r * sigma_base, r)
            sigma = sigma_base * m0 * msun / mass
            u=0
            for transition, linestyle in zip(params["transitions"], ['-', '--']):
                T_crit, crit_N = np.loadtxt(f'Ntau1_T_HCOP_J{transition}.txt', unpack=True)
                crit_N *= np.cos(np.deg2rad(params["inclination"]))

                rho_0 = sigma / H_midplane / np.sqrt(2 * np.pi)
                z_em = np.zeros_like(r)

                for ir, radius in enumerate(r):
                    z = np.linspace(0, 3*H_midplane[ir], nz)
                    temp_z = temp_dartois(radius, z, params["params_dartois"])
                    temp_z = np.maximum(temp_z, 10)

                    Omega_z2 = params["mstar"] * msun * G / (radius**2 + z**2)**1.5
                    c_s_z = T_to_cs(temp_z)

                    exponent = cumtrapz(Omega_z2**2 * z / c_s_z**2, z, initial=0)
                    rho = rho_0[ir] * (cs_midplane[ir] / c_s_z)**2 * np.exp(-exponent)
                    xHCO = np.sqrt((zeta*2.3 * m_H) / (delta(temp_z) * rho))
                    n_CO = xHCO * rho / mu /    m_H

                    N_crit_z = np.interp(temp_z, T_crit, crit_N, left=np.nan, right=np.nan)
                    tau_exact_integrand = xHCO * rho / mu / m_H / N_crit_z

                    try:
                        tau = -cumtrapz(tau_exact_integrand[::-1][0:] / np.cos(np.deg2rad(params["inclination"])), z[::-1][0:], initial=0)
                        index_tau1 = np.searchsorted(tau, 0.67) 
                        try:
                            z_em[ir] = z[::-1][index_tau1]
                        except IndexError:
                            z_em[ir] = 0
                    except ValueError as e:
                        if 0 == nz:
                            z_em[ir] = 0
                        else:
                            raise e
                if transition=="4-3":
                    u = 1
                else:
                    u = 0.4
                ax.semilogx(
                    r / au, z_em / r,
                    linestyle=linestyle,
                    linewidth=3,
                    color=color,
                    alpha=u
                )


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

    # Crea manualmente la legenda
    legend_elements = []
    for color, m0 in zip(colors, m0_values):
        exponent = int(np.log10(m0))
        legend_elements.append(plt.Line2D([], [], color=color, linestyle='-', label=rf"$m_0 = 10^{{{exponent}}}M_\odot: J=4-3\,(-), J=3-2\,(--)$"))

    axes[0].legend(handles=legend_elements, title=r"$m_0$ e transizioni", title_fontsize=17, fontsize=17, loc="upper left")


    # Show plot
    plt.tight_layout()
    plt.savefig("DA_slides.png")
    plt.show()