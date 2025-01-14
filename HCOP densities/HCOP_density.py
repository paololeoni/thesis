import numpy as np
import matplotlib.pyplot as plt

c = 3e10
h = 6.62e-27
k_b = 1.38e-16

# Funzione per calcolare la funzione di partizione (non approssimata)
def Part_funct(filename, Ts):

    temps = np.loadtxt(filename)
    J = np.arange(len(temps))  # I livelli corrispondenti a J
    
    # Meshgrid per fare il calcolo su ogni J e temperatura
    JJ, TT = np.meshgrid(J, Ts, indexing='ij')
    
    # Degenerazione dei livelli
    deg = 2 * JJ + 1
    
    # Termini della somma nella funzione di partizione
    terms = deg * np.exp(-temps[:, np.newaxis] / TT) # beta = 1 nelle unità  in cui sto lavorando
    
    # Somma su tutti i livelli
    Z = np.sum(terms, axis=0)
    
    return Z

# Funzione che calcola N (column density) utilizzando la funzione di partizione
def compute_N(Jup, Z, T):
    B0 = 89.189e9 / 2.  # inertia moment of the HCO+ molecule
    m0 = 29 * 1.66e-24  # HCO+ molecular mass

    if Jup == 4:
        nu = 356.7342230e9
        g2 = 9
        g1 = 7
        A = 3.626e-03
        El = 42.8
    elif Jup == 3:
        nu = 267.5576190e9
        g2 = 7
        g1 = 5
        A = 1.627e-03
        El = 25.68

    # Definizione dei parametri per il calcolo di N
    nl = g1 / Z * np.exp(-El / T)
    v_D = np.sqrt(2 * k_b * T / m0)
    phinu = c / nu * 1 / v_D / np.sqrt(np.pi)
    
    # Column density
    N = 1. / (c * c / 8 / np.pi / nu**2 * nl * g2 / g1 * A * (1 - np.exp(-h * nu / k_b / T)) * phinu)
    
    return N

fig, ax = plt.subplots(figsize=(7, 6))

for Jup in [4, 3]: 
    filename = 'HCOP_energies.txt'  # Nome del file di energia
    T = np.linspace(10, 1000, 10000)  # Array delle temperature
    
    Z = Part_funct(filename, T)
    N = compute_N(Jup, Z, T)
    ax.loglog(T, N, label=f'HCO+ J={Jup}-{Jup-1}')

ax.set_xlabel('Temperature [K]')
ax.set_ylabel('Column Density [g cm$^{-2}$]')
ax.set_title('Column Density for HCO+ Transitions')
ax.legend()

plt.tight_layout()
#plt.savefig("density_HCOP_transitions.png")

plt.show()

for Jup in [4, 3]:
    T = np.linspace(10, 1000, 10000)
    Z = Part_funct('HCOP_energies.txt', T)
    N = compute_N(Jup, Z, T)
    np.savetxt(f"Ntau1_T_HCOP_J{Jup}-{Jup-1}.txt", np.vstack([T, N]).T)
