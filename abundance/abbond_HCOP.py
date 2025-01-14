import numpy as np
import matplotlib.pyplot as plt
import argparse
import jsonpickle
from scipy.integrate import cumtrapz

G = 6.67430e-8          
k_B = 1.380649e-16      
m_H = 1.6735575e-24     
au = 1.496e13           
msun = 1.989e33 
mu = 2.35 
zeta = 1e-18
delta = lambda T: 3e-6 * T**(-0.5)  

parser = argparse.ArgumentParser(description="Given a density profile, compute the HCOP abundance ")

# richiede un argomento quando viene runnato, in questo caso BOOOOOHH
parser.add_argument("paramfile", help='JSON file with the parameters of the surface density and temperature profile')
    
args = parser.parse_args() # contiene tutti i parametri che ho aggiunto sopra (FILE)
with open(args.paramfile,'r') as json_file:
    json_string=json_file.read() # memorizzo ciò che leggo in questa stringa json
    params=jsonpickle.decode(json_string,keys=True) # converte la stringa json in un dizionario 'params'

R_vals = np.linspace(1 * au, 400 * au, 5000)  
Z_vals = np.linspace(0, 4, 5000) 

#tutto questo è perchè se no trapz nella funzione generica non funziona
sigma = 1 / R_vals * np.exp(-R_vals / (params["rc"]*au))
mass = np.trapz(2 * np.pi * R_vals * sigma, R_vals)  
sigma_normalized = sigma * params["m0"] * msun / mass 


def sound_speed(R):
    hr = params["h0"] * (R / (params["rpivot"]*au))**params["plh"]
    H = hr * R
    Omega = np.sqrt(params["mstar"] * msun*G / R**3)
    cs = H * Omega
    return cs

def T_to_cs(T): # valuto la velocità del suono ad una specifica temperatura
    return np.sqrt(k_B*T/mu/m_H)

if params["temp_type"] == "vert_iso":

    def temperature(R):
        cs = sound_speed(R)
        temp = mu * m_H * cs**2 / k_B
        return np.maximum(temp, 10)
    
    def density1(R, Z):
        H = params["h0"] * (R / (params["rpivot"]*au))**params["plh"] * R
        sigma_R = np.interp(R, R_vals, sigma_normalized) #interpolo
        rho_0 = sigma_R / (H * np.sqrt(2 * np.pi))
        return rho_0 * np.exp(-Z**2 / (2 * H**2))

    def ionization_fraction(R, Z):
        T = temperature(R)
        rho = density1(R, Z)
        n_n = rho / (2 * m_H)       
        xi = np.sqrt(zeta / (delta(T) * n_n))
        return np.log10(xi)
    
    Z_grid, R_grid = np.meshgrid(Z_vals, R_vals, indexing='ij')
    log_xi = ionization_fraction(R_grid, Z_grid * (params["h0"] * (R_grid / (params["rpivot"] * au)) ** params["plh"] * R_grid))
    #print(log_xi[1])
    # check sulle densità
    # print(str(np.sqrt(temperature(R_vals))/density1(R_vals, Z_vals))+"\n")
    # print("\n"+str(((3/10**24)/(2*m_H))*10**(2*(18+log_xi[1]))))
elif params["temp_type"] == "dartois":

    #import pdb; pdb.set_trace()
     
    def temperature(R, Z, params_dartois):

        temp_midplane = params_dartois["tmid_100"]*(R/100/au)**params_dartois["q_mid"] # midplane

        temp_atm = params_dartois["tatm_100"]*(R/100/au)**params_dartois["q_atm"] # atmosfera: strati superiori

        zq = params_dartois["z0"]*(R/100/au)**params_dartois["beta_z"]*au

        temp = temp_midplane + (temp_atm-temp_midplane)*np.sin(np.pi*np.abs(Z)/2/zq)**2
    
        try:
            temp[np.abs(Z)>zq]=temp_atm[np.abs(Z)>zq]
        except IndexError:
            temp[np.abs(Z)>zq]=temp_atm

        return temp

    temp_z = temperature(R_vals,Z_vals,params["params_dartois"])
    temp_z = np.maximum(temp_z,10) 
    c_s_z = T_to_cs(temp_z)
    Omega_z2 = params["mstar"]*msun*G/(R_vals**2+Z_vals**2)**1.5  
    exponent = cumtrapz(Omega_z2*Z_vals/c_s_z**2,Z_vals,initial=0)    

    def density2(R, Z):

        temp_midplane = temperature(R,0,params["params_dartois"])
        cs_midplane = T_to_cs(temp_midplane)
        Omega = np.sqrt(params["mstar"] * msun*G / R**3)
        H_midplane = cs_midplane/Omega
        sigma_R = np.interp(R, R_vals, sigma_normalized) #interpolo
        rho_0 = sigma_R/H_midplane/np.sqrt(2*np.pi)
        
        exponent2=np.interp(R, R_vals, exponent)
        rho = rho_0 * (cs_midplane/c_s_z)**2 * np.exp(-exponent2)

        return rho

    def ionization_fraction(R, Z):
        T = temperature(R, Z, params["params_dartois"])
        rho = density2(R, Z)
        n_n = rho / (2 * m_H)
        xi = np.sqrt(zeta / (delta(T) * n_n))
        return np.log10(xi)    

    Z_grid, R_grid = np.meshgrid(Z_vals, R_vals, indexing='ij')
    log_xi = ionization_fraction(R_grid, Z_grid * (params["h0"] * (R_grid / (params["rpivot"] * au)) ** params["plh"] * R_grid))

    # check sulle densità
    # print(str(np.sqrt(temperature(R_vals, Z_vals, params["params_dartois"])[1])/density2(R_vals, Z_vals))+"\n")
    # print("\n"+str(((3/10**24)/(2*m_H))*10**(2*(18+log_xi[1]))))
if params["temp_type"] == "vert_iso":
    np.savetxt("ion_iso.txt", ionization_fraction(R_vals, Z_vals * (params["h0"] * (R_vals / (params["rpivot"] * au)) ** params["plh"] * R_vals)) , fmt="%.5f") 
elif params["temp_type"] == "dartois":
    np.savetxt("ion_dartois.txt", ionization_fraction(R_vals, Z_vals * (params["h0"] * (R_vals / (params["rpivot"] * au)) ** params["plh"] * R_vals)) , fmt="%.5f") 

# # Plotting
plt.figure(figsize=(10, 6))
contour_levels = 10  
to_plot=log_xi
pcolor = plt.pcolormesh(R_vals / au, Z_vals, to_plot , shading='auto', cmap="viridis", vmin=to_plot.min(), vmax=to_plot.max())
plt.contour(R_vals / au, Z_vals, to_plot, levels=contour_levels, colors='black', linewidths=0.8, linestyles='--')

# barra dei colori
cbar = plt.colorbar(pcolor)
cbar.ax.tick_params(labelsize=18)
cbar.set_label(r"$\log_{10}(x_{HCO})$", fontsize=20)
# Impostazioni per etichette e titoli
plt.xlabel("r [AU]", fontsize=22)
plt.ylabel("z/H", fontsize=22)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.tight_layout()
if params["temp_type"] == "vert_iso":
    plt.savefig("500iso_ion_frac.png")
elif params["temp_type"] == "dartois":
    plt.savefig("50dartois_ion_frac.png")

plt.show()







