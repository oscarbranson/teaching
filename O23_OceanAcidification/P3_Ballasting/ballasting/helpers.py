import numpy as np


day_2_seconds = 24 * 60 * 60
seconds_2_day = 1 / day_2_seconds

def calc_wsink (r, rho_p, rho_f=1023.6):
    g = 9.80665  # m s-2
    mu = 1e-3  # kg m-1 s-1
    
    return 2 * g * r**2 * (rho_p - rho_f) / (9 * mu)  # m s-1

def calc_rho_f(z=0, rho_surf=1023.6, beta=5e-6):
    # z = depth in m
    # rho_surf = density at bottom of photic zone (kg m-3)
    # beta = change in seawater density with depth (m-1)
    return rho_surf * (1 + beta * z)

def calc_rho_p(F=0, rho_org=1280., rho_carb=2710.):
    return rho_org * (1 - F) + rho_carb * F

def calc_Rdiss(Omega, k=4.5, n=2.7):
    if Omega > 1:
        return 0

    r = np.real(k * (1 - Omega)**n)
    return max(0, r)

def calc_Rremin(Rmax, O2, K_M=5):
    r = (O2 * Rmax) / (K_M + O2)
    if isinstance(r, np.ndarray):
        r[r < 0] = 0
        r[O2 < 0] = 0
        return r
    else:
        return max(0, r)

def sphere_vol_from_radius(r):
    return 4 / 3 * np.pi * r**3

def sphere_radius_from_vol(V):
    if V < 0:
        return 0
    return (V * 3 / 4 / np.pi)**(1/3)