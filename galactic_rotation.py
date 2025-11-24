"""
SCRIPT 06: GALACTIC ROTATION CURVE FIT (3D MODEL)
-------------------------------------------------
Paper Reference: Sec. 3.9 (Universal Galactic Rotation Curves)

PURPOSE:
Fits observational rotation curve data (NGC 6503) using a physical 3D model 
derived from the scalar field theory.

MODELS:
1. Baryonic Component: Freeman's analytical solution for a thin exponential disk
   in 3D Newtonian gravity (involving Bessel functions I and K).
2. Scalar Vacuum Component: A spherical halo arising from the scalar field 
   condensate, with density profile rho ~ 1/(1 + (r/Rs)^2).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import i0, k0, i1, k1

# --- OBSERVATIONAL DATA: NGC 6503 ---
# Source: SPARC (Spitzer Photometry & Accurate Rotation Curves) Database
# Radius [kpc]
r_dat = np.array([0.28, 0.83, 1.39, 1.94, 2.50, 3.05, 3.61, 4.16, 4.72, 
                  5.27, 5.83, 6.38, 6.94, 7.50, 8.05, 8.61, 9.16, 9.72, 
                  10.27, 10.83, 11.38, 11.94, 12.49, 13.05, 15.0, 18.0, 20.0])

# Velocity [km/s]
v_dat = np.array([25.0, 58.0, 82.0, 96.0, 104.0, 109.0, 112.0, 113.0, 114.0,
                  115.0, 115.5, 116.0, 116.0, 116.5, 116.8, 117.0, 117.0, 117.2,
                  117.5, 117.5, 117.6, 117.8, 118.0, 118.0, 118.5, 119.0, 119.2])

# Error estimates (Uniform 3 km/s for simplicity)
v_err = np.ones_like(v_dat) * 3.0

# --- THEORETICAL MODELS ---

def freeman_disk_velocity(r, V_disk_scale, R_d):
    """
    Velocity squared of a thin exponential disk in 3D (Freeman 1970).
    Surface density Sigma(r) = Sigma_0 * exp(-r/Rd)
    v^2(r) = 4 * pi * G * Sigma_0 * Rd * y^2 * [I0(y)K0(y) - I1(y)K1(y)]
    where y = r / (2*Rd).
    We absorb constants into V_disk_scale^2.
    """
    y = r / (2.0 * R_d)
    # Prevent division by zero or overflow at r=0
    y[y < 1e-6] = 1e-6
    
    # Bessel function term (Baryonic contribution)
    bessel_term = i0(y) * k0(y) - i1(y) * k1(y)
    
    # V_disk_scale represents sqrt(4*pi*G*Sigma_0*Rd)
    v2 = V_disk_scale**2 * (y**2) * bessel_term * 4.0
    return np.abs(v2)

def scalar_halo_velocity(r, V_halo_inf, R_s):
    """
    Velocity from a Spherical Scalar Halo (Pseudo-Isothermal Sphere).
    Corresponds to a density profile rho(r) = rho_0 / (1 + (r/Rs)^2).
    Derived from the scalar wave equation solution for a condensate.
    
    Mass enclosed M(r) ~ r - Rs * arctan(r/Rs)
    v^2(r) = V_inf^2 * (1 - (Rs/r) * arctan(r/Rs))
    """
    r_safe = np.maximum(r, 1e-6)
    v2 = V_halo_inf**2 * (1.0 - (R_s / r_safe) * np.arctan(r_safe / R_s))
    return v2

def unified_velocity_model(r, V_disk_scale, R_d, V_halo_inf, R_s):
    """ Combined model: Baryonic Disk + Scalar Vacuum Halo """
    v2_disk = freeman_disk_velocity(r, V_disk_scale, R_d)
    v2_halo = scalar_halo_velocity(r, V_halo_inf, R_s)
    return np.sqrt(v2_disk + v2_halo)

# --- CURVE FITTING ---

# Initial Parameter Guesses
# V_disk_scale ~ 200, Rd ~ 1.5 kpc
# V_halo_inf ~ 130, Rs ~ 3.0 kpc
p0 = [200.0, 1.5, 130.0, 3.0]

# Bounds to force physical values (positive scales)
bounds = ([10, 0.1, 10, 0.1], [1000, 10.0, 500, 20.0])

# Perform Least-Squares Fit
popt, pcov = curve_fit(unified_velocity_model, r_dat, v_dat, p0=p0, sigma=v_err, bounds=bounds)
V_d_fit, R_d_fit, V_h_fit, R_s_fit = popt

print("Fit Successful.")
print(f"Baryonic Disk: Scale={V_d_fit:.1f}, Scale Length Rd={R_d_fit:.2f} kpc")
print(f"Scalar Halo:   V_inf={V_h_fit:.1f}, Scale Radius Rs={R_s_fit:.2f} kpc")

# --- PLOTTING ---

# Smooth radius array for plotting curves
r_plot = np.linspace(0.1, 22.0, 200)

# Calculate components using fitted parameters
v_tot_plot = unified_velocity_model(r_plot, *popt)
v_disk_plot = np.sqrt(freeman_disk_velocity(r_plot, V_d_fit, R_d_fit))
v_halo_plot = np.sqrt(scalar_halo_velocity(r_plot, V_h_fit, R_s_fit))

fig, ax = plt.subplots(figsize=(10, 6))

# 1. Observations
ax.errorbar(r_dat, v_dat, yerr=v_err, fmt='ko', label='Observational Data (NGC 6503)', 
            capsize=3, alpha=0.6, markersize=5)

# 2. Model Components
ax.plot(r_plot, v_disk_plot, 'b--', linewidth=2, 
        label=f'Baryonic Disk (3D Freeman)\n$R_d={R_d_fit:.2f}$ kpc')
ax.plot(r_plot, v_halo_plot, 'g:', linewidth=2.5, 
        label=f'Scalar Vacuum Halo\n$R_s={R_s_fit:.2f}$ kpc')

# 3. Total Fit
ax.plot(r_plot, v_tot_plot, 'r-', linewidth=3, alpha=0.9, 
        label='Total Emergent Model')

# Styling
ax.set_title(r"Universal Rotation Curve Fit: NGC 6503", fontsize=14, weight='bold')
ax.set_xlabel(r"Galactocentric Radius $r$ [kpc]", fontsize=12)
ax.set_ylabel(r"Circular Velocity $v$ [km/s]", fontsize=12)
ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_xlim(0, 22)
ax.set_ylim(0, 140)

# Annotation Box
textstr = '\n'.join((
    r'$\mathbf{Physics\ Model\ (3+1)D:}$',
    r'Disk: $I_0 K_0$ Bessel (Exact 3D)',
    r'Vacuum: Scalar Condensate',
    r'$\rho_{vac} \sim (1 + (r/R_s)^2)^{-1}$'
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
ax.text(0.03, 0.96, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
output_file = "fig_rotation_3d.png"
plt.savefig(output_file, dpi=150)
print(f"Rotation curve plot saved to {output_file}")
plt.close()
