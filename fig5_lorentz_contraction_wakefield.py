import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# =============================================================================
# CODE METADATA & PHYSICS ESSENCE 
# =============================================================================
# SCRIPT: fig5_lorentz_contraction_wakefield.py
# PURPOSE: Visualization of Relativistic Field Contraction (Wakefield).
#
# THEORETICAL BACKGROUND:
# A static scalar source generates a spherically symmetric potential:
# Phi(r) ~ 1/r.
#
# When the source moves at relativistic velocity v = beta * c, the field 
# configuration in the laboratory frame is obtained via Lorentz transformation 
# of the coordinates:
# z' = gamma * (z - v*t)
#
# Since the scalar field Phi is a Lorentz scalar (Phi(x) = Phi'(x')), 
# the equipotential surfaces are compressed along the direction of motion 
# by the Lorentz factor gamma = 1 / sqrt(1 - beta^2).
#
# This creates a "pancake" field shape perpendicular to the velocity vector,
# modifying the interaction cross-section in high-energy scattering.
# =============================================================================

# --- CONFIGURATION ---
L_z, L_y = 40.0, 40.0   # Domain Size
N = 512                 # Resolution
beta_rel = 0.92         # Relativistic velocity (v/c)

# Grid Setup
z = np.linspace(-L_z/2, L_z/2, N)
y = np.linspace(-L_y/2, L_y/2, N)
Z, Y = np.meshgrid(z, y)

# Physics Constants
G_coupling = 1.0
R_core = 2.0 # Softening radius (Soliton size)

# --- FIELD CALCULATION ---
def calculate_potential(beta, Z_grid, Y_grid):
    """
    Calculates the potential Phi(z,y) for a source moving along Z.
    Using Lorentz transformation of coordinates from rest frame.
    """
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    
    # Effective coordinate in rest frame (at t=0)
    # Z' = gamma * Z
    # R' = sqrt(Y^2 + Z'^2)
    
    Z_prime = gamma * Z_grid
    R_prime = np.sqrt(Y_grid**2 + Z_prime**2)
    
    # Potential Phi ~ 1 / sqrt(R'^2 + core^2)
    # We use a generalized Lorentzian/Soler profile for the core
    Phi = G_coupling / np.sqrt(R_prime**2 + R_core**2)
    
    return Phi, gamma

# 1. Static Case
Phi_static, gamma_static = calculate_potential(0.0, Z, Y)

# 2. Relativistic Case
Phi_rel, gamma_rel = calculate_potential(beta_rel, Z, Y)

# --- VISUALIZATION ---
print(f"Rendering Figure 5... Gamma factor: {gamma_rel:.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(wspace=0.25)

# --- PANEL 1: STATIC ---
# Heatmap
im1 = ax1.imshow(Phi_static, extent=[-L_z/2, L_z/2, -L_y/2, L_y/2], 
                 origin='lower', cmap='inferno', vmin=0, vmax=np.max(Phi_static))

# Contours (Isotropic)
levels = np.linspace(0.05, np.max(Phi_static)*0.8, 5)
ax1.contour(Z, Y, Phi_static, levels=[levels[1]], colors='white', linestyles='--', linewidths=1.0, alpha=0.6)

ax1.set_title(r"Static Scalar Potential ($\beta=0$)", fontweight='bold', fontsize=12)
ax1.set_xlabel("Longitudinal z")
ax1.set_ylabel("Transverse y")
ax1.text(-15, 15, "Isotropic 1/r", color='white', fontweight='bold', fontsize=10)

# --- PANEL 2: RELATIVISTIC ---
# Heatmap
im2 = ax2.imshow(Phi_rel, extent=[-L_z/2, L_z/2, -L_y/2, L_y/2], 
                 origin='lower', cmap='inferno', vmin=0, vmax=np.max(Phi_static)) # Same scale for comparison

# Contours (Flattened)
# We draw a specific contour to highlight the shape
ax2.contour(Z, Y, Phi_rel, levels=[levels[1]], colors='cyan', linestyles='--', linewidths=1.5)

ax2.set_title(r"Relativistic Potential ($\beta=0.92$)", fontweight='bold', fontsize=12)
ax2.set_xlabel("Longitudinal z (Motion Direction)")
ax2.set_ylabel("Transverse y")

# Annotation
ax2.text(-18, 16, "Lorentz Contraction", color='cyan', fontweight='bold', fontsize=11)
ax2.text(-18, 14, r"$L' = L / \gamma$", color='cyan', fontsize=10)

# --- INSET: LONGITUDINAL PROFILE ---
# We want to show the cut along Y=0
ax_ins = inset_axes(ax2, width="40%", height="30%", loc='lower right', borderpad=1)

mid_y = N // 2
profile_static = Phi_static[mid_y, :]
profile_rel = Phi_rel[mid_y, :]

ax_ins.plot(z, profile_static, 'r-', label=r'$\beta=0$', linewidth=1.5, alpha=0.7)
ax_ins.plot(z, profile_rel, 'c--', label=r'$\beta=0.92$', linewidth=1.5)

ax_ins.set_title("Longitudinal Profile", color='white', fontsize=8)
ax_ins.set_xlim(-15, 15)
ax_ins.set_yticks([])
ax_ins.tick_params(axis='x', colors='white', labelsize=7)
ax_ins.set_facecolor('black')
ax_ins.spines['bottom'].set_color('white')
ax_ins.spines['top'].set_color('white') 
ax_ins.spines['left'].set_color('white')
ax_ins.spines['right'].set_color('white')

# --- COLORBAR ---
cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.025, pad=0.04)
cbar.set_label(r"Potential Amplitude $\Phi$", fontsize=10)

# --- GLOBAL TITLE ---
plt.suptitle("Figure 5: Lorentz Contraction. Comparison of static and relativistic scalar potentials.", 
             fontsize=14, fontweight='bold', y=0.96)

# Save
filename = "fig5_wakefield_contraction.png"
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"Figure saved to {filename}")
plt.show()
