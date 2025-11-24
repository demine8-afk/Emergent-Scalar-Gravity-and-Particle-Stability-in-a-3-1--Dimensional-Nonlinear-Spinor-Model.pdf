"""
SCRIPT 03: 3D WAVE INTERFERENCE (DOUBLE SLIT)
---------------------------------------------
Purpose:
Demonstrates the emergent wave-particle duality of the model. 
A localized soliton (particle) is fired at a barrier with two slits.
The script solves the evolution in full 3D space to show diffraction 
and interference fringes, confirming the field nature of the "particle".

Key Features:
- Non-cubic grid (Optimized for propagation along X).
- Static Potential Barrier implementation.
- Visualization of Intensity map and Far-field profile.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftfreq
import matplotlib.gridspec as gridspec

# --- SIMULATION DOMAIN ---
# We use a longer domain in X to capture propagation, 
# and sufficient Y width for diffraction spreading.
Nx, Ny, Nz = 200, 128, 16
Lx, Ly = 40.0, 40.0
Lz = 10.0 # Shallow Z depth for 3D slab

# Coordinates
x = np.linspace(-10, 30, Nx, endpoint=False) # Propagates from -10 to +30
y = np.linspace(-20, 20, Ny, endpoint=False)
z = np.linspace(-Lz/2, Lz/2, Nz, endpoint=False)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# --- POTENTIAL BARRIER (THE SLITS) ---
# 1. Wall Position: X = 5.0, Thickness = 1.0
V_wall = np.zeros_like(X)
mask_wall = (np.abs(X - 5.0) < 0.5)

# 2. Slit Geometry: Y_centers = +/- 3.0, Width = 1.2
# We knock out holes in the wall mask
mask_slits = (np.abs(np.abs(Y) - 3.0) < 0.6)

# 3. Apply Potential Height
V_height = 20.0 # High barrier (quasi-infinite)
V_wall[mask_wall & (~mask_slits)] = V_height

# --- INITIAL PACKET ---
# Gaussian wave packet starting at X=-5, traveling right (+X)
k_x = 1.5
Psi = np.exp(-0.5 * ((X+5)**2 + Y**2 + Z**2)/2.0) * np.exp(1j * k_x * X)

# --- SPECTRAL PROPAGATOR ---
# Time settings
dt = 0.04
t_max = 18.0 # Time to reach X ~ 25

# Frequency Grid
kx = 2 * np.pi * fftfreq(Nx, Lx/Nx)
ky = 2 * np.pi * fftfreq(Ny, Ly/Ny)
kz = 2 * np.pi * fftfreq(Nz, Lz/Nz)
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

# Kinetic Energy E_k = sqrt(k^2 + m^2)
# Note: Using scalar propagator approximation for diffraction demonstration 
# (Dirac spinor components diffract identically in this potential)
Ek = np.sqrt(KX**2 + KY**2 + KZ**2 + 1.0)
U_op = np.exp(-1j * Ek * dt)

# --- EVOLUTION LOOP ---
steps = int(t_max / dt)
print(f"Starting Interference Sim: Steps={steps}, Grid={Nx}x{Ny}x{Nz}")

for step in range(steps + 1):
    # 1. Potential Step
    Psi *= np.exp(-0.5j * V_wall * dt)
    
    # 2. Kinetic Step (FFT)
    Psi_k = fftn(Psi)
    Psi_k *= U_op
    Psi = ifftn(Psi_k)
    
    # 3. Potential Step
    Psi *= np.exp(-0.5j * V_wall * dt)
    
    if step % 50 == 0:
        print(f"Progress: {step/steps*100:.1f}%")

# --- VISUALIZATION ---
# Extract 2D slice at Z=0 for plotting
intensity = np.abs(Psi[:, :, Nz//2])**2

fig = plt.figure(figsize=(10, 9))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.2], hspace=0.15)

# Plot 1: Top-down view of intensity (X-Y plane)
ax0 = plt.subplot(gs[0])
# pcolormesh arguments: (Y, X, C) because array is indexed [x, y]
# Transpose for intuitive "Left-to-Right" propagation plot
cm = ax0.pcolormesh(y, x, intensity, cmap='magma', shading='auto', vmax=np.max(intensity)*0.8)

# Draw barrier overlay
ax0.axhline(5.0, color='cyan', lw=2, linestyle='-', alpha=0.7)
# Draw slit markers
ax0.plot([-20, -3.6], [5, 5], 'c', lw=4) # Left Wall segment
ax0.plot([-2.4, 2.4], [5, 5], 'c', lw=4) # Center Wall segment
ax0.plot([3.6, 20],   [5, 5], 'c', lw=4) # Right Wall segment

ax0.text(-18, 6.5, "Double Slit Barrier", color='cyan', fontsize=12, fontweight='bold')
ax0.set_ylabel("Propagation Distance X", fontsize=11)
ax0.set_title("3D Soliton Interference Experiment (Slice Z=0)", fontweight='bold', fontsize=14)
plt.colorbar(cm, ax=ax0, label="Probability Density")

# Plot 2: Far-field Profile
ax1 = plt.subplot(gs[1], sharex=ax0)
# Take a cut at X = 25 (Far field)
cut_idx = int(Nx * (25 - (-10)) / 40.0) 
profile = intensity[cut_idx, :]

ax1.plot(y, profile, 'w-', lw=2, label='Intensity Profile at X=25')
ax1.fill_between(y, 0, profile, color='orange', alpha=0.4)
ax1.set_facecolor('#101010') # Dark background
ax1.set_ylabel("Intensity", fontsize=11)
ax1.set_xlabel("Transverse Position Y", fontsize=11)
ax1.grid(True, color='white', alpha=0.1)
ax1.legend(loc='upper right', frameon=False, labelcolor='white')

# Add annotation for fringes
peak_y = 0
ax1.annotate('Central Max', xy=(0, profile[Ny//2]), xytext=(5, profile[Ny//2]+0.05),
             arrowprops=dict(facecolor='white', shrink=0.05), color='white')

plt.tight_layout()
output_file = "fig_interference_recovered.png"
plt.savefig(output_file, dpi=150)
print(f"Interference pattern saved to {output_file}")
plt.close()
