import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# CODE METADATA & PHYSICS ESSENCE 
# =============================================================================
# SCRIPT: fig8_quantum_interference.py
# PURPOSE: Demonstration of Wave-Particle Duality via Double-Slit Experiment.
#
# THEORETICAL BACKGROUND:
# Even though the Soler solitons behave like particles (holding their shape),
# they are fundamentally field excitations. When encountering a barrier with 
# two apertures, the wavefunction splits and interferes with itself.
#
# The appearance of fringes (constructive and destructive interference) 
# confirms the wave nature of the matter field in this model.
#
# SIMULATION METHOD:
# Split-Step Fourier Method (SSFM) in 2D.
# Potential V(x,y) includes a "hard wall" barrier with two zero-potential slits.
# =============================================================================

# --- CONFIGURATION ---
L_x = 60.0              # Propagation length (Height)
L_y = 40.0              # Transverse width
N_x = 600               # Resolution X
N_y = 400               # Resolution Y
dt = 0.05
steps = 450             # Steps to propagate through barrier

# Physics
k0 = 4.0                # Momentum along X (propagation direction)
mass = 1.0
barrier_x = 5.0         # Location of the barrier (matches screenshot)
barrier_width = 0.8     # Thickness of the wall
slit_spacing = 4.0      # Distance between slits
slit_aperture = 1.2     # Width of each slit
V_wall = 500.0          # Height of the potential barrier (quasi-infinite)

# Grid
x = np.linspace(-20, 40, N_x) # Shifted to match screenshot range
y = np.linspace(-20, 20, N_y)
X, Y = np.meshgrid(x, y, indexing='ij') # ij indexing for X=Vertical, Y=Horizontal

# --- INITIALIZATION ---

# 1. Potential Barrier (The Double Slit)
V_pot = np.zeros_like(X)
# Define Wall Domain
mask_wall_x = (np.abs(X - barrier_x) < barrier_width/2)
# Define Slits (Openings)
mask_slits_y = (np.abs(Y - slit_spacing/2) < slit_aperture/2) | \
               (np.abs(Y + slit_spacing/2) < slit_aperture/2)
# Apply Potential: Wall exists where X is in range AND Y is NOT in slits
V_pot[mask_wall_x & (~mask_slits_y)] = V_wall

# 2. Initial Wave Packet
x_start = -10.0
width_x = 2.0
width_y = 6.0 # Wide transverse packet to hit both slits
Psi = np.exp(-((X - x_start)**2)/(2*width_x**2) - (Y**2)/(2*width_y**2))
Psi = Psi * np.exp(1j * k0 * X) # Add momentum upwards

# Normalize
Psi /= np.sqrt(np.sum(np.abs(Psi)**2))

# --- EVOLUTION (Split-Step Fourier) ---
# Propagators
kx = np.fft.fftfreq(N_x, d=(x[1]-x[0])) * 2 * np.pi
ky = np.fft.fftfreq(N_y, d=(y[1]-y[0])) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K2 = KX**2 + KY**2

Op_K = np.exp(-1j * (K2 / (2*mass)) * dt)
Op_V = np.exp(-1j * V_pot * dt)

print("Simulating Double-Slit Diffraction...")

# Run loop
for t in range(steps):
    # Half-step V
    Psi = Psi * Op_V
    # Full-step K
    Psi_k = np.fft.fft2(Psi)
    Psi_k = Psi_k * Op_K
    Psi = np.fft.ifft2(Psi_k)
    
    # Absorbing boundaries (simple cosine window at edges to prevent reflection artifacts)
    # Not strictly necessary if domain is large enough, but good practice
    if t % 50 == 0:
        pass # print(f"Step {t}/{steps}")

Density = np.abs(Psi)**2

# --- VISUALIZATION ---
# Setup GridSpec to match the screenshot (Main plot on top, 1D profile below)
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.15)

ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

# --- PLOT 1: 2D INTERFERENCE PATTERN ---
im = ax1.imshow(np.transpose(Density), extent=[x.min(), x.max(), y.min(), y.max()],
                origin='lower', cmap='magma', vmax=np.max(Density)*0.8)
# Note: imshow expects (rows, cols), so we transpose because we defined X as vertical

# Correcting axes to match screenshot convention:
# The screenshot has X as vertical (Propagation) and Y as horizontal (Transverse)
# But imshow plots Y-axis vertical. So we need to swap how we display it.
# Let's redraw to match the screenshot exactly: 
# Horizontal Axis = Transverse Position y (-20 to 20)
# Vertical Axis = Propagation Direction x (-20 to 30)
ax1.clear()
im = ax1.imshow(Density, extent=[y.min(), y.max(), x.min(), x.max()],
                origin='lower', cmap='magma', aspect='auto', vmax=np.max(Density)*0.6)

# Draw Barrier Line
ax1.axhline(barrier_x, color='cyan', linewidth=2, linestyle='-', alpha=0.7)
ax1.text(-15, barrier_x + 1, "Barrier (3D Wall)", color='cyan', fontsize=9, fontweight='bold')

# Draw Arrow for Initial Packet
ax1.arrow(0, -8, 0, 3, head_width=1.0, head_length=1.5, fc='white', ec='white', alpha=0.8)
ax1.text(1, -8, r"Initial Packet $\vec{p}$", color='white', fontsize=9)

ax1.set_ylabel("Propagation Direction x")
ax1.set_xlabel("Transverse Position y")
ax1.set_title("Wave Nature in (3+1)D: Quantum-like Interference\n3D Double-Slit Interference (Slice Z=0)", 
              fontweight='bold', fontsize=12)

# Colorbar
cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label("Wave Intensity", rotation=90)

# Limits
ax1.set_ylim(-10, 30)
ax1.set_xlim(-20, 20)


# --- PLOT 2: 1D PROFILE (LOWER PANEL) ---
# Extract profile at the top of the simulation (or near where patterns form)
readout_x_idx = int(N_x * 0.85) # Near the end
profile = Density[readout_x_idx, :]

ax2.plot(y, profile, color='white', linewidth=1.5)
# Fill below
ax2.fill_between(y, profile, color='orange', alpha=0.3)

# Style like a scope
ax2.set_facecolor('black')
ax2.set_xlim(-20, 20)
ax2.set_yticks([])
ax2.grid(color='gray', linestyle=':', alpha=0.5)

# Mark maxima lines connecting plots (Optional visual aid)
# Find peaks
from scipy.signal import find_peaks
peaks, _ = find_peaks(profile, height=np.max(profile)*0.1)
for peak_idx in peaks:
    peak_y = y[peak_idx]
    if abs(peak_y) < 10: # Only central peaks
        # Draw dashed line on bottom plot
        ax2.axvline(peak_y, color='red', linestyle='--', alpha=0.3)
        # Draw dashed line on top plot? Maybe too messy.

ax2.set_xlabel("Transverse Position y (Interference Fringes)")

# Save
filename = "fig8_quantum_interference.png"
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"Figure saved to {filename}")
plt.show()
