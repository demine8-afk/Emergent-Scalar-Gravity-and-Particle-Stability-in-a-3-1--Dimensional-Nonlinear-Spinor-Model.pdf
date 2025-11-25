import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CODE METADATA & PHYSICS ESSENCE
# =============================================================================
# SCRIPT: fig6_gravitational_lensing.py
# PURPOSE: Demonstration of Analog Gravitational Lensing via Scalar Potential.
#
# THEORETICAL BACKGROUND:
# According to Eq. (7) in the preprint, the coupling of the spinor field to the 
# scalar field Phi creates an effective metric or, equivalently, a position-dependent
# refractive index: n(r) ~ 1 - Phi(r).
#
# A localized scalar potential well (Phi < 0) acts as a region of higher 
# refractive index (convex lens) for the matter waves passing through it.
#
# METHOD:
# We solve the paraxial wave equation (Schrodinger limit of Eq. 5) using the 
# Split-Step Fourier Method (SSFM).
# Hamiltonian: H = p^2/(2m) + V_opt(r)
# where V_opt is the optical potential proportional to the scalar field Phi.
# =============================================================================

# --- CONFIGURATION ---
L = 40.0                # Domain Size (from -20 to 20)
N = 512                 # Grid Resolution
dt = 0.05               # Time step
steps = 160             # Number of time steps to reach focal point

# Physics Parameters
mass = 1.0              # Particle mass
k0 = 3.0                # Initial momentum (propagation along Z)
V_depth = -1.5          # Potential depth (Phi(0) in caption)
R_lens = 3.5            # Radius of the scalar potential well

# Grid Setup (2D Slice: X=Transverse, Z=Propagation)
x = np.linspace(-L/2, L/2, N)
z = np.linspace(-L/2, L/2, N)
X, Z = np.meshgrid(x, z)

# --- INITIALIZATION ---

# 1. The "Lens" (Scalar Potential Phi)
# Gaussian well representing the scalar field of a massive body
# Phi(r) = V_depth * exp(-r^2 / R^2)
R_sq = X**2 + Z**2
Potential = V_depth * np.exp(-R_sq / (R_lens**2))

# 2. The "Probe" (Matter Wave)
# Wide Gaussian packet starting at negative Z, moving towards +Z
z_start = -12.0
width_transverse = 6.0  # Wide enough to act as a wavefront
width_longitudinal = 3.0
Psi = np.exp(-((X)**2) / (2 * width_transverse**2) - ((Z - z_start)**2) / (2 * width_longitudinal**2))
Psi = Psi * np.exp(1j * k0 * Z) # Add momentum

# Normalize
Psi /= np.sqrt(np.sum(np.abs(Psi)**2))

# --- TIME EVOLUTION (Split-Step Fourier) ---
# Propagator in k-space (Kinetic term)
kx = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
kz = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
KX, KZ = np.meshgrid(kx, kz)
K2 = KX**2 + KZ**2
Evolution_K = np.exp(-1j * (K2 / (2*mass)) * dt)

# Propagator in real-space (Potential term)
Evolution_V = np.exp(-1j * Potential * dt)

print(f"Simulating lensing effect... Depth={V_depth}, Steps={steps}")

for t in range(steps):
    # 1. Half-step Potential
    Psi = Psi * Evolution_V
    
    # 2. Full-step Kinetic (FFT -> Phase -> IFFT)
    Psi_k = np.fft.fft2(Psi)
    Psi_k = Psi_k * Evolution_K
    Psi = np.fft.ifft2(Psi_k)

# Calculate Energy Density
Density = np.abs(Psi)**2

# --- VISUALIZATION ---
fig, ax = plt.subplots(figsize=(10, 8))

# Plot Density Heatmap
im = ax.imshow(Density, extent=[-L/2, L/2, -L/2, L/2], 
               origin='lower', cmap='inferno', interpolation='bicubic')

# Overlay: The Potential Well (The "Lens")
# Represented by dashed cyan circles as in the screenshot
levels = [V_depth * 0.2, V_depth * 0.5, V_depth * 0.8]
# Note: Potential is negative, so we contour absolute values or just the shape
lens_contour = ax.contour(X, Z, -Potential, levels=[-levels[0], -levels[1], -levels[2]], 
                          colors='cyan', linestyles='--', linewidths=1.0, alpha=0.8)

# Find Max Intensity (Focal Point)
max_idx = np.unravel_index(np.argmax(Density), Density.shape)
z_focal = z[max_idx[0]]
x_focal = x[max_idx[1]]

# Annotation: Focal Point
ax.scatter([x_focal], [z_focal], color='white', marker='x', s=50, linewidth=2, zorder=10)
ax.text(x_focal + 1, z_focal, "FOCAL POINT", color='white', fontsize=8, fontweight='bold', va='center')

# Grid and Axes Styling
ax.set_facecolor('black')
ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
ax.axvline(0, color='gray', linestyle=':', alpha=0.3)

ax.set_xlabel("X (Transverse)")
ax.set_ylabel("Z (Propagation Direction)")

# FIXED TITLE SYNTAX: Using raw f-string (rf) to handle LaTeX backslashes correctly
ax.set_title(rf"Gravitational Lensing (Convergence of Matter Waves)" + "\n" + 
             rf"Scalar Potential $\Phi(0)={V_depth}$ acts as Convex Lens", 
             fontsize=12, fontweight='bold', pad=10)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Energy Density", rotation=90, labelpad=10)

# Limits matching the screenshot approx
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)

filename = "fig6_gravitational_lensing.png"
plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Figure saved to {filename}")
plt.show()
